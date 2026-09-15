"""CLI and orchestration for synthetic IDDSI-Open image generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .judge import (
    DEFAULT_OPENAI_ENDPOINT,
    DEFAULT_QWEN_MODEL,
    ConsistencyJudge,
    DisabledJudge,
    JudgeThresholds,
    OpenAICompatibleJudge,
    TransformersQwenJudge,
)
from .pipelines import FALLBACK_MODEL, PRIMARY_MODEL, DiffusersConfig, DiffusersImagePipeline, ImagePipeline
from .prompt_bank import PromptSpec, build_prompt


@dataclass(frozen=True)
class GenerationConfig:
    level: int
    n: int
    out: Path
    batch_size: int = 4
    seed: int = 20260831
    width: int = 1024
    height: int = 1024
    cuisine: str = "all"
    cue_probability: float = 0.35
    split: str = "train"
    max_attempts: int | None = None
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        if self.level not in range(3, 8):
            raise ValueError("level must be from 3 through 7")
        if self.n <= 0 or self.batch_size <= 0:
            raise ValueError("n and batch_size must be positive")
        if self.width <= 0 or self.height <= 0 or self.width % 8 or self.height % 8:
            raise ValueError("width and height must be positive multiples of 8")
        if not 0.0 <= self.cue_probability <= 1.0:
            raise ValueError("cue_probability must be between 0 and 1")
        if self.split not in {"train", "val"}:
            raise ValueError("synthetic output split must be train or val, never external_test")
        if self.max_attempts is not None and self.max_attempts < self.n:
            raise ValueError("max_attempts cannot be smaller than n")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be from 1 through 100")


@dataclass(frozen=True)
class GenerationSummary:
    requested: int
    accepted: int
    rejected: int
    attempts: int
    manifest_path: str
    log_path: str
    run_id: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_append(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def _slug(value: str, limit: int = 48) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-").lower()
    return (result or "unknown")[:limit]


def _save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=quality, optimize=True, exif=b"")


def _event_row(
    *,
    event_id: str,
    media_path: Path,
    dataset_root: Path,
    spec: PromptSpec,
    seed: int,
    split: str,
    pipeline_model: str,
    judge_backend: str,
) -> dict[str, Any]:
    relative_path = media_path.relative_to(dataset_root).as_posix()
    phone_index = int(hashlib.sha256(spec.phone_profile.encode("utf-8")).hexdigest()[:4], 16) % 100
    return {
        "schema_version": "1.0.0",
        "event_id": event_id,
        "recipe_id": f"recipe_{event_id.lower()}",
        "batch_id": f"batch_{event_id.lower()}",
        "kitchen_id": f"synthetic_{split}_generator",
        "phone_id": f"synthetic_phone_{phone_index:02d}",
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sample_kind": "food",
        "level_declared": spec.level,
        "level_tested_RD": None,
        "level_tested_SLP": None,
        "level_adjudicated": None,
        "media_files": [
            {
                "path": relative_path,
                "media_type": "image",
                "capture_role": "plate_photo",
                "sha256": _sha256(media_path),
                "bytes": media_path.stat().st_size,
            }
        ],
        "test_type": ["plate_photo"],
        "capture_conditions": {
            "operator_id": "synthetic_generator",
            "lighting_category": spec.lighting_category,
            "lighting_notes": f"Prompted synthetic condition: {spec.lighting_description}",
            "plate_category": spec.plate_category,
            "plate_notes": f"Prompted synthetic condition: {spec.plate_description}",
            "serving_temperature_c": None,
            "cuisine_tags": list(spec.dish.cuisine_tags),
        },
        "ground_truth_record": {
            "rd_assessment_id": None,
            "slp_assessment_id": None,
            "adjudication_status": "not_applicable",
            "adjudicator_code": None,
            "adjudication_notes": "Synthetic image only; no physical test was performed.",
        },
        "notes": (
            f"SYNTHETIC PIPELINE-DEVELOPMENT IMAGE ONLY. Declared prompt L{spec.level}; not physically tested. "
            f"Seed={seed}; generator={pipeline_model}; judge={judge_backend}. "
            "A visible utensil cue, if present, is staged and is not evidence of rheology or a test result."
        ),
        "split": split,
        "source": "synthetic",
        "consent_recorded": False,
        "contains_face": False,
    }


class SyntheticGenerator:
    def __init__(self, pipeline: ImagePipeline, judge: ConsistencyJudge) -> None:
        self.pipeline = pipeline
        self.judge = judge

    def run(self, config: GenerationConfig) -> GenerationSummary:
        dataset_root = config.out.resolve()
        media_dir = dataset_root / "media" / "synthetic"
        rejected_dir = dataset_root / "rejected"
        manifest_path = dataset_root / "events.jsonl"
        log_path = dataset_root / "generation_log.jsonl"
        dataset_root.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        maximum = config.max_attempts if config.max_attempts is not None else max(config.n * 3, config.n + 10)
        master_rng = random.Random(config.seed)
        accepted = rejected = attempts = 0

        while accepted < config.n and attempts < maximum:
            current_batch = min(config.batch_size, maximum - attempts, config.n - accepted)
            specs: list[PromptSpec] = []
            seeds: list[int] = []
            candidate_paths: list[Path] = []
            for offset in range(current_batch):
                candidate_seed = master_rng.randrange(0, 2**63)
                candidate_rng = random.Random(candidate_seed)
                specs.append(
                    build_prompt(
                        config.level,
                        candidate_rng,
                        cuisine=config.cuisine,
                        cue_probability=config.cue_probability,
                    )
                )
                seeds.append(candidate_seed)
                number = attempts + offset + 1
                candidate_paths.append(rejected_dir / run_id / f"candidate_{number:06d}_seed_{candidate_seed}.jpg")

            images = self.pipeline.generate(
                [spec.prompt for spec in specs],
                [spec.negative_prompt for spec in specs],
                seeds,
                width=config.width,
                height=config.height,
                short_prompts=[spec.short_prompt for spec in specs],
            )
            for spec, image, seed, candidate_path in zip(specs, images, seeds, candidate_paths):
                if accepted >= config.n:
                    break
                attempts += 1
                _save_jpeg(image, candidate_path, config.jpeg_quality)
                decision = self.judge.judge(candidate_path, spec)
                event_id: str | None = None
                final_path: Path | None = None
                if decision.accepted:
                    accepted += 1
                    event_id = f"SYN_{run_id}_{accepted:06d}"
                    final_path = media_dir / f"{event_id}__plate_photo__t1.jpg"
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    candidate_path.replace(final_path)
                    row = _event_row(
                        event_id=event_id,
                        media_path=final_path,
                        dataset_root=dataset_root,
                        spec=spec,
                        seed=seed,
                        split=config.split,
                        pipeline_model=self.pipeline.model_id,
                        judge_backend=decision.backend,
                    )
                    _jsonl_append(manifest_path, row)
                else:
                    rejected += 1
                _jsonl_append(
                    log_path,
                    {
                        "run_id": run_id,
                        "attempt": attempts,
                        "event_id": event_id,
                        "accepted": decision.accepted,
                        "source_tag": "synthetic",
                        "level_declared": spec.level,
                        "seed": seed,
                        "generator_model": self.pipeline.model_id,
                        "candidate_path": candidate_path.relative_to(dataset_root).as_posix()
                        if candidate_path.exists()
                        else None,
                        "final_path": final_path.relative_to(dataset_root).as_posix() if final_path else None,
                        "prompt_spec": {
                            "dish_slug": spec.dish.slug,
                            "dish_zh": spec.dish.name_zh,
                            "dish_en": spec.dish.description_en,
                            "level_variant": spec.level_variant,
                            "visual_guidance": spec.visual_guidance,
                            "cuisine_tags": list(spec.dish.cuisine_tags),
                            "lighting_category": spec.lighting_category,
                            "lighting_description": spec.lighting_description,
                            "plate_category": spec.plate_category,
                            "plate_description": spec.plate_description,
                            "phone_profile": spec.phone_profile,
                            "angle": spec.angle,
                            "cue": spec.cue,
                            "prompt": spec.prompt,
                            "short_prompt": spec.short_prompt,
                            "negative_prompt": spec.negative_prompt,
                        },
                        "judge": decision.to_dict(),
                    },
                )

        summary = GenerationSummary(
            requested=config.n,
            accepted=accepted,
            rejected=rejected,
            attempts=attempts,
            manifest_path=str(manifest_path),
            log_path=str(log_path),
            run_id=run_id,
        )
        _jsonl_append(dataset_root / "runs.jsonl", asdict(summary) | {"source_tag": "synthetic"})
        if accepted < config.n:
            raise RuntimeError(
                f"Accepted only {accepted}/{config.n} candidates after {attempts} attempts; "
                f"partial outputs and honest rejection logs remain in {dataset_root}"
            )
        return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True, choices=range(3, 8))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--cuisine", choices=("all", "chinese", "cantonese", "western"), default="all")
    parser.add_argument("--cue-probability", type=float, default=0.35)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--model", default=PRIMARY_MODEL)
    parser.add_argument("--fallback-model", default=FALLBACK_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--flux-steps", type=int, default=4)
    parser.add_argument("--sdxl-steps", type=int, default=2)
    parser.add_argument("--judge-backend", choices=("transformers", "openai", "none"), default="transformers")
    parser.add_argument("--judge-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_OPENAI_ENDPOINT,
        help="base URL of the OpenAI-compatible judge server (default: the future W5 Qwen3-VL endpoint on 127.0.0.1:8091)",
    )
    parser.add_argument("--descriptor-threshold", type=float, default=0.65)
    parser.add_argument("--quality-threshold", type=float, default=0.50)
    parser.add_argument("--cue-threshold", type=float, default=0.50)
    parser.add_argument(
        "--allow-observed-level-unknown",
        action="store_true",
        help="permit a null observed_level when scores pass; an explicit mismatching level still rejects",
    )
    return parser


def _make_judge(args: argparse.Namespace) -> ConsistencyJudge:
    thresholds = JudgeThresholds(
        descriptor_match=args.descriptor_threshold,
        visual_quality=args.quality_threshold,
        cue_consistency=args.cue_threshold,
        require_observed_level=not args.allow_observed_level_unknown,
    )
    if args.judge_backend == "none":
        return DisabledJudge()
    if args.judge_backend == "openai":
        return OpenAICompatibleJudge(endpoint=args.endpoint, model=args.judge_model, thresholds=thresholds)
    return TransformersQwenJudge(
        model_id=args.judge_model,
        device=args.device,
        allow_cpu=args.allow_cpu,
        thresholds=thresholds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    pipeline = DiffusersImagePipeline(
        DiffusersConfig(
            primary_model=args.model,
            fallback_model=args.fallback_model,
            device=args.device,
            allow_cpu=args.allow_cpu,
            flux_steps=args.flux_steps,
            sdxl_steps=args.sdxl_steps,
        )
    )
    config = GenerationConfig(
        level=args.level,
        n=args.n,
        out=args.out,
        batch_size=args.batch_size,
        seed=args.seed,
        width=args.width,
        height=args.height,
        cuisine=args.cuisine,
        cue_probability=args.cue_probability,
        split=args.split,
        max_attempts=args.max_attempts,
        jpeg_quality=args.jpeg_quality,
    )
    summary = SyntheticGenerator(pipeline, _make_judge(args)).run(config)
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
