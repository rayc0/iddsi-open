#!/usr/bin/env python3
"""consume_release.py — run the documented "how to consume this" paths.

One command per path, all of them reading the same YAML
(``configs/consume/hub_release.yaml``) so the pipeline, the CI smoke test and
the docs cannot drift apart:

    # (a) the capture/eval pipeline: pull the released artifacts and evaluate
    python tools/consume_release.py --config configs/consume/hub_release.yaml \\
        --step eval --output-dir out/eval

    # (a) the capture pipeline: resolve the released model source and run the
    #     honest placeholder estimate with full provenance
    python tools/consume_release.py --config configs/consume/hub_release.yaml \\
        --step capture --output-dir out/capture

    # (b) the CI smoke test: both of the above plus the `expect:` assertions
    python tools/consume_release.py --config configs/consume/hub_release.yaml \\
        --step smoke --output-dir out/smoke

Endpoint selection (an operator-supplied mirror on a CN network, the real endpoint
elsewhere) is
made by ``eval.hub_source.resolve_endpoint`` and is recorded in the receipt, so
a run can always be traced to the bytes it consumed. Nothing here downloads a
classifier: the release publishes no weights.

Exit status is non-zero when an ``expect:`` assertion fails or when the release
shape has drifted from what the docs claim — a smoke test that cannot fail is
not a test.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAINER_DIR = _REPO_ROOT / "trainer"
for _path in (str(_REPO_ROOT), str(_TRAINER_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from eval import hub_source  # noqa: E402
from eval.cli import run_eval  # noqa: E402

STEPS = ("eval", "capture", "smoke")


def _add_hub_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hub-repo", default=None, help="override hub.repo")
    parser.add_argument("--hub-revision", default=None, help="override hub.revision")
    parser.add_argument(
        "--hub-endpoint",
        default=None,
        help=(
            "pin the endpoint. Default order: IDDSI_HF_ENDPOINT, HF_ENDPOINT, "
            "probe the canonical endpoint, probe the operator-supplied mirror"
        ),
    )
    parser.add_argument("--hub-cache-dir", default=None, help="override hub.cache_dir")
    parser.add_argument(
        "--require-network",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "force a fresh download so the named endpoint provably serves the "
            "bytes (default: hub.require_network from the config)"
        ),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _receipt_dict(receipt: Any) -> dict[str, Any] | None:
    return receipt.to_dict() if receipt is not None else None


def step_eval(cfg: hub_source.ConsumeConfig, out: Path, hub: dict[str, Any]) -> dict[str, Any]:
    """(a) pull the released predictions + threshold and evaluate them."""
    summary = run_eval(
        predictions=cfg.artifact("predictions"),
        threshold=cfg.artifact("threshold"),
        output_dir=out,
        bootstrap_samples=int(cfg.eval.get("bootstrap_samples", 1000)),
        bootstrap_seed=int(cfg.eval.get("bootstrap_seed", 2026)),
        hub=hub,
    )
    result = summary["result"]
    receipt = json.loads(Path(summary["hub_receipt"]).read_text(encoding="utf-8")) if (
        summary.get("hub_receipt")
    ) else None
    return {
        "step": "eval",
        "artifacts": {
            "predictions": cfg.artifact("predictions"),
            "threshold": cfg.artifact("threshold"),
        },
        "n": result.get("n"),
        "metrics": result.get("metrics", result),
        "hub_receipt": receipt,
        "outputs": {k: v for k, v in summary.items() if k != "result"},
    }


def step_capture(cfg: hub_source.ConsumeConfig, out: Path, hub: dict[str, Any]) -> dict[str, Any]:
    """(a) resolve the released model source through the capture pipeline."""
    from crowd.inference import (  # noqa: PLC0415
        predict_iddsi_level,
        resolve_model_source,
    )

    spec = cfg.artifact("model_source")
    source = resolve_model_source(
        spec,
        endpoint=hub.get("endpoint"),
        cache_dir=hub.get("cache_dir"),
        require_network=bool(hub.get("require_network")),
    )
    print(source.provenance_line())
    print(
        f"weights published for {source.repo}@{source.revision}: "
        f"{source.weights_published} "
        f"({len(source.published_weight_files)} file(s) matching "
        f"{', '.join(hub_source.WEIGHT_SUFFIXES)})"
    )
    print(f"released architecture: {json.dumps(source.architecture, sort_keys=True)}")

    # Run the estimate on a synthetic 8x8 RGB plate so the capture path is
    # exercised end to end without a camera image. The result must be the
    # placeholder, never a level.
    photo = out / "synthetic_plate.png"
    photo.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image  # noqa: PLC0415

        Image.new("RGB", (8, 8), (120, 80, 40)).save(photo)
    except ImportError:  # pragma: no cover - Pillow is a declared dependency
        photo = None

    estimate = None
    if photo is not None:
        result = predict_iddsi_level(
            photo,
            source=spec,
            endpoint=hub.get("endpoint"),
            cache_dir=hub.get("cache_dir"),
            require_network=bool(hub.get("require_network")),
        )
        estimate = {
            "status": result.status,
            "is_placeholder": result.is_placeholder,
            "predicted_level": result.predicted_level,
            "probabilities": {str(k): v for k, v in result.probabilities.items()},
            "detail": result.detail,
            "extras": result.extras,
        }
        print(
            f"capture estimate: status={result.status} "
            f"predicted_level={result.predicted_level} "
            f"is_placeholder={result.is_placeholder}"
        )
        print(f"capture detail: {result.detail}")

    return {
        "step": "capture",
        "model_source": spec,
        "provenance": {
            "repo": source.repo,
            "revision": source.revision,
            "endpoint": source.endpoint,
            "endpoint_provenance": source.endpoint_provenance,
            "resolved_commit": source.resolved_commit,
            "weights_published": source.weights_published,
            "published_weight_files": list(source.published_weight_files),
            "architecture": source.architecture,
            "calibrated_threshold": source.calibrated_threshold,
        },
        "estimate": estimate,
    }


def step_smoke(
    cfg: hub_source.ConsumeConfig, out: Path, hub: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """(b) the CI smoke test: both paths plus every ``expect:`` assertion."""
    failures: list[str] = []
    eval_part = step_eval(cfg, out / "eval", hub)
    capture_part = step_capture(cfg, out / "capture", hub)

    measured = {"n": eval_part["n"], **eval_part["metrics"]}
    failures.extend(hub_source.check_expectations(measured, cfg.expect))

    # The eval receipt must prove the bytes came off a Hub endpoint. A receipt
    # that is entirely cache hits names an endpoint that never answered, which
    # is the defect H-HUBCACHE records; a smoke test that never touched the Hub
    # proves nothing, so it fails rather than reporting green.
    receipt = eval_part.get("hub_receipt") or {}
    if not receipt:
        failures.append("eval: no hub receipt was written; nothing was consumed from the Hub")
    else:
        if receipt.get("all_from_cache"):
            failures.append(
                "eval: every artifact was already in the local cache, so the named "
                f"endpoint {receipt.get('endpoint')} served nothing. Set "
                "hub.require_network: true (or use an empty cache) so the smoke "
                "test actually consumes the release"
            )
        if not receipt.get("resolved_commit"):
            failures.append(
                "eval: the endpoint did not resolve a commit for "
                f"{receipt.get('repo')}@{receipt.get('revision')}; the receipt "
                "cannot be traced to a published revision"
            )
        if receipt.get("commit_mismatch"):
            failures.append(
                "eval: the served snapshot commit differs from the commit the "
                "endpoint reported; the bytes are not the pinned revision"
            )

    provenance = capture_part["provenance"]
    if provenance["weights_published"] is None:
        failures.append(
            "capture: could not determine whether the release publishes weights; "
            "the smoke test refuses to pass on an unknown release shape"
        )
    elif provenance["weights_published"]:
        failures.append(
            "capture: the Hub now publishes weight file(s) "
            f"{provenance['published_weight_files']} — the release shape the docs "
            "describe (weights withheld) has drifted; update "
            "hf/HOW_TO_CONSUME.md and configs/consume/hub_release.yaml before "
            "trusting this smoke test"
        )

    estimate = capture_part.get("estimate") or {}
    if estimate.get("predicted_level") is not None:
        failures.append(
            f"capture: the placeholder path emitted a level ({estimate['predicted_level']}); "
            "with no published weights it must not"
        )

    return {"step": "smoke", "eval": eval_part, "capture": capture_part}, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the documented Hub-consumption paths for this release",
        epilog="steps: " + ", ".join(STEPS),
    )
    parser.add_argument("--config", required=True, help="path to configs/consume/*.yaml")
    parser.add_argument("--step", required=True, choices=STEPS)
    parser.add_argument("--output-dir", required=True)
    _add_hub_overrides(parser)
    args = parser.parse_args(argv)

    cfg = hub_source.load_consume_config(args.config)
    for name, value in (
        ("repo", args.hub_repo),
        ("revision", args.hub_revision),
        ("endpoint", args.hub_endpoint),
        ("cache_dir", args.hub_cache_dir),
    ):
        if value:
            cfg = dataclasses.replace(cfg, **{name: value})
    if args.require_network is not None:
        cfg = dataclasses.replace(cfg, require_network=args.require_network)

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    hub = cfg.hub_kwargs()
    print(
        f"consume: config={cfg.path} step={args.step} repo={cfg.repo}@{cfg.revision} "
        f"endpoint={cfg.endpoint or '(auto)'} cache_dir={cfg.cache_dir or '(default)'} "
        f"require_network={cfg.require_network}"
    )

    failures: list[str] = []
    try:
        if args.step == "eval":
            payload = step_eval(cfg, out, hub)
        elif args.step == "capture":
            payload = step_capture(cfg, out, hub)
        else:
            payload, failures = step_smoke(cfg, out, hub)
    except hub_source.HubUnavailable as exc:
        # Fail closed and say why: no endpoint, no hub library, no file list.
        print(f"CONSUME FAILED: {exc}", file=sys.stderr)
        _write_json(
            out / "consume_summary.json",
            {"step": args.step, "error": f"{type(exc).__name__}: {exc}"},
        )
        return 2
    except Exception as exc:  # noqa: BLE001 - an unreachable endpoint raises here
        print(
            f"CONSUME FAILED: could not pull {cfg.repo}@{cfg.revision} via "
            f"{cfg.endpoint or 'the auto-resolved endpoint'}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        _write_json(
            out / "consume_summary.json",
            {"step": args.step, "error": f"{type(exc).__name__}: {exc}"},
        )
        return 2

    _write_json(out / "consume_summary.json", payload)
    if args.step == "smoke":
        _write_json(
            out / "smoke_failures.json",
            {"failures": failures, "passed": not failures},
        )
        if failures:
            for line in failures:
                print(f"SMOKE FAIL: {line}", file=sys.stderr)
            print(f"smoke test FAILED ({len(failures)} assertion(s))", file=sys.stderr)
            return 1
        print(f"smoke test PASSED: {len(cfg.expect)} assertions + release-shape checks")
    print(f"wrote {out / 'consume_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
