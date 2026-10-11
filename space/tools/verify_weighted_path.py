"""Prove the Space's *weighted* path works, using the real SigLIP-2 tower.

The v0.7 weights are withheld, so ``space/app.py`` ships on the placeholder
path.  This script exists so that claim is testable: it builds a genuine
``OrdinalImageModel`` over the local SigLIP-2 backbone with an *untrained*
CORAL head, saves it in the checkpoint format ``space/core.py`` accepts, points
the Space at it, and runs one inference.

**The untrained head's level output is meaningless and is never reported as a
measurement.**  What this script verifies is the *plumbing*: backbone loads
offline, preprocessor loads offline, CORAL head forward pass produces a
normalised L3-L7 distribution, the rank rule decodes a level inside L3..L7, and
the abstention gate fires at the documented threshold.

CPU only -- the backbone is loaded on ``map_location="cpu"``.  No GPU is used.

    python space/tools/verify_weighted_path.py --report /tmp/weighted.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "space"), str(_REPO_ROOT / "trainer")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

BACKBONE = os.getenv("IDDSI_SPACE_BACKBONE", "/work/models_local/siglip2-base-patch16-224")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="/tmp/space_weighted_report.json")
    args = parser.parse_args()

    evidence: dict[str, object] = {"backbone": BACKBONE}

    import torch  # noqa: PLC0415

    from train.model import LEVELS, OrdinalImageModel  # noqa: PLC0415

    evidence["torch"] = torch.__version__
    evidence["levels"] = list(LEVELS)

    if not Path(BACKBONE).is_dir():
        evidence["blocked"] = f"backbone directory not found: {BACKBONE}"
        Path(args.report).write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(f"BLOCKED: {evidence['blocked']}")
        return 2

    # 1. Load the real tower offline.
    model = OrdinalImageModel(kind="siglip2", name=BACKBONE, pretrained=True)
    model.eval()
    evidence["backbone_hidden_size"] = int(model.backbone.config.hidden_size)
    evidence["head_parameters"] = sum(p.numel() for p in model.head.parameters())

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "untrained_siglip2_coral.pt"
        torch.save(
            {
                "metadata": {
                    "model_kind": "siglip2",
                    "model_name": BACKBONE,
                    "levels": list(LEVELS),
                },
                "state_dict": model.state_dict(),
            },
            checkpoint,
        )
        evidence["checkpoint_bytes"] = checkpoint.stat().st_size

        # 2. Point the Space at it.
        os.environ["IDDSI_SPACE_CHECKPOINT"] = str(checkpoint)
        os.environ["IDDSI_SPACE_BACKBONE"] = BACKBONE

        import importlib  # noqa: PLC0415

        import core  # noqa: PLC0415

        importlib.reload(core)  # pick up the env vars set above

        # 3. One inference on a photo-shaped input.
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        rng = np.random.default_rng(20261012)
        photo = Image.fromarray(
            (rng.random((300, 300, 3)) * 255).astype("uint8"), mode="RGB"
        )

        permissive = core.predict(photo, abstain_threshold=0.0)
        strict = core.predict(photo, abstain_threshold=1.0)

    checks = {
        "status_ok": permissive.status == "ok",
        "not_placeholder": permissive.is_placeholder is False,
        "checkpoint_recorded": permissive.checkpoint == str(checkpoint),
        "distribution_covers_l3_to_l7": set(permissive.probabilities) == set(LEVELS),
        "distribution_sums_to_one": abs(sum(permissive.probabilities.values()) - 1.0) < 1e-6,
        "decoded_level_inside_l3_to_l7": permissive.decoded_level in LEVELS,
        "decoded_level_uses_rank_rule": permissive.decoded_level
        == LEVELS[
            int(
                (
                    1.0
                    - np.cumsum(
                        [permissive.probabilities[level] for level in LEVELS]
                    )[:-1]
                    > 0.5
                ).sum()
            )
        ],
        "strict_gate_abstains": strict.abstained is True and strict.decoded_level is None,
        "strict_gate_names_no_level": "unclear" in strict.verdict_line,
        "disclaimer_present": core.DISCLAIMER in permissive.to_markdown(),
        "no_banner_on_ok_path": permissive.banner == "",
    }
    evidence["checks"] = checks
    evidence["all_checks_passed"] = all(checks.values())
    evidence["permissive_summary"] = permissive.to_markdown()
    evidence["probabilities"] = {str(k): v for k, v in permissive.probabilities.items()}
    evidence["decoded_level"] = permissive.decoded_level
    evidence["confidence"] = permissive.confidence

    Path(args.report).write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"all_checks_passed": evidence["all_checks_passed"], **checks}, indent=2))
    print(
        "\nNOTE: the level above comes from an UNTRAINED CORAL head and carries no "
        "information about food. It is reported here only to show the weighted "
        "code path executes."
    )
    return 0 if evidence["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
