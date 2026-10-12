#!/bin/bash
# stage_release.sh — assemble the EXACT bytes of the IDDSI-Open v0 release (A398 shape:
# code + eval results + flow-test grader + the negative finding; NO weights, NO dataset)
# into _release_staging/v0/, then run the full leak scan over the staged tree.
# Rebuilds from scratch every run. Usage: bash ops/stage_release.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=_release_staging/v0
rm -rf "$DEST"
mkdir -p "$DEST"

# code + docs + results (the A398 content list).
# Excluded from v0: ops/ (box-ops: host paths, container names, the leak term list itself),
# docs/RUNBOOK.md (internal infra: ssh/tailscale/host paths), drafts/, all media/weights.
for d in harvest trainer flowtest synth dataschema crowd tools docs hf paper results; do
  [ -d "$d" ] && rsync -a --exclude '__pycache__' --exclude '.pytest_cache' --exclude '*.pyc' \
    --exclude '*.jpg' --exclude '*.jpeg' --exclude '*.png' --exclude '*.mp4' \
    --exclude '*.log' --exclude 'fixtures' \
    --exclude '*.pt' --exclude '*.npz' "$d" "$DEST/"
done
rm -f "$DEST/docs/RUNBOOK.md" "$DEST/docs/ARCHITECTURE.md"

# --- 2026-10-12 leak-scan remediation (lane rel03_leakscan) -------------------
# Three more exclusion classes, each one a measured BLOCK/REVIEW hit from
# ops/scan_release_leaks.sh over the previous staged tree. Nothing here is
# deleted from the repository; these files stay on the box where they are used.

# (1) WITHHELD DATASET BYTES. results/v04_groupdisjoint_consensus_manifest.jsonl
#     is the 684-row training manifest: per-event image paths, split assignment,
#     group_id, the consensus level, and all four labeller votes. The release
#     card withholds the harvested image DATASET, and this file is that dataset
#     minus the pixels -- shipping it would publish the label set and the exact
#     train/test assignment. Remove this one known file; any OTHER *.jsonl that
#     appears makes staging refuse outright below rather than being dropped
#     silently, so a future manifest cannot ship unnoticed.
rm -f "$DEST/results/v04_groupdisjoint_consensus_manifest.jsonl"

# (2) BOX-OPS LANE DRIVERS living outside ops/. They hardcode the DGX host root,
#     private lane checkout paths, container names and the GPU slot gate, and
#     they cannot run from any other clone. Same category as ops/, which the
#     header above already excludes for the same reason. They stay in the repo
#     (the box still runs them); only the published bytes drop them.
for f in tools/collect_kappa.py tools/run_arm.sh trainer/tools/run_lr_sweep.sh; do
  rm -f "$DEST/$f"
done

# (3) WEIGHT/ARCHIVE REFUSAL. The publish workflow (.github/workflows/publish-hf.yml)
#     refuses *.npz in the staged tree, and the rsync above already excludes it, so
#     the old "restore test_predictions.npz" step contradicted both and would have
#     shipped a refused file the moment one existed. It is removed rather than
#     kept: this strengthens the gate, it does not weaken it.
# results keep their json/md/yaml (eval evidence) — the exclusion above only drops media/weights

# refuse outright if any weight/archive slipped in
if find "$DEST" -name '*.pt' -o -name '*.safetensors' -o -name '*.bin' | grep -q .; then
  echo "🚫 weight file in staging — aborting"; exit 1
fi

# refuse outright if any dataset manifest slipped in (the dataset is WITHHELD).
# Fail-closed: a manifest that is not the one known-withheld file above stops the
# release instead of being quietly removed.
if find "$DEST" -name '*.jsonl' | grep -q .; then
  echo "🚫 dataset manifest (*.jsonl) in staging — aborting:"
  find "$DEST" -name '*.jsonl'
  exit 1
fi

# manifest of the staged bytes (this is what the leak scan + the publish gate cover)
( cd "$DEST" && find . -type f | sort > ../MANIFEST.txt )
echo "staged $(wc -l < _release_staging/MANIFEST.txt | tr -d ' ') files -> $DEST"

bash ops/scan_release_leaks.sh "$DEST"
