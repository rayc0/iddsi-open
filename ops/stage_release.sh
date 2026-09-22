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
# results keep their json/md/yaml (eval evidence) — the exclusion above only drops media/weights
for f in results/fix1_siglip2_coral_seed17/*.npz; do
  # test_predictions.npz is eval evidence (arrays only, no images) — restore it
  [ -f "$f" ] && rsync -a "$f" "$DEST/results/fix1_siglip2_coral_seed17/"
done

# refuse outright if any weight/archive slipped in
if find "$DEST" -name '*.pt' -o -name '*.safetensors' -o -name '*.bin' | grep -q .; then
  echo "🚫 weight file in staging — aborting"; exit 1
fi

# manifest of the staged bytes (this is what the leak scan + the publish gate cover)
( cd "$DEST" && find . -type f | sort > ../MANIFEST.txt )
echo "staged $(wc -l < _release_staging/MANIFEST.txt | tr -d ' ') files -> $DEST"

bash ops/scan_release_leaks.sh "$DEST"
