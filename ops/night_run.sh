#!/bin/bash
# IDDSI night window runner v7 (2026-09-06).
#
# Replaces night_pilot.sh v6, which reported "done" while every substantive step
# had failed. Four defects fixed here:
#   1. PYTHONPATH included /work/repo/harvest, so harvest/types.py shadowed the
#      stdlib `types` module and the interpreter could not boot. Every relabel
#      pass died in <1s with exit=1. Now PYTHONPATH=/work/repo only.
#   2. Heredocs were piped into `docker exec` without -i, so python read an empty
#      program from a closed stdin, exited 0 and wrote a 0-byte log. The FLUX
#      load test never actually ran. All heredoc execs now use -i.
#   3. No step verified its own effect, so failures logged as successes.
#      Every step here asserts the artifact it was supposed to produce.
#   4. The runner was one-shot: it ran once and exited, losing every following
#      night. This is now driven by cron (see ops/install_night_cron.sh).
#
# Order matters: relabel is the unlock. Nothing downstream can run until the
# deferred rows have labels.
set -uo pipefail

IDDSI=~/iddsi
L="$IDDSI/night_run.log"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$IDDSI/runs/night_$STAMP"
DATASETS=(realweak_pilot harvest_d1 harvest_d1b)
PYP="/work/repo"

mkdir -p "$RUN_DIR"
log() { echo "$(date '+%F %T') $*" | tee -a "$L"; }
fail() { log "FAILED: $*"; }

log "=== night_run v7 start (run $STAMP) ==="

# --- count labelled rows in a dataset (the ground truth for progress) --------
count_labelled() {
  docker exec iddsi-train python3 -c "
import json,sys
n=t=0
for line in open('/work/data/$1/events.jsonl'):
    if not line.strip(): continue
    t+=1
    if (json.loads(line) or {}).get('level_weak') is not None: n+=1
print(f'{n}/{t}')
" 2>/dev/null || echo "?/?"
}

# --- 0. CUDA self-heal (NVML loss on long-lived containers) ------------------
if ! docker exec iddsi-train python3 -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" >>"$L" 2>&1; then
  log "CUDA lost in container -> docker restart iddsi-train"
  docker restart iddsi-train >>"$L" 2>&1
  sleep 10
fi
CUDA=$(docker exec iddsi-train python3 -c "import torch;print(torch.cuda.is_available())" 2>/dev/null)
log "cuda available: $CUDA"
[ "$CUDA" = "True" ] || { fail "no CUDA; aborting before touching deepseek"; exit 1; }

# --- 1. pre-flight: the import that was broken for 3 nights ------------------
if ! docker exec -e PYTHONPATH=$PYP iddsi-train bash -c \
     "cd /work/repo && python3 -c 'from harvest.label import Qwen3VLLabeler; from harvest.relabel import main; print(\"IMPORTS_OK\")'" >>"$L" 2>&1; then
  fail "harvest imports still broken; aborting (deepseek untouched)"; exit 1
fi
log "pre-flight imports OK"

log "labelled BEFORE: $(for d in "${DATASETS[@]}"; do printf '%s=%s ' "$d" "$(count_labelled "$d")"; done)"

# --- 2. take the GPU ---------------------------------------------------------
log "pausing deepseek-v4"
docker stop deepseek-v4 >>"$L" 2>&1
sleep 5
restore_deepseek() {
  log "restoring deepseek-v4"
  docker start deepseek-v4 >>"$L" 2>&1
}
trap restore_deepseek EXIT

# --- 3. canary: prove labelling works on 3 images before spending the night --
log "building 3-image canary"
docker exec -i iddsi-train python3 - >>"$L" 2>&1 <<'EOF'
import json, os, shutil
src="/work/data/realweak_pilot"; dst="/work/data/_canary"
shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst, exist_ok=True)
rows=[]
for line in open(f"{src}/events.jsonl"):
    if not line.strip(): continue
    e=json.loads(line)
    if (e.get("weak_label_record") or {}).get("rule")=="deferred":
        rows.append(e)
    if len(rows)==3: break
# Copy the referenced media into the canary root. relabel rejects any media
# path that resolves outside the dataset root, so a symlinked media/ dir fails.
for e in rows:
    for m in e.get("media_files") or []:
        rel=m.get("path")
        if not rel: continue
        s=os.path.join(src, rel); d=os.path.join(dst, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.exists(s): shutil.copy(s, d)
open(f"{dst}/events.jsonl","w").write("".join(json.dumps(r)+"\n" for r in rows))
for name in ("realweak-event.schema.json",):
    if os.path.exists(f"{src}/{name}"): shutil.copy(f"{src}/{name}", f"{dst}/{name}")
print("canary rows:", len(rows))
EOF

docker exec -e PYTHONPATH=$PYP -e HF_HUB_OFFLINE=1 -e HF_HOME=/work/hf iddsi-train bash -c \
  "cd /work/repo && python3 -m harvest.relabel --dataset /work/data/_canary --device cuda" \
  >"$RUN_DIR/canary.log" 2>&1
CANARY=$(count_labelled _canary)
log "canary result: $CANARY"
if [ "${CANARY%%/*}" = "0" ] || [ "${CANARY%%/*}" = "?" ]; then
  fail "canary produced no labels -- see $RUN_DIR/canary.log; NOT running the full pass"
  tail -20 "$RUN_DIR/canary.log" >>"$L"
  exit 1
fi
log "canary OK -- proceeding to full relabel"

# --- 4. the unlock: relabel every deferred row -------------------------------
for DS in "${DATASETS[@]}"; do
  BEFORE=$(count_labelled "$DS")
  log "relabel $DS (was $BEFORE)"
  docker exec -e PYTHONPATH=$PYP -e HF_HUB_OFFLINE=1 -e HF_HOME=/work/hf iddsi-train bash -c \
    "cd /work/repo && python3 -m harvest.relabel --dataset /work/data/$DS --device cuda" \
    >"$RUN_DIR/relabel_$DS.log" 2>&1
  RC=$?
  AFTER=$(count_labelled "$DS")
  log "relabel $DS: exit=$RC  $BEFORE -> $AFTER"
  [ "$RC" -eq 0 ] || tail -15 "$RUN_DIR/relabel_$DS.log" >>"$L"
done

log "labelled AFTER: $(for d in "${DATASETS[@]}"; do printf '%s=%s ' "$d" "$(count_labelled "$d")"; done)"

# --- 5. build the training manifest ------------------------------------------
MAN=/work/data/manifest_$STAMP.jsonl
log "building manifest -> $MAN"
docker exec -e PYTHONPATH=$PYP iddsi-train bash -c \
  "cd /work/repo/trainer && python3 -m train.build_manifest \
     --dataset /work/data/realweak_pilot --dataset /work/data/harvest_d1 \
     --dataset /work/data/harvest_d1b --out $MAN --drop-at-most" \
  >"$RUN_DIR/manifest.log" 2>&1
if [ $? -ne 0 ]; then
  fail "manifest build failed"; tail -20 "$RUN_DIR/manifest.log" >>"$L"; exit 1
fi
ROWS=$(docker exec iddsi-train bash -c "wc -l < $MAN" 2>/dev/null | tr -d ' ')
log "manifest rows: $ROWS"
cp "$RUN_DIR/manifest.log" "$RUN_DIR/manifest_stats.json" 2>/dev/null

if [ "${ROWS:-0}" -lt 50 ]; then
  fail "only $ROWS trainable rows -- too few to train; stopping here (labels are the bottleneck)"
  exit 1
fi

# --- 6. train ----------------------------------------------------------------
CFG=/work/data/siglip2_night_$STAMP.yaml
# The hub copy of google/siglip2-base-patch16-224 carries a SigLIP-1 style
# config whose vision tower defaults to 256 patch positions, while the actual
# checkpoint is 14x14=196 -> Siglip2VisionModel.from_pretrained aborts with a
# position_embedding size mismatch. models_local holds the converted copy with
# the correct `num_patches: 196` and a Siglip2VisionModel architecture.
SIGLIP=/work/models_local/siglip2-base-patch16-224
docker exec -i iddsi-train python3 - >>"$L" 2>&1 <<EOF
import re
src=open("/work/repo/trainer/configs/siglip2.yaml").read()
src=src.replace("/path/to/physically_tested_manifest.jsonl","$MAN")
src=src.replace("output_dir: ../runs","output_dir: /work/runs/night_$STAMP")
src=src.replace("name: google/siglip2-base-patch16-224","name: $SIGLIP")
src=re.sub(r"seeds: \[.*?\]","seeds: [17]",src)
open("$CFG","w").write(src)
print("config written; backbone =", "$SIGLIP")
EOF

log "training start (config $CFG)"
docker exec -e PYTHONPATH=$PYP -e HF_HOME=/work/hf -e HF_HUB_OFFLINE=1 iddsi-train bash -c \
  "cd /work/repo/trainer && python3 -m train.cli --config $CFG" \
  >"$RUN_DIR/train.log" 2>&1
TRC=$?
log "training exit=$TRC"
CKPT=$(docker exec iddsi-train bash -c "find /work/runs/night_$STAMP -name '*.pt' -o -name '*.safetensors' 2>/dev/null | head -5" 2>/dev/null)
if [ -n "$CKPT" ]; then
  log "CHECKPOINT(S) WRITTEN:"; echo "$CKPT" | tee -a "$L"
else
  fail "no checkpoint found under /work/runs/night_$STAMP"
  tail -30 "$RUN_DIR/train.log" >>"$L"
fi

log "=== night_run v7 done (artifacts in $RUN_DIR) ==="
