#!/usr/bin/env bash
# ops/flux_load_test.sh — FLUX.1-schnell GPU load test for the DGX Spark box.
#
# Proves the FLUX weights actually generate a real image on the box GPU: CUDA
# is up, FLUX.1-schnell loads from /work/models_local/FLUX.1-schnell, and ONE
# 1024x1024 image from a FIXED long bank prompt is a non-trivial photograph
# (file-size + pixel-variance artifact assertions), not a blank frame.
#
# Exact run command (on the Spark HOST as tun — NOT in the container, NOT on
# the Mac, which has no GPU):
#   bash ~/iddsi/repo/ops/flux_load_test.sh
#
# Artefacts: ~/iddsi/flux_load_test_out/flux_1024.png, flux_gen.log, plus the
# single PASS/FAIL line this script prints.  Run inside the GPU night window
# with free memory (docs/RUNBOOK.md section 5); this script never stops or
# starts deepseek-v4 — scheduling is the operator's job.
#
# Lesson encoded here (ops/night_run.sh header, defect 2): a heredoc piped
# into `docker exec` WITHOUT -i leaves python reading a CLOSED stdin, so it
# exits 0 on an EMPTY program and a 0-byte log gets recorded as success —
# that is how the FLUX load test "passed" without ever running.  Every
# `docker exec` below that consumes a heredoc uses `docker exec -i`; the -i
# is the whole point.  Likewise every step asserts its ARTIFACT (a real PNG
# with non-trivial size and pixel variance), never an exit code or log line.
#
# Container/env conventions follow docs/RUNBOOK.md section 2: proxy vars
# cleared, HF_HOME=/work/hf, HF_ENDPOINT=https://hf-mirror.com.  On the host
# the state dir is ~/iddsi; in the container it is /work.
#
# Do NOT run on the Mac (no GPU).  This script is shell-checked locally with
# `bash -n`; its assertion logic is unit-tested with fixture images in
# ops/tests/test_flux_image_check.py.
set -uo pipefail

IDDSI="${HOME}/iddsi"
CONTAINER="iddsi-train"
FLUX_DIR_HOST="$IDDSI/models_local/FLUX.1-schnell"
FLUX_MAIN="$FLUX_DIR_HOST/flux1-schnell.safetensors"
MIN_FLUX_BYTES=23000000000
OUT_DIR="$IDDSI/flux_load_test_out"
OUT_PNG="$OUT_DIR/flux_1024.png"
OUT_PNG_WORK="/work/flux_load_test_out/flux_1024.png"
GEN_LOG="$OUT_DIR/flux_gen.log"

log() { echo "$(date '+%F %T') $*"; }
fail() { log "FAIL: flux_load_test: $*"; exit 1; }

log "=== flux_load_test start ==="

# --- 1. CUDA first: self-heal a long-lived container that lost NVML --------
if ! docker exec "$CONTAINER" python3 -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"; then
    log "CUDA unavailable inside $CONTAINER — restarting container, waiting 30 s, re-checking"
    docker restart "$CONTAINER" || fail "docker restart $CONTAINER failed"
    sleep 30
    docker exec "$CONTAINER" python3 -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" \
        || fail "CUDA still unavailable after restart — aborting loudly, no image generated"
fi
log "step 1/4: CUDA available inside $CONTAINER"

# --- 2. FLUX weights present (size-checked, never log-claimed) --------------
if [ ! -f "$FLUX_MAIN" ]; then
    fail "FLUX weights missing: $FLUX_MAIN not found — aborting, no fallback in a FLUX load test"
fi
FLUX_BYTES="$(stat -c %s "$FLUX_MAIN")"
[ "$FLUX_BYTES" -ge "$MIN_FLUX_BYTES" ] \
    || fail "FLUX weights incomplete: $FLUX_BYTES bytes < $MIN_FLUX_BYTES — aborting"
log "step 2/4: FLUX weights OK ($FLUX_BYTES bytes)"

# --- 3. Generate ONE 1024x1024 from a FIXED long bank prompt ----------------
mkdir -p "$OUT_DIR"
rm -f "$OUT_PNG" # a stale image must never pass as a fresh one
# NOTE: `docker exec -i` below is mandatory, not stylistic.  Without -i the
# heredoc arrives on a closed stdin, python runs an empty program, exits 0,
# and writes nothing — the exact false-success this test exists to prevent.
docker exec -i \
    -e HTTP_PROXY= -e HTTPS_PROXY= -e http_proxy= -e https_proxy= \
    -e NO_PROXY='*' \
    -e HF_HOME=/work/hf \
    -e HF_ENDPOINT=https://hf-mirror.com \
    "$CONTAINER" python3 - >"$GEN_LOG" 2>&1 <<'PYEOF'
import random
import sys
sys.path.insert(0, "/work/repo/synth")
import torch
from diffusers import FluxPipeline
from synth.prompt_bank import build_prompt

spec = build_prompt(4, random.Random(20260831))  # FIXED seed: same prompt every run
clip_words = len(spec.short_prompt.split())
full_words = len(spec.prompt.split())
print(f"CLIP core words: {clip_words}; T5 full words: {full_words}", flush=True)
assert full_words > 77, "fixed prompt no longer exceeds CLIP — bank changed?"
assert clip_words <= 77, "CLIP core over 77-token budget — bank changed?"

pipe = FluxPipeline.from_pretrained("/work/models_local/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
image = pipe(
    prompt=spec.short_prompt,  # CLIP: compressed core, at most 77 tokens
    prompt_2=spec.prompt,      # T5: full texture descriptor, un-truncated
    height=1024,
    width=1024,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator(device="cuda").manual_seed(20260831),
).images[0]
assert image.size == (1024, 1024), f"unexpected output size {image.size}"
image.save("/work/flux_load_test_out/flux_1024.png")
print("saved /work/flux_load_test_out/flux_1024.png", flush=True)
PYEOF
GEN_RC=$?
[ "$GEN_RC" -eq 0 ] || fail "generation failed (exit=$GEN_RC) — see $GEN_LOG"
[ -s "$OUT_PNG" ] || fail "generation exited 0 but wrote no image — see $GEN_LOG (0-byte-log guard)"
log "step 3/4: generation done (log: $GEN_LOG)"

# --- 4. Artifact assertions: size + pixel variance, inside the container ----
CHECK_OUT="$(docker exec \
    -e HTTP_PROXY= -e HTTPS_PROXY= -e http_proxy= -e https_proxy= \
    -e NO_PROXY='*' \
    -e HF_HOME=/work/hf \
    -e HF_ENDPOINT=https://hf-mirror.com \
    "$CONTAINER" python3 /work/repo/ops/flux_image_check.py "$OUT_PNG_WORK" 2>&1)"
CHECK_RC=$?
[ "$CHECK_RC" -eq 0 ] || fail "artifact assertions failed: $CHECK_OUT"
log "step 4/4: $CHECK_OUT"

# The ONLY success line, printed after every assertion above has passed.
log "PASS: flux_load_test passed ($CHECK_OUT; file: $OUT_PNG)"
