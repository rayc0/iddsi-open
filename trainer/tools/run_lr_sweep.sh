#!/usr/bin/env bash
# exp01_lr — sequential driver for the 3-config learning-rate sweep.
# HOST-SIDE script (invoked through ~/iddsi/lane_gpu.sh on the host). It holds
# ONE GPU slot for the whole sweep rather than three, so the lane does not crowd
# out the rest of the fleet. Paths under /work/... are CONTAINER paths and are
# only ever used inside the docker exec; host paths are used by this script.
set -u
# The workspace root is supplied by the caller (IDDSI_LANE_ROOT); nothing about
# any particular machine is baked in, so this script runs on any host.
: "${IDDSI_LANE_ROOT:?set IDDSI_LANE_ROOT to the workspace root}"
HOST_ROOT="$IDDSI_LANE_ROOT"
LOG="$HOST_ROOT/lane_runs/exp01_lr/sweep_driver.log"
mkdir -p "$HOST_ROOT/lane_runs/exp01_lr"

for variant in lr1e-5 lr3e-5 lr1e-4; do
  cfg="configs/exp01_lr_${variant}.yaml"
  echo "=== $(date -u +%FT%TZ) START ${variant} (${cfg}) ===" | tee -a "$LOG"
  start=$(date +%s)
  if docker exec -w /work/lanes/exp01_lr/trainer exp01_lr_train \
       python -m train.cli --config "$cfg" >> "$LOG" 2>&1; then
    echo "=== $(date -u +%FT%TZ) OK ${variant} in $(( $(date +%s) - start ))s ===" | tee -a "$LOG"
  else
    rc=$?
    echo "=== $(date -u +%FT%TZ) FAILED ${variant} rc=${rc} after $(( $(date +%s) - start ))s ===" | tee -a "$LOG"
    exit "${rc}"
  fi
done
echo "=== $(date -u +%FT%TZ) SWEEP COMPLETE ===" | tee -a "$LOG"
