#!/bin/bash
# exp05_loss lane launcher: run one loss-shaping arm (2 seeds) inside a
# GPU-capable side container.
#
# Why a side container: `docker exec iddsi-train ...` currently fails with
# "CUDA was requested but is unavailable" because the long-lived iddsi-train
# container (started 2026-10-08) has /dev/nvidia* nodes whose cgroup device
# filter denies open() with EPERM.  A freshly created container with
# `--gpus all` on the same host sees the GPU fine, so this lane snapshots the
# iddsi-train environment into exp05-loss-gpu:latest and runs from that.
# The shared iddsi-train container is left untouched.
#
# Usage: bash run_arm.sh <arm>   where arm in {ls005, ls010, beta099}
set -euo pipefail

ARM="${1:?usage: run_arm.sh <ls005|ls010|beta099>}"
HOST_REPO=/home/tun/iddsi
IMAGE=exp05-loss-gpu:latest
LOG=/home/tun/iddsi/lane_runs/exp05_loss/logs/${ARM}.log

mkdir -p "$(dirname "$LOG")"
echo "$(date -u +%FT%TZ) launching arm=$ARM" | tee -a "$LOG"

bash /home/tun/iddsi/lane_gpu.sh docker run --rm --gpus all \
  --ipc=host --shm-size=16g \
  -v "$HOST_REPO":/work \
  -w /work/lanes/exp05_loss/trainer \
  "$IMAGE" python -m train.cli --config "configs/exp05_loss_${ARM}.yaml" 2>&1 | tee -a "$LOG"

echo "$(date -u +%FT%TZ) finished arm=$ARM" | tee -a "$LOG"
