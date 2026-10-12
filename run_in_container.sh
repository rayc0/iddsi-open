#!/bin/bash
# fix03_retrain lane: run the IDDSI trainer inside a GPU-enabled container.
#
# WHY NOT `docker exec iddsi-train`: the fleet's shared `iddsi-train` container
# cannot see the GPU. Measured in it on 2026-10-12 by this lane:
#   docker exec iddsi-train nvidia-smi -L  -> "Failed to initialize NVML: Unknown Error"
#   docker exec iddsi-train python -c "import torch; torch.cuda.is_available()" -> False
#   docker inspect iddsi-train --format '{{.HostConfig.Devices}}' -> null   (runtime: runc)
# i.e. it was started without --gpus, so it has no /dev/nvidia* access at all.
# The same failure mode is recorded for the fleet in
# lanes/exp04_aug/run_in_container.sh ("open(/dev/nvidiactl) -> EPERM ...
# cudaGetDeviceCount() -> 100"). Every lane therefore spins up its own
# `--gpus all` container from the same base image. This lane's container is
# `iddsi-train-fix03`.
#
# Usage:
#   bash run_in_container.sh start                     # create + start (idempotent)
#   bash run_in_container.sh exec <args...>            # run inside it
#   bash run_in_container.sh stop                      # remove it
set -euo pipefail

IMAGE=nvcr.m.daocloud.io/nvidia/pytorch:25.04-py3
NAME=iddsi-train-fix03
WORKDIR=/work/lanes2/fix03_retrain/trainer

case "${1:-}" in
  start)
    if ! docker inspect "$NAME" >/dev/null 2>&1; then
      docker run -d --name "$NAME" --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --shm-size=16g \
        -v /home/tun/iddsi:/work \
        -w "$WORKDIR" \
        -e HF_HOME=/work/hf \
        -e HF_ENDPOINT=https://hf-mirror.com \
        "$IMAGE" sleep infinity >/dev/null
    fi
    docker start "$NAME" >/dev/null 2>&1 || true
    echo "container $NAME up"
    ;;
  exec)
    shift
    exec docker exec -w "$WORKDIR" "$NAME" "$@"
    ;;
  stop)
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    echo "container $NAME removed"
    ;;
  *)
    echo "usage: $0 {start|exec <args...>|stop}" >&2
    exit 2
    ;;
esac
