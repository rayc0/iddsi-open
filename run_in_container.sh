#!/bin/bash
# exp04_aug lane: run the IDDSI trainer inside a GPU-enabled container.
#
# WHY NOT `docker exec iddsi-train`: the fleet's shared `iddsi-train` container
# cannot see the GPU. Measured in it on 2026-10-12:
#   open("/dev/nvidiactl") -> EPERM ("Operation not permitted")
#   open("/dev/nvidia0")   -> EPERM
#   nvmlInit_v2()          -> 999 (NVML_ERROR_NO_PERMISSION)
#   cudaGetDeviceCount()   -> 100 (cudaErrorNoDevice)
# The device nodes exist and are world-writable, so this is a cgroup device
# filter problem in how that container was started, not a driver problem: the
# host's own nvidia-smi and the host CUDA processes work fine. Every lane
# therefore spins up its own `--gpus all` container from the same base image.
#
# Usage: bash run_in_container.sh <args...>     (e.g. python -m train.cli --config ...)
set -euo pipefail

IMAGE=nvcr.m.daocloud.io/nvidia/pytorch:25.04-py3

exec docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=16g \
  -v /home/tun/iddsi:/work \
  -w /work/lanes/exp04_aug/trainer \
  -e HF_HOME=/work/hf \
  -e HF_ENDPOINT=https://hf-mirror.com \
  "$IMAGE" "$@"
