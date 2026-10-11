#!/bin/bash
# Lane qa02_determinism: start a GPU-capable training container.
#
# The shared `iddsi-train` container was started with Runtime=runc and no
# device rules (docker inspect shows HostConfig.Runtime == "runc",
# Devices == null, DeviceCgroupRules == null), so every /dev/nvidia* open
# inside it fails with EPERM and torch.cuda.is_available() is False.  We do
# not restart that container (it is shared by 32 lanes).  Instead this lane
# runs its own container from the same image with --gpus all, which does
# expose the GB10.
set -euo pipefail

if docker ps --format '{{.Names}}' | grep -qx iddsi-train-qa02; then
  echo "iddsi-train-qa02 already running"
  exit 0
fi

docker rm -f iddsi-train-qa02 >/dev/null 2>&1 || true

docker run -d \
  --gpus all \
  --name iddsi-train-qa02 \
  --ipc=host \
  --shm-size 16g \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_HOME=/work/hf \
  -e NO_PROXY='*' \
  -v /home/tun/iddsi:/work \
  nvcr.m.daocloud.io/nvidia/pytorch:25.04-py3 \
  sleep infinity >/dev/null

sleep 3
docker ps --filter name=iddsi-train-qa02 --format '{{.Names}} {{.Status}}'
