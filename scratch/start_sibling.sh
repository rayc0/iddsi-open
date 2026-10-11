#!/bin/bash
# Create + start a sibling GPU container for lane exp06_res.
#
# Why not reuse iddsi-train: its /dev/nvidia* nodes date from Oct 8 while the
# host driver is now 580.142 (the image env still says 575.51.02), so NVML
# fails with "Unknown Error" and torch.cuda.is_available() is False inside it.
# Fixing that needs a restart of a container 31 other lanes share, which is not
# this lane's call to make. A sibling container gets fresh device nodes from the
# nvidia runtime and leaves iddsi-train untouched.
set -x
docker rm -f iddsi-train-exp06 >/dev/null 2>&1
docker create \
  --name iddsi-train-exp06 \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e HF_HOME=/work/hf \
  -e NO_PROXY='*' \
  -v /home/tun/iddsi:/work \
  --ipc=host \
  nvcr.m.daocloud.io/nvidia/pytorch:25.04-py3 \
  sleep infinity
docker start iddsi-train-exp06
sleep 3
docker exec iddsi-train-exp06 python -c "import torch; print('cuda_ok', torch.cuda.is_available(), torch.cuda.device_count())"
