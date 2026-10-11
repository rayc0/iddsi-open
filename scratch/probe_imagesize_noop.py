"""Prove whether `data.image_size` changes anything on the siglip2 path.

Loads a real manifest image and compares the exact tensors that the siglip2
transform produces at image_size=224 vs 384. If they are bit-identical, then a
config that only changes image_size trains on byte-identical data.
"""

import json
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/work/lanes/exp06_res/trainer")
from train.data import build_transform, load_manifest

rows = load_manifest("/work/data/manifest_v07_weighted.jsonl")
train_rows = [r for r in rows if r["split"] == "train"]
print("manifest rows:", len(rows), "train rows:", len(train_rows))

outs = {}
for size in (224, 384):
    transform = build_transform("siglip2", "/work/models_local/siglip2-base-patch16-224", size, False)
    sample = []
    for row in train_rows[:8]:
        with Image.open(row["image_path"]) as image:
            sample.append({k: v.clone() for k, v in transform(image).items()})
    outs[size] = sample

for size in (224, 384):
    shapes = {k: tuple(v.shape) for k, v in outs[size][0].items()}
    print(f"image_size={size}: first train image tensors -> {shapes}")

identical = True
for index in range(8):
    for key in outs[224][index]:
        a, b = outs[224][index][key], outs[384][index][key]
        if a.shape != b.shape or not torch.equal(a, b):
            identical = False
            print(f"  DIFFERS at image {index} key {key}: {tuple(a.shape)} vs {tuple(b.shape)}")
print()
print("224 tensors == 384 tensors for all 8 sampled train images:", identical)
print(
    "=> data.image_size is a NO-OP on the siglip2 path"
    if identical
    else "=> data.image_size DOES change the siglip2 pipeline"
)

# Also record the token count, which fixes the resolution actually used.
pv = outs[224][0]["pixel_values"]
print("tokens per image at 224:", pv.shape[0], "= grid", int(pv.shape[0] ** 0.5), "x", int(pv.shape[0] ** 0.5))
