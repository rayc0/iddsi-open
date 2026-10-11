"""Confirm the patched build_transform: 224 unchanged, 384 now really 384."""

import sys

import torch
from PIL import Image

sys.path.insert(0, "/work/lanes/exp06_res/trainer")
from train.data import _siglip2_native_size, build_transform, load_manifest
from transformers import AutoImageProcessor

proc = AutoImageProcessor.from_pretrained("/work/models_local/siglip2-base-patch16-224")
print("native size recovered:", _siglip2_native_size(proc))

rows = load_manifest("/work/data/manifest_v07_weighted.jsonl")
train_rows = [r for r in rows if r["split"] == "train"]

outs = {}
for size in (224, 384):
    transform = build_transform("siglip2", "/work/models_local/siglip2-base-patch16-224", size, False)
    sample = []
    for row in train_rows[:8]:
        with Image.open(row["image_path"]) as image:
            sample.append({k: v.clone() for k, v in transform(image).items()})
    outs[size] = sample
    first = sample[0]
    grid = tuple(int(v) for v in first["spatial_shapes"].tolist())
    print(
        f"image_size={size}: pixel_values {tuple(first['pixel_values'].shape)} "
        f"spatial_shapes {grid} -> effective {grid[0] * 16}x{grid[1] * 16}"
    )

same = all(
    torch.equal(outs[224][i][k], outs[384][i][k]) for i in range(8) for k in outs[224][i]
)
print("224 == 384 tensors (must be False now):", same)
print("224 token count:", outs[224][0]["pixel_values"].shape[0])
print("384 token count:", outs[384][0]["pixel_values"].shape[0])
print("384/224 token ratio:", outs[384][0]["pixel_values"].shape[0] / outs[224][0]["pixel_values"].shape[0])
