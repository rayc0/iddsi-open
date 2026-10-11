import sys

import torch

sys.path.insert(0, "/work/lanes/exp06_res/trainer")
from train.model import OrdinalImageModel

device = torch.device("cuda")
model = OrdinalImageModel("siglip2", "/work/models_local/siglip2-base-patch16-224", 0.1, True).to(device)
tower = getattr(model.backbone, "vision_model", model.backbone)
emb = getattr(tower.embeddings, "position_embedding", None)
print("position_embedding shape:", None if emb is None else tuple(emb.weight.shape))
print("config num_patches:", model.backbone.config.num_patches)
print("config image_size:", getattr(model.backbone.config, "image_size", None))

for n_patches, grid in ((196, (14, 14)), (576, (24, 24))):
    pixel_values = torch.randn(2, n_patches, 768, device=device)
    mask = torch.ones(2, n_patches, dtype=torch.int32, device=device)
    shapes = torch.tensor([grid, grid], dtype=torch.int64, device=device)
    try:
        with torch.no_grad():
            out = model(pixel_values=pixel_values, pixel_attention_mask=mask, spatial_shapes=shapes)
        print(f"forward with {n_patches} patches grid={grid} -> logits {tuple(out.shape)} OK")
    except Exception as exc:  # noqa: BLE001
        print(f"forward with {n_patches} patches grid={grid} -> ERROR {type(exc).__name__}: {exc}")

# peak memory probe at a realistic batch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
for bs, np_ in ((32, 196), (16, 576), (8, 576), (12, 576)):
    try:
        pv = torch.randn(bs, np_, 768, device=device, requires_grad=True)
        mask = torch.ones(bs, np_, dtype=torch.int32, device=device)
        shapes = torch.tensor([[(24 if np_ == 576 else 14)] * 2] * bs, dtype=torch.int64, device=device)
        out = model(pixel_values=pv, pixel_attention_mask=mask, spatial_shapes=shapes)
        loss = out.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        print(f"train-step sim bs={bs} patches={np_} peak={torch.cuda.max_memory_stats()['allocated_bytes.all.peak']/2**20:.0f} MiB")
        del pv, out, loss
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except torch.OutOfMemoryError as exc:
        print(f"train-step sim bs={bs} patches={np_} -> OOM")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
print("total GPU MiB:", torch.cuda.get_device_properties(0).total_memory / 2**20)
