import numpy as np, torch
from PIL import Image
from train.data import build_transform
from transformers import AutoImageProcessor
MP = "/work/models_local/siglip2-base-patch16-224"
def photo(size=260, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.stack([(xx/size*200+20),(yy/size*160+40),((xx+yy)/(2*size)*120+60)], axis=-1)
    return Image.fromarray(np.clip(base+rng.normal(0,12,base.shape),0,255).astype("uint8"),"RGB")

proc = AutoImageProcessor.from_pretrained(MP)
def v07_siglip2(image):
    if bool(torch.rand(()) < 0.5):          # v0.7 parenthesisation, verbatim
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return proc(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]

img = photo()
ok = True; flips = 0
for seed in range(12):
    torch.manual_seed(seed); a = build_transform("siglip2", MP, 224, True)(img)["pixel_values"]
    torch.manual_seed(seed); b = v07_siglip2(img)
    if not torch.equal(a, b):
        ok = False
        print("  mismatch seed", seed, "maxdiff", (a-b).abs().max().item())
    flips += int(not torch.equal(b, proc(images=img.convert("RGB"), return_tensors="pt")["pixel_values"][0]))
print("siglip2 baseline bit-identical to v0.7 over 12 seeds:", ok, "| flipped in", flips, "of 12")

def v07_tiny(image, size=32):
    image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    if bool(torch.rand(()) < 0.5):
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5
img2 = photo(size=96)
ok2 = True; flips2 = 0
for seed in range(12):
    torch.manual_seed(seed); a = build_transform("tiny_cnn", "t", 32, True)(img2)
    torch.manual_seed(seed); b = v07_tiny(img2)
    if not torch.equal(a, b):
        ok2 = False; print("  tiny mismatch seed", seed, (a-b).abs().max().item())
print("tiny_cnn baseline bit-identical to v0.7 over 12 seeds:", ok2)
