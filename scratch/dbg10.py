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

# --- siglip2 baseline equivalence (literal v0.7 transform) ---
proc = AutoImageProcessor.from_pretrained(MP)
def v07_siglip2(image):
    if bool(torch.rand(())) < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return proc(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]

img = photo()
ok = True
for seed in range(8):
    torch.manual_seed(seed); a = build_transform("siglip2", MP, 224, True)(img)["pixel_values"]
    torch.manual_seed(seed); b = v07_siglip2(img)
    ok &= torch.equal(a, b)
print("siglip2 baseline bit-identical to v0.7 over 8 seeds:", ok)

# --- tiny_cnn baseline equivalence ---
def v07_tiny(image, size=32):
    image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    if bool(torch.rand(())) < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5
img2 = photo(size=96)
ok2 = True
for seed in range(8):
    torch.manual_seed(seed); a = build_transform("tiny_cnn", "t", 32, True)(img2)
    torch.manual_seed(seed); b = v07_tiny(img2)
    ok2 &= torch.equal(a, b)
print("tiny_cnn baseline bit-identical to v0.7 over 8 seeds:", ok2)

# --- eval split never augments, even with a strong config ---
mapping = {"crop_scale": [0.2, 0.2], "brightness": 0.9, "hue": 0.2, "blur_p": 1.0}
ok3 = True
for seed in range(4):
    torch.manual_seed(seed); e = build_transform("tiny_cnn", "t", 32, False, mapping)(img2)
    torch.manual_seed(seed); p = build_transform("tiny_cnn", "t", 32, False)(img2)
    ok3 &= torch.equal(e, p)
print("eval split ignores augment_config:", ok3)

# --- augmented train output differs from baseline ---
diff = 0
for seed in range(8):
    torch.manual_seed(seed); a = build_transform("tiny_cnn", "t", 32, True, mapping)(img2)
    torch.manual_seed(seed); p = build_transform("tiny_cnn", "t", 32, True)(img2)
    diff += int(not torch.equal(a, p))
print("augmented != baseline in", diff, "of 8 seeds")
