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
img = photo()
proc = AutoImageProcessor.from_pretrained(MP)
t = build_transform("siglip2", MP, 224, True)

torch.manual_seed(0); flip = bool(torch.rand(()) < 0.5)
print("flip:", flip)

plain = proc(images=img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
flipped = proc(images=img.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT), return_tensors="pt")["pixel_values"][0]

torch.manual_seed(0); a = t(img)["pixel_values"]
print("a == plain:", torch.equal(a, plain), "a == flipped:", torch.equal(a, flipped))

torch.manual_seed(0)
im = img
if bool(torch.rand(())) < 0.5: im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
b = proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]
print("b == plain:", torch.equal(b, plain), "b == flipped:", torch.equal(b, flipped))
print("a vs b maxdiff", (a-b).abs().max().item())
# also: is the two-processor instance deterministic?
c = proc(images=img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
print("processor deterministic:", torch.equal(plain, c))
