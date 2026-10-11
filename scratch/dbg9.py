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

plain = proc(images=img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
flipped = proc(images=img.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT), return_tensors="pt")["pixel_values"][0]
print("plain != flipped:", not torch.equal(plain, flipped))

torch.manual_seed(0); v = float(torch.rand(())); print("A", v)
a = t(img)["pixel_values"]
print("a==plain", torch.equal(a, plain), "a==flipped", torch.equal(a, flipped))

torch.manual_seed(0); v2 = float(torch.rand(())); print("B", v2)
im = img
if v2 < 0.5:
    im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
b = proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]
print("b==plain", torch.equal(b, plain), "b==flipped", torch.equal(b, flipped))
print("v==v2", v == v2)
