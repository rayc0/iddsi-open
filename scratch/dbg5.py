import numpy as np, torch
from PIL import Image
import train
print("train from:", train.__file__)
from train.data import build_transform

def photo(size=96, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.stack([(xx/size*200+20),(yy/size*160+40),((xx+yy)/(2*size)*120+60)], axis=-1)
    return Image.fromarray(np.clip(base+rng.normal(0,12,base.shape),0,255).astype("uint8"),"RGB")

img = photo()
t = build_transform("tiny_cnn","test",32,True)
torch.manual_seed(17)
a = t(img)

torch.manual_seed(17)
im = img.convert("RGB")
flip = bool(torch.rand(()) < 0.5)
print("flip:", flip)
if flip: im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
im = im.resize((32,32), Image.Resampling.BICUBIC)
arr = np.asarray(im, dtype=np.float32)/255.0
m = (torch.from_numpy(arr).permute(2,0,1)-0.5)/0.5
print("equal flipfirst:", torch.equal(a,m), "maxdiff:", (a-m).abs().max().item())

torch.manual_seed(17)
_ = torch.rand(())
im2 = img.convert("RGB").resize((32,32), Image.Resampling.BICUBIC)
arr2 = np.asarray(im2, dtype=np.float32)/255.0
m2 = (torch.from_numpy(arr2).permute(2,0,1)-0.5)/0.5
print("equal unflipped:", torch.equal(a,m2), "maxdiff:", (a-m2).abs().max().item())
print("a[0,0,:4]", a[0,0,:4].tolist())
print("m[0,0,:4]", m[0,0,:4].tolist())
print("m2[0,0,:4]", m2[0,0,:4].tolist())
