import numpy as np, torch
from PIL import Image
from train.data import build_transform

def photo(size=96, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.stack([(xx/size*200+20),(yy/size*160+40),((xx+yy)/(2*size)*120+60)], axis=-1)
    return Image.fromarray(np.clip(base+rng.normal(0,12,base.shape),0,255).astype("uint8"),"RGB")

img = photo()
torch.manual_seed(17)
a = build_transform("tiny_cnn","test",32,True)(img)
torch.manual_seed(17)
print("flip decision seed17:", bool(torch.rand(()) < 0.5))

def old(image, size=32):
    image = image.convert("RGB").resize((size,size), Image.Resampling.BICUBIC)
    if bool(torch.rand(())) < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    array = np.asarray(image, dtype=np.float32)/255.0
    t = torch.from_numpy(array).permute(2,0,1)
    return (t-0.5)/0.5
torch.manual_seed(17)
b = old(img)
print("equal old(resize-then-flip)-vs-new:", torch.equal(a,b), "maxdiff:", (a-b).abs().max().item())

def old2(image, size=32):
    image = image.convert("RGB")
    if bool(torch.rand(())) < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = image.resize((size,size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32)/255.0
    t = torch.from_numpy(array).permute(2,0,1)
    return (t-0.5)/0.5
torch.manual_seed(17)
c = old2(img)
print("equal flipfirst-vs-new:", torch.equal(a,c), "maxdiff:", (a-c).abs().max().item())
