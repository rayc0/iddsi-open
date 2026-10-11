import numpy as np, torch
from PIL import Image
import train
from train.data import build_transform
from transformers import AutoImageProcessor

MP = "/work/models_local/siglip2-base-patch16-224"

def photo(size=260, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.stack([(xx/size*200+20),(yy/size*160+40),((xx+yy)/(2*size)*120+60)], axis=-1)
    return Image.fromarray(np.clip(base+rng.normal(0,12,base.shape),0,255).astype("uint8"),"RGB")

img = photo()
t = build_transform("siglip2", MP, 224, True)
print("processor type:", type(t.__closure__[0].cell_contents) if t.__closure__ else "?")

torch.manual_seed(0)
a = t(img)["pixel_values"]
print("shape", tuple(a.shape))

# literal v0.7 implementation
proc = AutoImageProcessor.from_pretrained(MP)
torch.manual_seed(0)
im = img
if bool(torch.rand(())) < 0.5:
    im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
b = proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]
print("b shape", tuple(b.shape), "equal:", torch.equal(a,b), "maxdiff", (a-b).abs().max().item())

# does building the transform consume RNG?
torch.manual_seed(0)
_ = build_transform("siglip2", MP, 224, True)
print("rng after build_transform:", torch.rand(()).item())
torch.manual_seed(0)
print("rng baseline:", torch.rand(()).item())
