import sys, torch
sys.path.insert(0, "/work/lanes/exp06_res/trainer")
from train.data import build_transform

for size in (224, 384):
    t = build_transform("siglip2", "/work/models_local/siglip2-base-patch16-224", size, False)
    from PIL import Image
    im = Image.new("RGB", (640, 480), (128, 64, 32))
    out = t(im)
    print("requested image_size =", size)
    for k, v in out.items():
        print("   ", k, tuple(v.shape), v.dtype)
