import sys

sys.path.insert(0, "/work/lanes/exp06_res/trainer")
from PIL import Image
from transformers import AutoImageProcessor

p = AutoImageProcessor.from_pretrained("/work/models_local/siglip2-base-patch16-224")
print("processor size attr:", getattr(p, "size", None))
print("max_num_patches:", getattr(p, "max_num_patches", None))
im = Image.new("RGB", (640, 480), (128, 64, 32))
for kwargs in [
    dict(size={"height": 384, "width": 384}),
    dict(size={"height": 384, "width": 384}, max_num_patches=576),
]:
    try:
        out = p(images=im, return_tensors="pt", **kwargs)
        print(kwargs, "->", {k: tuple(v.shape) for k, v in out.items()})
    except Exception as exc:  # noqa: BLE001
        print(kwargs, "-> ERROR", type(exc).__name__, exc)
