import torch
from PIL import Image
from train.data import build_transform
from transformers import AutoImageProcessor
MP = "/work/models_local/siglip2-base-patch16-224"
img = Image.new("RGB", (260, 260), (120, 60, 30))
proc = AutoImageProcessor.from_pretrained(MP)
t = build_transform("siglip2", MP, 224, True)

torch.manual_seed(0); d1 = float(torch.rand(())); print("draw1", d1, d1 < 0.5)
plain = proc(images=img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
flip_img = img.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
flipped = proc(images=flip_img, return_tensors="pt")["pixel_values"][0]
torch.manual_seed(0); d2 = float(torch.rand(())); print("draw2", d2, d2 < 0.5)
torch.manual_seed(0); a = t(img)["pixel_values"]
print("a==plain", torch.equal(a, plain), "a==flipped", torch.equal(a, flipped))

torch.manual_seed(0)
im = img
v = float(torch.rand(())); print("draw3", v, v < 0.5)
if v < 0.5:
    im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
b = proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]
print("b==plain", torch.equal(b, plain), "b==flipped", torch.equal(b, flipped))
