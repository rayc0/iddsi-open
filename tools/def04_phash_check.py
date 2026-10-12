#!/usr/bin/env python3
"""Compare recorded events.jsonl phash against the HARVEST algorithm
(harvest/image_ops.py:perceptual_hash, conventional 32x32 imagehash pHash --
the code that produced the recorded values) AND the an04 8x8 hand-rolled
recompute. Reports exact-match counts and the borderline cases.
Run with -I. Writes nothing; prints a summary.
"""
import json, os, statistics
from collections import Counter
import numpy as np
from PIL import Image, ImageOps

DATA = os.environ.get("IDDSI_DATA", "data")
if not os.path.isdir(DATA):
    DATA = "/work/data"

def _dct(size):
    p = np.arange(size, dtype=np.float64); f = p[:, None]
    m = np.cos((np.pi / size) * (p + 0.5) * f)
    m[0, :] *= 1.0 / np.sqrt(size); m[1:, :] *= np.sqrt(2.0 / size)
    return m
C32 = _dct(32)

def phash_harvest(path):  # harvest/image_ops.py:perceptual_hash
    g = ImageOps.grayscale(Image.open(path)).resize((32, 32), Image.Resampling.LANCZOS)
    px = np.asarray(g, dtype=np.float64); t = C32 @ px @ C32.T; low = t[:8, :8]
    med = float(np.median(low[1:, :])); v = 0
    for b in (low > med).flatten():
        v = (v << 1) | int(b)
    return f"{v:016x}"

def phash_an04(path):  # verify_group_disjointness.py non-standard 8x8
    size = 8
    a = np.asarray(Image.open(path).convert("L").resize((size, size), Image.LANCZOS), dtype="float64")
    n = np.arange(size); k = n.reshape(-1, 1)
    C = np.cos(np.pi * (2 * n + 1) * k / (2 * size)) * np.sqrt(2.0 / size); C[0] /= np.sqrt(2.0)
    d = C @ a @ C.T; med = np.median(d.ravel()[1:]); v = 0
    for b in (d > med).flatten():
        v = (v << 1) | int(b)
    return f"{v:016x}"

def ham(a, b): return (int(a, 16) ^ int(b, 16)).bit_count()

def host(p):
    """Map a container path to a host path via the environment.

    No host path is baked in: set IDDSI_HOST_ROOT when the data lives outside the
    working directory.
    """
    if os.path.exists(p):
        return p
    return p.replace("/work/", os.environ.get("IDDSI_HOST_ROOT", ".") + "/", 1)

man = [json.loads(l) for l in open(f"{DATA}/manifest_v07_weighted.jsonl", encoding="utf-8") if l.strip()]
rec = {}
for ev in ["realweak_pilot", "harvest_d1", "harvest_d1b"]:
    for r in (json.loads(l) for l in open(f"{DATA}/{ev}/events.jsonl", encoding="utf-8") if l.strip()):
        rec.setdefault(str(r["event_id"]), str(r["phash"]))

import PIL
print(f"PIL {PIL.__version__}  numpy {np.__version__}  DATA={DATA}")
h_ex = a_ex = 0; h_d = []; a_d = []; border = []
for row in man:
    e = str(row["event_id"]); p = host(str(row["image_path"]))
    r = rec[e]
    hp = phash_harvest(p); ap = phash_an04(p)
    if hp == r: h_ex += 1
    else: h_d.append(ham(hp, r)); border.append((e, ham(hp, r), r, hp))
    if ap == r: a_ex += 1
    else: a_d.append(ham(ap, r))
print(f"HARVEST(32x32): exact={h_ex}/532 mean_all={statistics.mean([0]*h_ex+h_d):.4f} "
      f"when_diff_mean={statistics.mean(h_d) if h_d else 0:.3f} range={min(h_d) if h_d else 0}-{max(h_d) if h_d else 0}")
print(f"AN04(8x8)     : exact={a_ex}/532 mean_all={statistics.mean([0]*a_ex+a_d):.4f} "
      f"when_diff_mean={statistics.mean(a_d) if a_d else 0:.3f} range={min(a_d) if a_d else 0}-{max(a_d) if a_d else 0}")
print("borderline (harvest algo, !=recorded) corpus histogram:",
      dict(Counter(e.split('_')[1] for e, _, _, _ in border)))
for e, d, r, hp in sorted(border, key=lambda x: -x[1]):
    print(f"   {e}  hamming={d}  recorded={r}  recomputed={hp}")
