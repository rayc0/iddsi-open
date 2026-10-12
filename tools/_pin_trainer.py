"""Pin ``train`` / ``eval`` to THIS lane's trainer clone.

The shared ``iddsi-train`` container has the trainer installed editable against
whichever lane last ran ``pip install -e`` — at the time this lane started that
was ``/work/lanes/qa01_tests/trainer``.  Importing ``train.model`` without this
pin would silently score another lane's code.  Insert this lane's trainer at the
front of ``sys.path`` and drop any already-imported copy before the real import.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

TRAINER_DIR = Path("/work/lanes2/fix01_decode/trainer")

for name in [m for m in list(sys.modules) if m == "train" or m.startswith("train.") or m == "eval" or m.startswith("eval.")]:
    del sys.modules[name]

if str(TRAINER_DIR) in sys.path:
    sys.path.remove(str(TRAINER_DIR))
sys.path.insert(0, str(TRAINER_DIR))

MODULES = [
    "train/model.py",
    "train/data.py",
    "train/losses.py",
    "train/engine.py",
    "eval/metrics.py",
    "eval/calibration.py",
    "eval/report.py",
    "eval/abstention.py",
]


def provenance() -> dict[str, str]:
    out = {}
    for relative in MODULES:
        path = TRAINER_DIR / relative
        out[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    import train
    import eval

    out["train.__file__"] = train.__file__
    out["eval.__file__"] = eval.__file__
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(provenance(), indent=1))
