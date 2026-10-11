"""Pin ``train``/``eval``/``explain``/``fixtures`` imports to THIS lane's clone.

Why this exists
---------------
The 32 lanes share one container (``iddsi-train``) and therefore one
site-packages.  The editable install of ``iddsi_open_trainer`` in that
site-packages maps the top-level package names to whichever lane ran
``pip install -e`` last.  Observed on 2026-10-12:

    MAPPING = {'eval': '/work/lanes/qa01_tests/trainer/eval', ...}

``/work/lanes/qa01_tests/trainer/eval/metrics.py`` and ``report.py`` differ from
this clone's copies, so an unpinned ``import eval.metrics`` silently evaluates
with another lane's working tree.  Any number computed that way is not
reproducible from this lane's files.

This module drops the editable meta-path finders and puts this lane's
``trainer/`` directory at the front of ``sys.path`` *before* the first
``train``/``eval`` import, so every audit number is produced by the code that
is committed in this clone.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

TRAINER_DIR = Path(__file__).resolve().parents[1] / "trainer"
PACKAGES = ("train", "eval", "explain", "fixtures")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(trainer_dir: str | Path | None = None) -> dict[str, str]:
    """Make this lane's ``trainer/`` authoritative for the pinned packages."""

    directory = str(Path(trainer_dir) if trainer_dir else TRAINER_DIR)
    # Editable installs register a MetaPathFinder, which is consulted before
    # sys.path; remove every editable finder so it cannot shadow our copy.
    sys.meta_path = [
        finder for finder in sys.meta_path if not type(finder).__module__.startswith("__editable__")
    ]
    for name in PACKAGES:
        sys.modules.pop(name, None)
    if directory in sys.path:
        sys.path.remove(directory)
    sys.path.insert(0, directory)
    provenance: dict[str, str] = {"trainer_dir": directory}
    import eval  # noqa: PLC0415  (must happen after the path is pinned)
    import train  # noqa: PLC0415

    provenance["eval__file__"] = eval.__file__
    provenance["train__file__"] = train.__file__
    for module_name in ("eval.metrics", "eval.abstention", "eval.calibration", "train.model", "train.engine"):
        __import__(module_name)
        module = sys.modules[module_name]
        provenance[f"{module_name}__file__"] = module.__file__
        provenance[f"{module_name}__sha256"] = _sha256(Path(module.__file__))
    return provenance


if __name__ == "__main__":
    import json

    print(json.dumps(pin(), indent=2))
