"""Ensure the trainer package directory is importable when pytest is invoked
from the repository root (e.g. `python -m pytest -q trainer`). Pytest collects
this conftest before importing test modules, so inserting the directory here
makes `train`, `eval`, `explain`, and `fixtures` resolvable without requiring
an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_DIR = str(Path(__file__).resolve().parent)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)
