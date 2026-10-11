"""Make the repository root importable for the Space tests.

Pytest inserts ``space/tests`` on ``sys.path`` (rootdir-relative, no
``__init__.py`` here), which is not enough for ``from space import core``.  This
conftest adds the repository root, and also the trainer directory so the CORAL
port can be compared against ``train.model`` without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINER_DIR = _REPO_ROOT / "trainer"

for _path in (str(_REPO_ROOT), str(_TRAINER_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
