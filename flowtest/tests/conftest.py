"""Make the flowtest package importable regardless of pytest's invocation dir.

Allows both ``cd flowtest && pytest`` and ``cd iddsi-open && pytest flowtest``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)
if PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, PACKAGE_PARENT)
