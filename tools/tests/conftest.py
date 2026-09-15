"""Make `tools.capture_kit` importable when pytest is invoked from the repo root
together with other subprojects (their conftests reset rootdir/sys.path)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
