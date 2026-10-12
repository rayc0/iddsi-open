#!/usr/bin/env python3
"""Locate the shared IDDSI-Open working tree without naming a host path.

The lane clones live under ``<root>/lanes2/<lane>/`` on the DGX box and under
``<root>/lanes/<lane>/`` in older lanes; the shared inputs (``data/``, ``runs/``,
``models_local/``, ``lane_runs/``) live one or two levels above the clone. Every
script in ``tools/`` needs those inputs, and every one of them used to hardcode
the box's absolute paths to find them -- which the release leak scan BLOCKs on
(``ops/scan_release_leaks.sh``, the "internal infra" rule).

Discovery order, first match wins:

1. ``IDDSI_ROOT`` (explicit override, wins over everything);
2. the current working directory and its parents, then this file's parents --
   so a lane clone finds the shared tree it sits inside, on the box or anywhere
   else, with no absolute path written down;
3. ``None`` -- callers decide whether that is fatal or a skip.

Nothing here reaches the network and nothing here writes.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_ROOT = "IDDSI_ROOT"
ENV_MODEL_DIR = "IDDSI_MODEL_DIR"

# How far above this file to look for the shared tree. A lane clone is
# <root>/lanes2/<lane>/tools/, i.e. four parents up; the extra two cover
# deeper nesting and older <root>/lanes/<lane>/tools/ layouts.
_MAX_PARENTS = 6

# The marker that identifies the shared tree. Every lane script needs the v0.7
# manifest, so it is the canonical probe.
_MARKER = Path("data") / "manifest_v07_weighted.jsonl"


def _candidates() -> list[Path]:
    """Directories to try, most specific first, none of them hardcoded."""
    out: list[Path] = []
    env = os.environ.get(ENV_ROOT, "").strip()
    if env:
        out.append(Path(env))
    try:
        out.append(Path.cwd().resolve())
    except OSError:  # pragma: no cover - cwd deleted underneath us
        pass
    here = Path(__file__).resolve()
    out.extend(here.parents[:_MAX_PARENTS])
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in out:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def find_root() -> Path | None:
    """The shared IDDSI-Open tree, or None when it is not reachable."""
    for candidate in _candidates():
        if (candidate / _MARKER).is_file():
            return candidate
    return None


def require_root() -> Path:
    """``find_root`` or a fail-closed ``SystemExit`` with an actionable message."""
    root = find_root()
    if root is None:
        raise SystemExit(
            f"cannot locate the shared IDDSI-Open tree (no {_MARKER} under any "
            f"candidate). Set {ENV_ROOT} to it explicitly."
        )
    return root


def find_model_dir() -> Path | None:
    """The local SigLIP-2 base checkpoint directory, or None when absent.

    ``IDDSI_MODEL_DIR`` wins; otherwise ``<root>/models_local/`` is used, which
    is where the shared read-only copy lives. No absolute host path is named.
    """
    env = os.environ.get(ENV_MODEL_DIR, "").strip()
    if env:
        candidate = Path(env)
        if candidate.is_dir():
            return candidate
    root = find_root()
    if root is None:
        return None
    candidate = root / "models_local" / "siglip2-base-patch16-224"
    return candidate if candidate.is_dir() else None
