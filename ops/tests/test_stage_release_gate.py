"""The release-staging gate, enforced on every run rather than only at publish.

``ops/stage_release.sh`` assembles the exact bytes that
``.github/workflows/publish-hf.yml`` uploads to Hugging Face, and
``ops/scan_release_leaks.sh`` is the leak scan that runs over those bytes. The
workflow already runs the pair on every publish; this module runs them on every
commit, so a leak that a wave reintroduces fails here instead of surfacing at
publish time.

This is the gate that caught (lane rel03_leakscan, 2026-10-12) three classes of
defect that had survived wave 1: hardcoded DGX host paths in ``tools/`` and
``trainer/eval/``, a CN mirror hostname in ``trainer/eval/hub_source.py``, and a
full 684-row dataset manifest staged into ``results/``.

Runs under pytest or standalone -- the latter matters because the trainer
container has no ``rsync``, so the staging step can only run on the host:

    python -m pytest ops/tests/test_stage_release_gate.py -q
    python ops/tests/test_stage_release_gate.py

It rebuilds ``_release_staging/`` (gitignored) and skips when ``ops/`` or
``rsync`` is absent -- ``ops/`` is deliberately NOT part of the published tree,
so a checkout of the staged release has no staging script to test.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE_SCRIPT = REPO / "ops" / "stage_release.sh"
STAGED = REPO / "_release_staging" / "v0"
MANIFEST = REPO / "_release_staging" / "MANIFEST.txt"

# Weight, archive and dataset suffixes. The publish workflow refuses
# *.pt/*.safetensors/*.bin/*.npz/*.jpg/*.jpeg/*.png; this list is deliberately
# broader so a format the workflow does not name yet still fails here.
FORBIDDEN_SUFFIXES = {
    # weights / checkpoints
    ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".h5", ".onnx", ".pb",
    ".npz", ".npy", ".pkl", ".pickle", ".joblib", ".tflite", ".gguf",
    # archives
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",
    # image / video / audio media
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wav", ".mp3", ".flac",
    # dataset / tabular payloads
    ".jsonl", ".ndjson", ".csv", ".parquet", ".db", ".sqlite", ".sqlite3",
    ".arrow", ".feather", ".tfrecord",
}

# DGX box-ops carriers that hardcode the host root, private lane checkout paths,
# container names and the GPU slot gate. Not release content; kept on the box.
BOX_OPS = (
    "tools/collect_kappa.py",
    "tools/run_arm.sh",
    "trainer/tools/run_lr_sweep.sh",
)

try:
    import pytest
except ImportError:  # standalone run on the host, which has no pytest
    pytest = None  # type: ignore[assignment]

_SKIP_REASON = (
    "ops/stage_release.sh or rsync not present "
    "(published-tree checkout, or the trainer container, which has no rsync)"
)
if pytest is not None and (not STAGE_SCRIPT.is_file() or shutil.which("rsync") is None):
    pytestmark = pytest.mark.skip(reason=_SKIP_REASON)

_STAGED_PROC: subprocess.CompletedProcess[str] | None = None


def _stage() -> subprocess.CompletedProcess[str]:
    """Run ops/stage_release.sh once and cache the result for every check."""
    global _STAGED_PROC
    if _STAGED_PROC is None:
        _STAGED_PROC = subprocess.run(
            ["bash", str(STAGE_SCRIPT)],
            cwd=str(REPO),
            text=True,
            capture_output=True,
            check=False,
        )
    return _STAGED_PROC


def _suffix_offenders() -> list[str]:
    return sorted(
        str(p.relative_to(REPO))
        for p in STAGED.rglob("*")
        if p.is_file() and p.suffix.lower() in FORBIDDEN_SUFFIXES
    )


def _manifest_rows() -> list[str]:
    return [line.strip() for line in MANIFEST.read_text().splitlines() if line.strip()]


def _tree_rows() -> list[str]:
    """The same "./relative/path" form `find . -type f | sort` emits."""
    return sorted(
        "./" + str(p.relative_to(STAGED)) for p in STAGED.rglob("*") if p.is_file()
    )


def _sealed_files() -> list[str]:
    return sorted(str(p.relative_to(REPO)) for p in STAGED.rglob("*sealed*"))


def _box_ops_present() -> list[str]:
    return [f for f in BOX_OPS if (STAGED / f).is_file()]


def _checks() -> list[tuple[str, bool, str]]:
    """(name, passed, detail) for every gate, in report order."""
    if not STAGE_SCRIPT.is_file() or shutil.which("rsync") is None:
        return [("gate runnable", True, f"SKIPPED: {_SKIP_REASON}")]

    proc = _stage()
    out: list[tuple[str, bool, str]] = []

    out.append((
        "ops/stage_release.sh exits 0",
        proc.returncode == 0,
        "exit 0" if proc.returncode == 0 else
        f"exit {proc.returncode}\n--- stdout tail ---\n{proc.stdout[-4000:]}"
        f"\n--- stderr tail ---\n{proc.stderr[-2000:]}",
    ))
    out.append((
        "leak scan reports no BLOCK hits",
        "RESULT: ✅ no BLOCK hits" in proc.stdout,
        proc.stdout[-4000:],
    ))

    if not STAGED.is_dir():
        out.append(("staged tree produced", False, f"{STAGED} missing"))
        return out
    out.append(("staged tree produced", True, str(STAGED)))

    offenders = _suffix_offenders()
    out.append((
        f"no weight/dataset/media file staged ({len(FORBIDDEN_SUFFIXES)} suffixes refused)",
        not offenders,
        f"{len(offenders)} offender(s): {offenders}" if offenders else "none",
    ))

    sealed = _sealed_files()
    out.append((
        "no sealed evaluation file staged",
        not sealed,
        f"staged: {sealed}" if sealed else "none",
    ))

    box_ops = _box_ops_present()
    out.append((
        "no box-ops host-path carrier staged",
        not box_ops,
        f"staged: {box_ops}" if box_ops else "none",
    ))

    # Compare as sets: `find | sort` uses the shell locale's collation, which is
    # not Python's codepoint order, so an ordered comparison would be comparing
    # two sort algorithms rather than two file lists.
    listed, actual = set(_manifest_rows()), set(_tree_rows())
    out.append((
        "MANIFEST.txt matches the staged tree",
        listed == actual,
        f"MANIFEST lists {len(listed)}, tree holds {len(actual)}; "
        f"only-in-MANIFEST={sorted(listed - actual)[:5]} "
        f"only-in-tree={sorted(actual - listed)[:5]}",
    ))
    return out


def test_stage_release_script_exits_zero() -> None:
    name, ok, detail = _checks()[0]
    assert ok, f"{name}: {detail}"


def test_leak_scan_reports_no_block_hits() -> None:
    name, ok, detail = _checks()[1]
    assert ok, f"{name}: {detail}"


def test_no_weight_dataset_or_media_file_is_staged() -> None:
    name, ok, detail = _checks()[4]
    assert ok, f"{name}: {detail}"


def test_no_sealed_evaluation_file_is_staged() -> None:
    name, ok, detail = _checks()[5]
    assert ok, f"{name}: {detail}"


def test_no_box_ops_carrier_is_staged() -> None:
    name, ok, detail = _checks()[6]
    assert ok, f"{name}: {detail}"


def test_manifest_matches_the_staged_tree() -> None:
    name, ok, detail = _checks()[7]
    assert ok, f"{name}: {detail}"


def main() -> int:
    """Standalone entry point: prints every gate and exits non-zero on failure."""
    failures = 0
    for name, ok, detail in _checks():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failures += 1
            print(f"      {detail}")
    print(f"\n{len(_checks()) - failures}/{len(_checks())} gates passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
