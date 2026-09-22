"""Tests for the guarded FLUX rsync-temp cleanup script."""

import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "cleanup_rsync_temps.sh"
MIN_FINAL_BYTES = 23_000_000_000


def run_cleanup(flux_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["FLUX_DIR"] = str(flux_dir)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_then_apply_with_verified_sparse_final(tmp_path: Path) -> None:
    final_file = tmp_path / "flux1-schnell.safetensors"
    temp_file = tmp_path / ".flux1-schnell.safetensors.ABC123"
    final_file.touch()
    os.truncate(final_file, MIN_FINAL_BYTES)
    temp_file.write_bytes(b"partial rsync data")

    dry_run = run_cleanup(tmp_path)
    assert dry_run.returncode == 0, dry_run.stderr
    assert "mode=DRY-RUN" in dry_run.stdout
    assert f"WOULD-DELETE  {temp_file}" in dry_run.stdout
    assert temp_file.exists()

    apply = run_cleanup(tmp_path, "--apply")
    assert apply.returncode == 0, apply.stderr
    assert "mode=APPLY" in apply.stdout
    assert f"DELETED  {temp_file}" in apply.stdout
    assert not temp_file.exists()
    assert final_file.exists()
    assert final_file.stat().st_size == MIN_FINAL_BYTES


def test_apply_refuses_temp_when_final_is_below_size_floor(tmp_path: Path) -> None:
    final_file = tmp_path / "flux1-schnell.safetensors"
    temp_file = tmp_path / ".flux1-schnell.safetensors.SMALL1"
    final_file.touch()
    os.truncate(final_file, MIN_FINAL_BYTES - 1)
    temp_file.write_bytes(b"partial rsync data")

    apply = run_cleanup(tmp_path, "--apply")
    assert apply.returncode == 2
    assert "SKIP" in apply.stderr
    assert temp_file.exists()
    assert final_file.exists()
