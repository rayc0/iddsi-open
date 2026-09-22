"""Tests for ops/flux_image_check.py — the FLUX load-test artifact assertions.

Run from repo root:
    python -m pytest ops/tests/test_flux_image_check.py -q

No GPU needed: fixtures are generated locally.  A solid-colour PNG (stddev
exactly 0, like a blank/black frame) must FAIL the variance floor, while a
noise PNG (large, high-variance, like a real photograph) must PASS the
production thresholds.
"""

import os
import random
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flux_image_check import (  # noqa: E402
    MIN_BYTES_DEFAULT,
    STDDEV_FLOOR_DEFAULT,
    ImageCheckError,
    check_image,
    image_stats,
)


def test_production_thresholds_are_what_the_script_claims():
    assert MIN_BYTES_DEFAULT == 200 * 1024
    assert STDDEV_FLOOR_DEFAULT == 5.0


def _solid_png(path, size=(512, 512), color=(18, 94, 160)):
    Image.new("RGB", size, color).save(path)
    return path


def _noise_png(path, size=(512, 512), seed=7):
    rng = random.Random(seed)
    Image.frombytes("RGB", size, rng.randbytes(size[0] * size[1] * 3)).save(
        path, compress_level=0
    )
    return path


def test_solid_colour_fails_variance_floor_even_with_size_waived(tmp_path):
    png = _solid_png(str(tmp_path / "solid.png"))
    assert image_stats(png).stddev == 0.0
    # min_bytes=0 isolates the variance assertion from the size assertion.
    with pytest.raises(ImageCheckError, match="stddev"):
        check_image(png, min_bytes=0)


def test_solid_colour_also_fails_production_size_floor(tmp_path):
    png = _solid_png(str(tmp_path / "solid.png"))
    with pytest.raises(ImageCheckError, match="too small"):
        check_image(png)


def test_noise_passes_production_thresholds(tmp_path):
    png = _noise_png(str(tmp_path / "noise.png"))
    stats = check_image(png)
    assert stats.size_bytes >= MIN_BYTES_DEFAULT
    assert stats.width == 512 and stats.height == 512
    assert stats.stddev >= STDDEV_FLOOR_DEFAULT


def test_missing_file_fails(tmp_path):
    with pytest.raises(ImageCheckError, match="missing"):
        check_image(str(tmp_path / "never_written.png"))


def test_small_file_fails_size_floor(tmp_path):
    png = _solid_png(str(tmp_path / "tiny.png"), size=(8, 8))
    with pytest.raises(ImageCheckError, match="too small"):
        check_image(png)


def test_checker_cli_reports_measurements_and_exit_codes(tmp_path):
    import subprocess

    solid = _solid_png(str(tmp_path / "solid.png"))
    noise = _noise_png(str(tmp_path / "noise.png"))
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flux_image_check.py")

    bad = subprocess.run([sys.executable, script, solid], text=True, capture_output=True)
    assert bad.returncode == 1
    assert "FAIL" in bad.stderr

    good = subprocess.run([sys.executable, script, noise], text=True, capture_output=True)
    assert good.returncode == 0, good.stderr
    assert "bytes=" in good.stdout and "stddev=" in good.stdout
    assert "PASS" not in good.stdout  # only flux_load_test.sh owns the PASS line
