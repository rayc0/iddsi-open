"""Artifact assertions for the FLUX GPU load test (ops/flux_load_test.sh).

One implementation serves two callers, so the unit test pins the exact logic
the box runs:

- inside the DGX Spark container, ``flux_load_test.sh`` runs
  ``python3 /work/repo/ops/flux_image_check.py <png>`` and asserts on the
  exit code plus the printed measurements;
- locally, ``ops/tests/test_flux_image_check.py`` calls :func:`check_image`
  with fixture images (no GPU needed).

A generated image passes only if it is a real file with non-trivial size and
non-trivial pixel variance.  A blank/black frame has every pixel equal, so
its population stddev is 0 and it fails the variance floor.  Exit codes and
log lines alone never count — only these measurements do.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from PIL import Image, ImageStat

#: Minimum file size for a 1024x1024 FLUX PNG.  A blank or truncated write is
#: orders of magnitude smaller; a real photographic PNG is several MB.
MIN_BYTES_DEFAULT = 200 * 1024

#: Floor for the grayscale population stddev.  Solid-colour and blank frames
#: measure exactly 0; real food photographs measure in the tens.
STDDEV_FLOOR_DEFAULT = 5.0


class ImageCheckError(ValueError):
    """Raised when a load-test image fails an artifact assertion."""


@dataclass(frozen=True)
class ImageStats:
    path: str
    size_bytes: int
    width: int
    height: int
    stddev: float


def image_stats(path: str) -> ImageStats:
    """Measure a PNG on disk.  Raises :class:`ImageCheckError` if unreadable."""
    if not os.path.isfile(path):
        raise ImageCheckError(f"image missing (never written?): {path}")
    size_bytes = os.path.getsize(path)
    try:
        with Image.open(path) as img:
            width, height = img.size
            stddev = float(ImageStat.Stat(img.convert("L")).stddev[0])
    except Exception as exc:
        raise ImageCheckError(f"unreadable image {path}: {exc}") from exc
    return ImageStats(
        path=path, size_bytes=size_bytes, width=width, height=height, stddev=stddev
    )


def check_image(
    path: str,
    *,
    min_bytes: int = MIN_BYTES_DEFAULT,
    stddev_floor: float = STDDEV_FLOOR_DEFAULT,
) -> ImageStats:
    """Assert size and pixel variance; return stats or raise ImageCheckError."""
    stats = image_stats(path)
    failures: list[str] = []
    if stats.size_bytes < min_bytes:
        failures.append(f"file too small: {stats.size_bytes} bytes < {min_bytes} minimum")
    if stats.width <= 0 or stats.height <= 0:
        failures.append(f"degenerate dimensions: {stats.width}x{stats.height}")
    if stats.stddev < stddev_floor:
        failures.append(
            f"pixel stddev {stats.stddev:.2f} < floor {stddev_floor} "
            "(blank, black, or solid-colour image?)"
        )
    if failures:
        raise ImageCheckError(
            "; ".join(failures)
            + f" [bytes={stats.size_bytes} dimensions={stats.width}x{stats.height}"
            f" stddev={stats.stddev:.2f}]"
        )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="PNG to assert on")
    parser.add_argument("--min-bytes", type=int, default=MIN_BYTES_DEFAULT)
    parser.add_argument("--stddev-floor", type=float, default=STDDEV_FLOOR_DEFAULT)
    args = parser.parse_args(argv)
    try:
        stats = check_image(args.path, min_bytes=args.min_bytes, stddev_floor=args.stddev_floor)
    except ImageCheckError as exc:
        print(f"flux_image_check FAIL: {exc}", file=sys.stderr)
        return 1
    # Measurements only — no PASS line here.  flux_load_test.sh owns the
    # single PASS line and prints it only after every assertion has passed.
    print(
        f"flux_image_check bytes={stats.size_bytes} "
        f"width={stats.width} height={stats.height} stddev={stats.stddev:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
