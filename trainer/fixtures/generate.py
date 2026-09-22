from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


COLORS = {
    3: (70, 120, 190),
    4: (80, 165, 145),
    5: (205, 175, 70),
    6: (210, 115, 70),
    7: (155, 80, 150),
}


def _texture(level: int, size: int, rng: np.random.Generator) -> Image.Image:
    base = np.empty((size, size, 3), dtype=np.float32)
    base[:] = COLORS[level]
    yy, xx = np.mgrid[:size, :size]
    frequency = level - 1
    pattern = 18.0 * np.sin(xx / frequency) * np.cos(yy / (frequency + 1))
    noise = rng.normal(0, 5 + 2 * (level - 3), (size, size, 1))
    array = np.clip(base + pattern[..., None] + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, "RGB")
    draw = ImageDraw.Draw(image)
    radius = max(2, level - 1)
    for _ in range(level * 2):
        x, y = rng.integers(radius, size - radius, size=2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(245, 245, 245), width=1)
    return image


def generate_fixture(output_dir: str | Path, *, per_level_per_split: int = 3, size: int = 48, seed: int = 7) -> Path:
    """Generate visually separable fake textures to test plumbing only."""
    output = Path(output_dir)
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = output / "manifest.jsonl"
    records = []
    for split in ("train", "val", "test"):
        for level in range(3, 8):
            for sample in range(per_level_per_split):
                event_id = f"synthetic_{split}_L{level}_{sample:03d}"
                path = images / f"{event_id}.png"
                _texture(level, size, rng).save(path)
                records.append(
                    {
                        "event_id": event_id,
                        "image_path": str(path.relative_to(output)),
                        "level": level,
                        "split": split,
                        "group_id": event_id,
                        "provenance": "synthetic_colored_texture",
                        "physically_tested": False,
                        "release_eligible": False,
                    }
                )
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    (output / "SYNTHETIC_ONLY.txt").write_text(
        "SYNTHETIC PIPELINE-DEVELOPMENT FIXTURE. NOT PHYSICALLY TESTED. DO NOT RELEASE OR REPORT AS PERFORMANCE.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic colored-texture pipeline fixtures")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-level-per-split", type=int, default=3)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(generate_fixture(args.output_dir, per_level_per_split=args.per_level_per_split, size=args.size, seed=args.seed))


if __name__ == "__main__":
    main()
