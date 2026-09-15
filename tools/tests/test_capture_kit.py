#!/usr/bin/env python3
"""Tests for tools/capture_kit.py (W38).

Run from repo root:  python -m pytest tools/tests/test_capture_kit.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `tools` importable regardless of how pytest was invoked.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.capture_kit import (  # noqa: E402
    A4_H_PT,
    A4_W_PT,
    build_instructions,
    generate_pdf,
    mm_to_pt,
)


def test_mm_to_pt_exact():
    # 1 pt = 1/72 inch = 25.4/72 mm  ->  15 mm = 15/25.4*72 pt exactly.
    assert mm_to_pt(15) == 15 / 25.4 * 72
    assert mm_to_pt(0) == 0.0
    assert mm_to_pt(25.4) == pytest.approx(72.0)
    # A4 = 210 x 297 mm -> 595.2755... x 841.8897... pt
    assert A4_W_PT == pytest.approx(595.28, abs=0.01)
    assert A4_H_PT == pytest.approx(841.89, abs=0.01)
    # the fiducial square side in points, referenced by the drawing code
    assert mm_to_pt(15) == pytest.approx(42.5197, abs=1e-4)


def test_cli_generates_valid_a4_pdf(tmp_path):
    out = tmp_path / "capture_kit_A4.pdf"
    generate_pdf(out)

    data = out.read_bytes()
    assert data[:5] == b"%PDF-", "missing %PDF header"
    assert len(data) > 10_000, "PDF suspiciously small"

    pypdf = pytest.importorskip("pypdf", reason="pypdf not installed")
    reader = pypdf.PdfReader(out)
    assert 1 <= len(reader.pages) <= 2, f"expected 1-2 pages, got {len(reader.pages)}"
    box = reader.pages[0].mediabox
    assert (float(box.width), float(box.height)) == pytest.approx(
        (595.2755905511812, 841.8897637795277), abs=0.1
    ), f"page 1 not A4: {float(box.width)} x {float(box.height)} pt"


def test_build_instructions_bilingual_keywords():
    instructions = build_instructions(cjk_font="Heiti TC")  # force zh sheet
    zh, en = instructions["zh"], instructions["en"]

    # EN: scale reference + syringe mention
    assert "15 mm" in en
    assert "syringe" in en.lower()
    assert "100%" in en

    # ZH (繁體中文): required keywords
    assert "15 mm" in zh
    assert "針筒" in zh
    assert "側視圖" in zh            # syringe side view
    assert "尺規卡" in zh            # fiducial card in frame
    assert "100%" in zh
    assert "照明" in zh              # good lighting
    assert "一段影片" in zh          # one clip per sample
    assert "背景" in zh              # clean background
    assert "水平穩定" in zh          # phone steady/level

    # fallback path when no CJK font
    fallback = build_instructions(cjk_font=None)
    assert "15 mm" in fallback["zh"]
    assert "syringe" in fallback["zh"].lower()


def test_output_deterministic(tmp_path):
    """Same input -> byte-identical PDF (fixed seed/layout, no timestamps)."""
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    generate_pdf(a)
    generate_pdf(b)
    assert a.read_bytes() == b.read_bytes(), "PDF output is not deterministic"
    assert b"CreationDate (D:20260101" in a.read_bytes()
