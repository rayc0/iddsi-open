#!/usr/bin/env python3
"""IDDSI-Open printable capture kit generator.

Generates ``docs/capture_kit_A4.pdf`` — a 2-page A4 print-scale-accurate kit:

* Page 1: 15 mm fiducial/scale card (exactly 15 mm x 15 mm at 100% print
  scale), a 100 mm ruler strip for verifying print scale, and a 10 mL
  syringe side-view guide with the key capture requirements annotated.
* Page 2: Traditional-Chinese (繁體中文) one-page capture instructions,
  adapted from ``drafts/intern_capture_brief.md``.

Print assumption: **100% scale ("Actual size"), no fit-to-page**.
1 pt = 1/72 inch = 25.4/72 mm; A4 = 210 x 297 mm = 595.28 x 841.89 pt.

Usage:
    python -m tools.capture_kit --out docs/capture_kit_A4.pdf
    python tools/capture_kit.py --out docs/capture_kit_A4.pdf

Deterministic: fixed layout, fixed metadata date, no timestamps.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (headless)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402

# ---------------------------------------------------------------------------
# Exact unit conversion (order of operations kept identical to the spec so
# mm_to_pt(15) == 15 / 25.4 * 72 holds bit-for-bit).
# ---------------------------------------------------------------------------
PT_PER_INCH = 72.0
MM_PER_INCH = 25.4


def mm_to_pt(mm: float) -> float:
    """Convert millimetres to PDF points (1 pt = 1/72 inch)."""
    return mm / MM_PER_INCH * PT_PER_INCH


A4_W_PT = mm_to_pt(210.0)   # 595.2755905511812
A4_H_PT = mm_to_pt(297.0)   # 841.8897637795277
A4_W_IN = 210.0 / MM_PER_INCH
A4_H_IN = 297.0 / MM_PER_INCH

# Fixed metadata date keeps the output byte-deterministic (no wall clock).
FIXED_PDF_DATE = _dt.datetime(2026, 1, 1, 0, 0, 0)

CJK_FONT_CANDIDATES = [
    "Arial Unicode MS",
    "Heiti TC",
    "PingFang HK",
    "Hiragino Sans TC",
    "Hiragino Sans GB",
    "Hiragino Sans",
    "STHeiti",
    "Noto Sans CJK TC",
    "SimHei",
]

# Margins (pt) used on page 1.
MARGIN = mm_to_pt(15)


def pick_cjk_font() -> str | None:
    """Return the first installed CJK-capable font name, else None."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONT_CANDIDATES:
        if name in installed:
            return name
    return None


# ---------------------------------------------------------------------------
# Instruction text (source: drafts/intern_capture_brief.md, adapted)
# ---------------------------------------------------------------------------
ZH_INSTRUCTION_LINES = [
    "IDDSI-Open 拍攝指引（繁體中文）",
    "列印說明：請以 100% 實際大小列印本頁及尺規卡頁（不要「縮放至符合頁面」），",
    "並用第 1 頁的 100 mm 尺規核對列印比例；尺規不符請重新列印。",
    "",
    "一、手機設定與穩定",
    "  1. 使用後置主鏡頭，1080p／30 fps；關閉美顏、濾鏡、人像模式、數碼變焦及水印。",
    "  2. 電話以固定支架或穩定雙手持機，保持水平穩定；拍攝途中不要切換鏡頭或移動機位。",
    "",
    "二、照明與背景",
    "  3. 使用均勻白光或自然光，照明充足；避免背光、強反光及深色陰影。",
    "  4. 背景乾淨整潔：只拍食物、測試工具及尺規卡，移走雜物；不要拍任何人的臉、",
    "     名牌、病歷或可識別個人的資料。",
    "",
    "三、尺規卡（15 mm 尺規卡）",
    "  5. 把第 1 頁印出的 15 mm 尺規卡放在樣本旁邊同一平面，拍攝時尺規卡必須完整入鏡，",
    "     並與鏡頭平面平行，作為比例基準。",
    "",
    "四、針筒側視圖（飲品 L0–L4 流動測試）",
    "  6. 只用項目指定的 10 mL 針筒（筒身約 61.5 mm）；手機固定在針筒正側面，",
    "     拍攝針筒側視圖：針筒垂直、刻度無遮擋、液面清晰可見。",
    "  7. 先拍到準確 10 mL 起始量；完全放開下端時為 0 秒，連續拍攝至少 12 秒，",
    "     不得暫停、剪接或加速。",
    "",
    "五、每個樣本一段影片",
    "  8. 一個檔案只拍一個製備／測試事件；一個樣本一段完整影片，不要把多個樣件",
    "     混在同一段影片。照片用 1x、45° 向下拍，餐碟、食物及標準餐叉完整入鏡。",
    "",
    "六、核對與上傳",
    "  9. 拍完即檢查對焦、比例尺、測試起點及終點是否完整；失焦、遮擋或不完整便重拍，",
    "     不要自行補寫結果或猜測級別（標記 REVIEW）。",
    " 10. 原檔當日上傳，不經即時通訊軟件、不壓縮、不剪片。",
]

EN_INSTRUCTION_LINES = [
    "IDDSI-Open Capture Instructions (English summary)",
    "Print at 100% actual size (no fit-to-page); verify with the 100 mm ruler on page 1.",
    "1. Phone: rear main camera, 1080p/30fps, no filters/beauty/portrait/digital zoom.",
    "   Keep the phone level and steady on a mount for the whole clip; never switch cameras mid-clip.",
    "2. Lighting/background: even white or natural light, well lit, no backlight or glare;",
    "   clean background - no faces, name tags, records, or identifiable personal data.",
    "3. Fiducial card: place the printed 15 mm fiducial card next to the sample on the same",
    "   plane, parallel to the camera; the 15 mm card must be fully in frame as the scale reference.",
    "4. Syringe side view (drinks L0-L4 flow test): use the designated 10 mL syringe; fix the",
    "   phone exactly to the side of the vertical syringe so graduations are unobstructed and",
    "   the liquid level is clearly visible; film from 10 mL start, >= 12 s, unedited.",
    "5. One clip per sample: one file per preparation/test event only; never mix samples in one clip.",
    "6. Check focus, scale, and complete start/end of every test; re-shoot rather than guess",
    "   (mark REVIEW); upload originals the same day, uncompressed.",
]


def build_instructions(cjk_font: str | None = None) -> dict[str, str]:
    """Build the instruction text in both languages.

    Returns ``{"zh": ..., "en": ...}``. If no CJK font is available the
    Traditional-Chinese sheet is replaced by an English fallback notice.
    """
    en = "\n".join(EN_INSTRUCTION_LINES)
    if cjk_font is None:
        zh = "\n".join(
            [
                "NOTE: No CJK font available on this machine.",
                "Traditional-Chinese instructions were replaced by this English fallback.",
                "",
            ]
            + EN_INSTRUCTION_LINES
        )
    else:
        zh = "\n".join(ZH_INSTRUCTION_LINES)
    return {"zh": zh, "en": en}


# ---------------------------------------------------------------------------
# Page 1: fiducial card + ruler + syringe guide
# ---------------------------------------------------------------------------
def _full_page_ax(fig) -> plt.Axes:
    """Axes covering the whole A4 page, 1 data unit = 1 pt, origin bottom-left."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_W_PT)
    ax.set_ylim(0, A4_H_PT)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return ax


def draw_fiducial_card(ax, x_pt: float, y_pt: float, zh: bool = True) -> None:
    """Draw a 15 mm x 15 mm fiducial card with lower-left corner at (x_pt, y_pt)."""
    size = mm_to_pt(15.0)  # exactly 15 mm when printed at 100%
    # Solid black border square (the measurable fiducial).
    ax.add_patch(Rectangle((x_pt, y_pt), size, size, facecolor="black",
                           edgecolor="black", linewidth=0.75))
    # 4x4 white checkerboard inside for high-contrast edge detection.
    cell = size / 4.0
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                ax.add_patch(Rectangle((x_pt + i * cell, y_pt + j * cell),
                                       cell, cell, facecolor="white",
                                       edgecolor="none"))
    # Registration ticks extending outward from the four corners.
    tick = mm_to_pt(4)
    lw = 0.9
    for (cx, cy, dx, dy) in [
        (x_pt, y_pt, -1, 0), (x_pt, y_pt, 0, -1),
        (x_pt + size, y_pt, 1, 0), (x_pt + size, y_pt, 0, -1),
        (x_pt, y_pt + size, -1, 0), (x_pt, y_pt + size, 0, 1),
        (x_pt + size, y_pt + size, 1, 0), (x_pt + size, y_pt + size, 0, 1),
    ]:
        ax.plot([cx, cx + dx * tick], [cy, cy + dy * tick],
                color="black", linewidth=lw, solid_capstyle="butt")
    # Label under the card.
    label = "15 mm 尺規卡" if zh else "15 mm fiducial card"
    ax.text(x_pt + size / 2, y_pt - mm_to_pt(9), label + "  (15 mm x 15 mm)",
            ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(x_pt + size / 2, y_pt - mm_to_pt(14),
            "量度黑框外沿 = 15 mm" if zh else "measure outer black frame = 15 mm",
            ha="center", va="top", fontsize=7.5, color="0.25")


def draw_ruler(ax, x_pt: float, y_pt: float, length_mm: float = 100.0) -> None:
    """Draw a horizontal ruler strip with mm minor ticks and cm major ticks."""
    length = mm_to_pt(length_mm)
    h = mm_to_pt(4)   # body height of the strip
    minor = mm_to_pt(2.2)
    mid = mm_to_pt(3.0)
    major = mm_to_pt(4.0)
    ax.add_patch(Rectangle((x_pt, y_pt), length, h, facecolor="white",
                           edgecolor="black", linewidth=1.0))
    for mm in range(int(length_mm) + 1):
        px = x_pt + mm_to_pt(mm)
        if mm % 10 == 0:
            ax.plot([px, px], [y_pt + h, y_pt + h + major], color="black", linewidth=1.0)
            if mm > 0:
                ax.text(px, y_pt + h + major + mm_to_pt(1.2), f"{mm // 10}",
                        ha="center", va="bottom", fontsize=7)
        elif mm % 5 == 0:
            ax.plot([px, px], [y_pt + h, y_pt + h + mid], color="black", linewidth=0.7)
        else:
            ax.plot([px, px], [y_pt + h, y_pt + h + minor], color="black", linewidth=0.4)
    ax.text(x_pt, y_pt - mm_to_pt(2),
            "0", ha="center", va="top", fontsize=7)
    ax.text(x_pt + length / 2, y_pt - mm_to_pt(3),
            "100 mm 尺規 — 列印後請量度核對比例" if True else "",
            ha="center", va="top", fontsize=8, color="0.25")
    ax.text(x_pt + length, y_pt - mm_to_pt(2),
            "10 cm", ha="center", va="top", fontsize=7)
    ax.text(x_pt, y_pt - mm_to_pt(8),
            "Print at 100% actual size — no fit-to-page. 以 100% 實際大小列印，勿縮放。",
            ha="left", va="top", fontsize=8.5, color="0.15")


def draw_syringe(ax, x_pt: float, y_pt: float, zh: bool = True) -> None:
    """Draw an annotated 10 mL syringe side view with lower-left of the barrel
    at (x_pt, y_pt). Barrel drawn ~61.5 mm long, per the project brief."""
    barrel_len = mm_to_pt(61.5)
    barrel_h = mm_to_pt(14.0)
    # --- barrel ---
    ax.add_patch(Rectangle((x_pt, y_pt), barrel_len, barrel_h,
                           facecolor="white", edgecolor="black", linewidth=1.6))
    # graduation marks: 0 (tip end, left) .. 10 mL (right)
    for ml in range(11):
        px = x_pt + barrel_len * ml / 10.0
        big = ml % 5 == 0
        t = mm_to_pt(5.0) if big else mm_to_pt(3.0)
        ax.plot([px, px], [y_pt, y_pt + t], color="black",
                linewidth=1.1 if big else 0.6)
        if big:
            ax.text(px, y_pt + mm_to_pt(6.2), f"{ml}", ha="center",
                    va="bottom", fontsize=7)
    ax.text(x_pt + barrel_len / 2, y_pt + barrel_h + mm_to_pt(4),
            "mL", ha="center", va="bottom", fontsize=7.5)
    # --- luer tip on the left ---
    tip_l = mm_to_pt(10.0)
    tip_h = mm_to_pt(3.2)
    ax.plot([x_pt, x_pt - tip_l], [y_pt + barrel_h / 2 - tip_h,
                                   y_pt + barrel_h / 2 - tip_h],
            color="black", linewidth=1.4)
    ax.plot([x_pt, x_pt - tip_l], [y_pt + barrel_h / 2 + tip_h,
                                   y_pt + barrel_h / 2 + tip_h],
            color="black", linewidth=1.4)
    ax.plot([x_pt - tip_l, x_pt - tip_l],
            [y_pt + barrel_h / 2 - tip_h, y_pt + barrel_h / 2 + tip_h],
            color="black", linewidth=1.4)
    # --- plunger (partly inserted, from the right) ---
    rod_in = barrel_len * 0.45
    rod_y = y_pt + barrel_h / 2
    ax.add_patch(Rectangle((x_pt + barrel_len - rod_in, rod_y - mm_to_pt(1.2)),
                           rod_in, mm_to_pt(2.4), facecolor="0.85",
                           edgecolor="black", linewidth=1.0))
    # plunger disc
    ax.add_patch(Rectangle((x_pt + barrel_len - rod_in, y_pt + mm_to_pt(0.8)),
                           mm_to_pt(2.0), barrel_h - mm_to_pt(1.6),
                           facecolor="0.6", edgecolor="black", linewidth=1.0))
    # rod out to the flange + flange + thumb rest
    out_l = mm_to_pt(18.0)
    ax.plot([x_pt + barrel_len, x_pt + barrel_len + out_l],
            [rod_y, rod_y], color="black", linewidth=1.6)
    fx = x_pt + barrel_len + out_l
    ax.add_patch(Rectangle((fx, y_pt - mm_to_pt(2)), mm_to_pt(1.6),
                           barrel_h + mm_to_pt(4), facecolor="white",
                           edgecolor="black", linewidth=1.4))
    # barrel flange on the right end of the barrel
    ax.add_patch(Rectangle((x_pt + barrel_len, y_pt - mm_to_pt(1.5)),
                           mm_to_pt(1.4), barrel_h + mm_to_pt(3),
                           facecolor="white", edgecolor="black", linewidth=1.2))
    # --- liquid level hint (shaded column up to 10 mL start) ---
    ax.add_patch(Rectangle((x_pt, y_pt), barrel_len * 0.35, barrel_h,
                           facecolor="0.75", edgecolor="none", zorder=0.5))
    lvl_x = x_pt + barrel_len * 0.35
    ax.plot([lvl_x, lvl_x], [y_pt, y_pt + barrel_h], color="black",
            linewidth=0.8, linestyle=(0, (3, 2)))
    # --- annotations ---
    ann = [
        (x_pt - tip_l, y_pt - mm_to_pt(14),
         "針筒側視圖：手機在正側面" if zh else "side view: phone exactly to the side"),
        (lvl_x, y_pt + barrel_h + mm_to_pt(10),
         "液面必須清晰可見" if zh else "liquid level must be visible"),
        (x_pt + barrel_len + out_l + mm_to_pt(3), rod_y + mm_to_pt(8),
         "10 mL 針筒（筒身約 61.5 mm）" if zh else "10 mL syringe (barrel ~61.5 mm)"),
    ]
    for (tx, ty, s) in ann:
        ha = "left"
        ax.text(tx, ty, s, ha=ha, va="center", fontsize=8.5, color="0.1")
    # arrow marking the side-view camera position
    ax.add_patch(FancyArrow(x_pt - tip_l - mm_to_pt(6),
                            y_pt - mm_to_pt(11), mm_to_pt(8), mm_to_pt(5),
                            width=0.8, head_width=mm_to_pt(2.4),
                            head_length=mm_to_pt(2.8), length_includes_head=True,
                            color="0.35"))


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------
def generate_pdf(out_path: str | Path, cjk_font: str | None = None) -> Path:
    """Generate the 2-page A4 capture kit PDF. Returns the output path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if cjk_font is None:
        cjk_font = pick_cjk_font()
    zh = cjk_font is not None
    instructions = build_instructions(cjk_font)

    if cjk_font:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [cjk_font, "DejaVu Sans"]
    else:
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (keeps CJK glyphs)

    metadata = {
        "Title": "IDDSI-Open Capture Kit (A4)",
        "Author": "IDDSI-Open",
        "Subject": "15 mm fiducial card, 100 mm ruler, 10 mL syringe guide, capture instructions",
        "Creator": "tools/capture_kit.py",
        "CreationDate": FIXED_PDF_DATE,
        "ModDate": FIXED_PDF_DATE,
    }

    with PdfPages(str(out_path), metadata=metadata) as pdf:
        # ------------------------- Page 1 -------------------------
        fig = plt.figure(figsize=(A4_W_IN, A4_H_IN))
        ax = _full_page_ax(fig)
        top = A4_H_PT - MARGIN

        ax.text(MARGIN, top, "IDDSI-Open Capture Kit — 1/2",
                ha="left", va="top", fontsize=16, fontweight="bold")
        ax.text(A4_W_PT - MARGIN, top,
                "100% print / 實際大小列印", ha="right", va="top",
                fontsize=10, color="0.3")

        # fiducial card (upper-left area)
        draw_fiducial_card(ax, MARGIN, top - mm_to_pt(70), zh=zh)

        # ruler strip
        draw_ruler(ax, MARGIN, top - mm_to_pt(115))

        # syringe guide (lower half of the page)
        ax.text(MARGIN, top - mm_to_pt(140),
                "10 mL 針筒側視圖 Syringe side-view guide" if zh
                else "10 mL syringe side-view guide",
                ha="left", va="top", fontsize=13, fontweight="bold")
        draw_syringe(ax, MARGIN + mm_to_pt(20), top - mm_to_pt(230), zh=zh)

        fig.savefig(pdf, format="pdf")
        plt.close(fig)

        # ------------------------- Page 2 -------------------------
        fig = plt.figure(figsize=(A4_W_IN, A4_H_IN))
        ax2 = _full_page_ax(fig)
        ax2.text(MARGIN, A4_H_PT - MARGIN, "IDDSI-Open Capture Kit — 2/2",
                 ha="left", va="top", fontsize=16, fontweight="bold")
        body = instructions["zh"] if zh else instructions["en"]
        ax2.text(MARGIN, A4_H_PT - MARGIN - mm_to_pt(14), body,
                 ha="left", va="top", fontsize=9.2, linespacing=1.45)
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tools.capture_kit",
        description="Generate the IDDSI-Open A4 capture kit PDF "
                    "(fiducial card + ruler + syringe guide + instructions).",
    )
    parser.add_argument("--out", default="docs/capture_kit_A4.pdf",
                        help="output PDF path (default: docs/capture_kit_A4.pdf)")
    args = parser.parse_args(argv)

    font = pick_cjk_font()
    out = generate_pdf(args.out, cjk_font=font)
    print(f"font (CJK): {font if font else 'NONE — English fallback used'}")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
