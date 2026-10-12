"""Tests for ops/build_capture_protocol.py — the specimen-card section parser.

The defect these pin down (found by qa03_capture, 2026-10-12):

  bf52_sweet_and_sour_pork_L7's SHOTS header read
      SHOTS — 3 stills + 2 clips: sweet_and_sour_pork_L7_take1.jpg — ...
  a bare section name with prose on the same line. It matched neither
  SECTION_NUM (no leading "4.") nor SECTION_BARE (the line does not end after
  the name), so no section boundary fired: the whole SHOTS section — every media
  filename the operator has to capture — was folded into the FAIL MODES buffer.
  The card shipped shots="" and media_expected=[] and the build still printed
  "missing required sections: 0", because that check looked at PREP and IDDSI
  TEST only, so an empty SHOTS was not an error.

  The card itself was rewritten into the numbered shape by commit a1f4851, which
  hid the symptom; the parser hole stayed open, so any card written that way
  would have lost its media list again. HISTORICAL_BF52 below is that card
  verbatim, and it is the regression fixture.

Run from repo root:
    python -m pytest ops/tests/test_build_capture_protocol.py -q
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import build_capture_protocol as bcp  # noqa: E402

# The card as it stood before a1f4851: SHOTS is an unnumbered bare header with
# its prose inline, sitting after "4. FAIL MODES".
HISTORICAL_BF52 = """1. PREP

Bought form (BASE_FOOD_LIST BF52): takeaway sweet and sour pork. Weigh 200 g of
the takeaway onto the capture tray. Do NOT blend, sieve, chop, dice, or add water
or thickener: L7 is the food as bought and normally served.

2. IDDSI TEST

5.1 Fork pressure — ACTIVE, the only deciding test at L7. Press the bowl of the
fork down with the thumb until the thumbnail blanches white.

3. VISUAL QC

(1) Distinct battered pieces, individually visible.

4. FAIL MODES

(1) Piece squashes flat, smears, or extrudes through the tines under the
blanching press → softer than L7.

SHOTS — 3 stills + 2 clips: sweet_and_sour_pork_L7_take1.jpg — 45° hero, 100 mm ruler + 15 mm checkerboard in frame, 3–4 pieces visible. sweet_and_sour_pork_L7_take2.jpg — top-down, checkerboard only, full 200 g portion. sweet_and_sour_pork_L7_take3.jpg — fork-press close-up, tines mid-press into the largest piece, ruler in frame for the >= 15 mm read. sweet_and_sour_pork_L7_take1.mp4 — 8–12 s unbroken fork-pressure take. sweet_and_sour_pork_L7_take2.mp4 — 8–12 s full IDDSI test film (fork pressure + ruler read).
"""

MEDIA = [
    "sweet_and_sour_pork_L7_take1.jpg",
    "sweet_and_sour_pork_L7_take2.jpg",
    "sweet_and_sour_pork_L7_take3.jpg",
    "sweet_and_sour_pork_L7_take1.mp4",
    "sweet_and_sour_pork_L7_take2.mp4",
]

SHOTS_PROSE = "3 stills + 2 clips"

NUMBERED_INLINE = """1. PREP — Weigh 200 g onto the tray. Add 8 g thickener, add 120 ml water, blend 30 s.
2. IDDSI TEST — 5.1 Fork pressure, ACTIVE.
3. VISUAL QC — (1) Distinct pieces.
4. SHOTS — 3 stills: bf99_test_dish_L7_take1.jpg, bf99_test_dish_L7_take2.jpg.
5. FAIL MODES — (1) Piece squashes flat.
"""

BARE_PROSE_BELOW = """PREP

Weigh 200 g onto the tray.

IDDSI TEST

5.1 Fork pressure, ACTIVE.

VISUAL QC

(1) Distinct pieces.

SHOTS

bf99_test_dish_L7_take1.jpg — 45° hero with ruler.

FAIL MODES

(1) Piece squashes flat.
"""

# A header a human reads as a boundary that no section pattern knows: the number
# is closed with ")" instead of ".". Nothing can safely guess where it starts, so
# the requirement is that it is REPORTED rather than swallowed.
UNPLACEABLE_HEADER = """1. PREP — Weigh 200 g onto the tray.
2. IDDSI TEST — 5.1 Fork pressure, ACTIVE.
3. VISUAL QC — (1) Distinct pieces.
4) SHOTS — 3 stills: bf99_test_dish_L7_take1.jpg.
5. FAIL MODES — (1) Piece squashes flat.
"""


@pytest.fixture
def card_dir(tmp_path):
    d = tmp_path / "docs" / "specimen_cards"
    d.mkdir(parents=True)
    return d


def write_card(card_dir, name, text):
    p = card_dir / name
    p.write_text(text, encoding="utf-8")
    return p


# --- the defect itself -------------------------------------------------------

def test_bare_header_with_inline_prose_starts_a_section(card_dir):
    """The shape that was swallowed: 'SHOTS — <prose>' on one line."""
    p = write_card(card_dir, "bf52_sweet_and_sour_pork_L7.md", HISTORICAL_BF52)
    s = bcp.parse_card(p)

    assert s["shots"], "the SHOTS section was swallowed"
    assert s["shots"].startswith(SHOTS_PROSE)
    assert s["media_expected"] == MEDIA
    assert s["orphan_header_lines"] == []

    # and the prose must not have been absorbed by the section before it either
    assert SHOTS_PROSE not in s["fail_modes"]
    assert "full IDDSI test film (fork pressure + ruler read)" not in s["fail_modes"]
    assert s["fail_modes"].startswith("(1) Piece squashes flat")
    # the other four sections are untouched by the change
    assert s["prep"].startswith("Bought form")
    assert s["iddsi_test"].startswith("5.1 Fork pressure")
    assert s["visual_qc"].startswith("(1) Distinct battered pieces")


def test_the_swallowed_shape_matches_neither_legacy_pattern():
    """Documents WHY it was silent: neither pattern the parser used to have fires."""
    line = "SHOTS — 3 stills + 2 clips: x.jpg"
    assert not bcp.SECTION_NUM.match(line)
    assert not bcp.SECTION_BARE.match(line)
    assert bcp.SECTION_INLINE.match(line)


def test_unrecognised_header_is_reported_not_silently_swallowed(card_dir):
    p = write_card(card_dir, "bf99_test_dish_L7.md", UNPLACEABLE_HEADER)
    s = bcp.parse_card(p)

    assert len(s["orphan_header_lines"]) == 1
    o = s["orphan_header_lines"][0]
    assert o["text"].startswith("4) SHOTS")
    assert o["line"] == 4
    # not parsed -- nothing may guess where a section starts -- but never silent
    assert s["shots"] == ""


def test_main_warns_on_stderr_for_an_unplaceable_header(card_dir, tmp_path,
                                                        monkeypatch, capsys):
    write_card(card_dir, "bf99_test_dish_L7.md", UNPLACEABLE_HEADER)
    out = tmp_path / "capture"
    monkeypatch.setattr(bcp, "OUT", out)

    rc = bcp.main([str(card_dir)])

    assert rc == 0, "regenerating the protocol must not be blocked"
    err = capsys.readouterr().err
    assert "ORPHAN HEADER" in err
    assert "bf99_test_dish_L7" in err
    proto = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    assert len(proto["orphan_header_lines"]) == 1


def test_main_is_clean_on_the_good_shapes(card_dir, tmp_path, monkeypatch, capsys):
    write_card(card_dir, "bf98_test_dish_L7.md", NUMBERED_INLINE)
    write_card(card_dir, "bf99_test_dish_L7.md", BARE_PROSE_BELOW)
    out = tmp_path / "capture"
    monkeypatch.setattr(bcp, "OUT", out)

    rc = bcp.main([str(card_dir)])

    assert rc == 0
    assert capsys.readouterr().err == ""
    proto = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    assert proto["count"] == 2
    assert proto["orphan_header_lines"] == []
    assert proto["cards_missing_required_sections"] == []
    for s in proto["specimens"]:
        assert len(s["media_expected"]) >= 1


# --- the two shapes that always worked must keep working ---------------------

def test_numbered_header_with_inline_prose(card_dir):
    p = write_card(card_dir, "bf98_test_dish_L7.md", NUMBERED_INLINE)
    s = bcp.parse_card(p)
    assert s["shots"] == ("3 stills: bf99_test_dish_L7_take1.jpg, "
                          "bf99_test_dish_L7_take2.jpg.")
    assert s["media_expected"] == ["bf99_test_dish_L7_take1.jpg",
                                   "bf99_test_dish_L7_take2.jpg"]
    assert s["prep_params"] == {"thickener_g": 8.0, "liquid_ml": 120.0,
                                "blend_seconds": 30}
    assert s["orphan_header_lines"] == []


def test_bare_header_with_prose_below(card_dir):
    p = write_card(card_dir, "bf99_test_dish_L7.md", BARE_PROSE_BELOW)
    s = bcp.parse_card(p)
    assert s["shots"] == "bf99_test_dish_L7_take1.jpg — 45° hero with ruler."
    assert s["media_expected"] == ["bf99_test_dish_L7_take1.jpg"]
    assert s["orphan_header_lines"] == []


# --- corpus-wide gates -------------------------------------------------------

def corpus():
    return bcp.build()


def test_corpus_parses_every_card():
    proto = corpus()
    assert proto["unparsed_cards"] == ["INDEX.md"], (
        "INDEX.md is the corpus index, not a card; anything else is unparsed")
    assert proto["count"] == 267, "update when the corpus grows past 267 cards"
    # int keys from build(), str keys after a JSON round-trip -- accept both
    assert {int(k): v for k, v in proto["levels"].items()} == {
        3: 59, 4: 58, 5: 55, 6: 51, 7: 44}


def test_no_card_has_an_unplaceable_header():
    """The gate that would have failed before the fix."""
    proto = corpus()
    assert proto["orphan_header_lines"] == []


def test_every_card_carries_all_five_sections():
    proto = corpus()
    assert proto["cards_missing_required_sections"] == []
    for s in proto["specimens"]:
        for key in ("prep", "iddsi_test", "visual_qc", "shots", "fail_modes"):
            assert s[key].strip(), f"{s['specimen_id']} has an empty {key}"


def test_bf52_sweet_and_sour_pork_L7_keeps_its_media_list():
    proto = corpus()
    s = [x for x in proto["specimens"]
         if x["specimen_id"] == "bf52_sweet_and_sour_pork_L7"][0]
    assert s["media_expected"] == MEDIA
    assert "3 stills + 2 clips" in s["shots"]
    assert s["orphan_header_lines"] == []


def test_cards_with_no_parsed_media_are_the_one_known_case():
    """A card whose media list is expressed as a range has no filename token to
    find, so media_expected is empty for a legitimate reason. If a second card
    joins this set, its SHOTS section needs the same look bf52 got."""
    proto = corpus()
    none = {s["specimen_id"] for s in proto["specimens"] if not s["media_expected"]}
    assert none == {"bf21_blanched_choy_sum_L5"}
    s = [x for x in proto["specimens"]
         if x["specimen_id"] == "bf21_blanched_choy_sum_L5"][0]
    assert "take1 through take5" in s["shots"]
