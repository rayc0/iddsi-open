#!/usr/bin/env python3
"""Tests for the ops/capture.py pass/fail accounting — defect qa03_capture.

Run from repo root:
    python -m pytest ops/tests/test_capture_flow.py -q

No GPU, no network: the scripted operator drives the real ``cmd_run`` against the
real ``capture/protocol.json`` and the records are read back off disk.

The defect under test (HIGH severity): ``cmd_run`` asked the L3 deciding test
``flow_ml_remaining`` as free text but computed

    overall = all(v in ("pass", "na") for v in tests.values())

so a valid reading such as "9" was never in that set and all 59 L3 specimens
(22.1% of the 267-specimen corpus) were recorded as FAIL. The window is 8-10 ml
remaining at t=10 s, so 9 ml is a pass and 7 ml is a fail.

The L4-L7 controls are not decoration: ``flow_ml_remaining`` is the only numeric
test in the protocol, so any change to the shared accounting must leave the
pass/fail-word levels byte-for-byte identical in behaviour.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import importlib.util
import io
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
CAPTURE_PY = REPO / "ops" / "capture.py"
PROTOCOL_JSON = REPO / "capture" / "protocol.json"


def _load_capture():
    """Import ops/capture.py by path.

    Deliberately not ``sys.path`` + ``import capture``: the repo root holds a
    ``capture/`` directory, which Python 3 would bind as a namespace package and
    shadow the module under test.
    """
    spec = importlib.util.spec_from_file_location("capture_under_test", CAPTURE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cap = _load_capture()
PROTOCOL = json.loads(PROTOCOL_JSON.read_text())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _scripted(answers):
    """A stand-in for stdin that pops scripted answers, one per ``input()`` call."""
    remaining = list(answers)

    def fake_input(prompt=""):
        if not remaining:
            raise AssertionError(f"ran out of scripted answers at {prompt!r}")
        return remaining.pop(0)

    return fake_input


def run_capture(level, answers, out_dir, limit=None):
    """Drive ``cmd_run`` for one level with a scripted operator.

    Returns ``(records, transcript)``. ``cmd_run`` prints its prompts with
    ``print(...)`` and then calls a bare ``input()``, so the prompts are read back
    from the captured transcript rather than from ``input``'s argument.
    """
    args = argparse.Namespace(level=level, food=None, limit=limit, out=str(out_dir),
                              est_min_per=2.0, manifest=None)
    fake = _scripted(answers)
    real_input, real_stdout = builtins.input, sys.stdout
    sink = io.StringIO()
    builtins.input = fake
    sys.stdout = sink
    try:
        cap.cmd_run(args)
    finally:
        sys.stdout = real_stdout
        builtins.input = real_input
    records = {f.stem: json.loads(f.read_text())
               for f in sorted(pathlib.Path(out_dir).glob("*.json"))}
    return records, sink.getvalue()


def answers_for(specimens, flow=None, verdict="pass", overrides=None, flow_retry=None,
                flow_for=None):
    """The exact answer sequence ``cmd_run`` consumes, in the order it asks.

    ``flow_for(specimen, index)`` overrides the uniform ``flow`` per specimen, so
    one run can record a mixed batch.
    """
    out = []
    for i, s in enumerate(specimens):
        for t in s["applicable_tests"]:
            if t in cap.FLOW_TESTS:
                reading = flow_for(s, i) if flow_for else flow
                out.append(reading if reading is not None else "skip")
                if flow_retry is not None:
                    out.append(flow_retry)
            else:
                out.append((overrides or {}).get(t, verdict))
        out += ["-", "-"]  # media files, notes
    return out


def specimens_at(level):
    return [s for s in PROTOCOL["specimens"] if s["level"] == level]


# ---------------------------------------------------------------------------
# the protocol shape the fix depends on
# ---------------------------------------------------------------------------

def test_flow_ml_remaining_is_asked_only_at_l3():
    """Measured from the protocol: the numeric test's blast radius is L3 alone."""
    asking = sorted({s["level"] for s in PROTOCOL["specimens"]
                     if "flow_ml_remaining" in s["applicable_tests"]})
    assert asking == [3]
    l3 = specimens_at(3)
    assert len(l3) == 59
    assert all("flow_ml_remaining" in s["applicable_tests"] for s in l3)
    # and it is the deciding test at L3, not one of several equal ones
    assert any("deciding test at L3" in s["iddsi_test"] for s in l3)


def test_no_other_test_is_treated_as_numeric():
    """Every non-flow test must stay a pass/fail/na word — nothing reclassified."""
    names = {t for s in PROTOCOL["specimens"] for t in s["applicable_tests"]}
    assert cap.FLOW_TESTS == ("flow_ml_remaining",)
    for name in sorted(names - set(cap.FLOW_TESTS)):
        assert cap.test_outcome(name, "pass") == "pass"
        assert cap.test_outcome(name, "fail") == "fail"
        assert cap.test_outcome(name, "na") == "na"


# ---------------------------------------------------------------------------
# the window itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reading,expected", [
    (9, "pass"), (9.0, "pass"), (8, "pass"), (10, "pass"),
    (8.4, "pass"), (9.6, "pass"),
    (7.9, "fail"), (7, "fail"), (0, "fail"), (10.1, "fail"), (11, "fail"),
    (None, "fail"),          # the deciding test was not performed
    ("9", "fail"),           # a raw string is not a reading
])
def test_flow_outcome_follows_the_l3_window(reading, expected):
    assert cap.test_outcome("flow_ml_remaining", reading) == expected


@pytest.mark.parametrize("raw,parsed", [
    ("9", 9.0), ("9.0", 9.0), (" 9 ", 9.0), ("9 ml", 9.0), ("9ml", 9.0),
    ("~9", 9.0), ("8", 8.0), ("10", 10.0), ("7.5", 7.5), ("10.5", 10.5),
])
def test_parse_flow_ml_accepts_ordinary_readings(raw, parsed):
    assert cap.parse_flow_ml(raw) == pytest.approx(parsed)


@pytest.mark.parametrize("raw", ["skip", "na", "n/a", "-", "not performed", "SKIP",
                                 " na "])
def test_parse_flow_ml_skip_returns_none(raw):
    """An explicit skip is recorded as None, which is a fail — never a pass."""
    assert cap.parse_flow_ml(raw) is None


@pytest.mark.parametrize("raw", ["", "not a number", "ml", "eight", "9..", "9 9",
                                 "two ml out", "0-2 ml flowed out"])
def test_parse_flow_ml_rejects_nonsense(raw):
    """Anything that is neither a number nor an explicit skip is re-asked."""
    with pytest.raises(ValueError):
        cap.parse_flow_ml(raw)


def test_window_is_inclusive_and_stated_in_the_card():
    """The window must match the card text, not drift from it."""
    lo, hi = cap.FLOW_WINDOW_ML
    assert (lo, hi) == (8.0, 10.0)
    l3_text = " ".join(s["iddsi_test"] + " " + s["fail_modes"] for s in specimens_at(3))
    # every L3 card states one side of the window or the other
    assert "8" in l3_text and "10" in l3_text


# ---------------------------------------------------------------------------
# the regression: a valid L3 reading is recorded as a pass
# ---------------------------------------------------------------------------

def test_l3_reading_of_9ml_is_a_pass(tmp_path):
    records, transcript = run_capture(3, answers_for(specimens_at(3), flow="9"),
                                      tmp_path / "l3")
    assert len(records) == 59
    assert all(r["tests"]["pass"] is True for r in records.values())
    sample = next(iter(records.values()))
    assert sample["tests"]["flow_ml_remaining"] == 9.0
    assert sample["test_outcomes"]["flow_ml_remaining"] == "pass"
    # the operator was asked for a reading, not for pass/fail
    assert "ml left in the 10 ml syringe at t=10s" in transcript
    assert "L3 window is 8-10 ml" in transcript
    assert transcript.count("ml left in the 10 ml syringe") == 59


@pytest.mark.parametrize("reading", ["7", "11", "7.9", "10.1"])
def test_l3_reading_outside_the_window_is_a_fail(tmp_path, reading):
    records, _ = run_capture(3, answers_for(specimens_at(3), flow=reading),
                             tmp_path / f"l3_{reading}")
    assert len(records) == 59
    assert all(r["tests"]["pass"] is False for r in records.values())
    assert all(r["test_outcomes"]["flow_ml_remaining"] == "fail"
               for r in records.values())


def test_l3_window_boundaries_recorded_as_pass(tmp_path):
    for reading in ("8", "10"):
        records, _ = run_capture(3, answers_for(specimens_at(3), flow=reading),
                                 tmp_path / f"l3_bound_{reading}")
        assert all(r["tests"]["pass"] is True for r in records.values()), reading


def test_every_l3_specimen_can_pass(tmp_path):
    """The headline regression: no L3 specimen is force-failed by the accounting.

    Before the fix this was 0 of 59 with a reading of 9 ml.
    """
    records, _ = run_capture(3, answers_for(specimens_at(3), flow="9"), tmp_path / "all")
    passed = sum(1 for r in records.values() if r["tests"]["pass"])
    assert passed == len(records) == 59


def test_l3_still_requires_its_other_tests(tmp_path):
    """A good syringe reading must not rescue a failed fork drip (and vice versa)."""
    records, _ = run_capture(3, answers_for(specimens_at(3), flow="9",
                                            overrides={"fork_drip": "fail"}),
                             tmp_path / "l3_fork_drip_fail")
    assert len(records) == 59
    assert all(r["tests"]["pass"] is False for r in records.values())
    assert all(r["test_outcomes"]["fork_drip"] == "fail" for r in records.values())


def test_l3_na_on_an_inapplicable_test_still_passes(tmp_path):
    records, _ = run_capture(3, answers_for(specimens_at(3), flow="9",
                                            overrides={"spoon_tilt": "na"}),
                             tmp_path / "l3_na")
    assert all(r["tests"]["pass"] is True for r in records.values())


def test_unparseable_reading_is_reasked_not_recorded(tmp_path):
    records, transcript = run_capture(3, answers_for(specimens_at(3), flow="garbage",
                                                     flow_retry="9"),
                                      tmp_path / "l3_retry")
    assert all(r["tests"]["flow_ml_remaining"] == 9.0 for r in records.values())
    assert all(r["tests"]["pass"] is True for r in records.values())
    # once per specimen for the first ask, once more for the re-ask
    assert transcript.count("ml left in the 10 ml syringe") == 2 * 59
    assert transcript.count("is not a reading in ml") == 59


def test_skipped_flow_test_is_a_fail_not_a_pass(tmp_path):
    """A skipped deciding test is fail-closed: it cannot certify the level."""
    records, _ = run_capture(3, answers_for(specimens_at(3), flow=None),
                             tmp_path / "l3_skip")
    assert all(r["tests"]["flow_ml_remaining"] is None for r in records.values())
    assert all(r["tests"]["pass"] is False for r in records.values())
    assert all(r["test_outcomes"]["flow_ml_remaining"] == "fail"
               for r in records.values())


# ---------------------------------------------------------------------------
# controls: the other levels must be untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", [4, 5, 6, 7])
def test_other_levels_all_pass_control(tmp_path, level):
    records, _ = run_capture(level, answers_for(specimens_at(level)),
                             tmp_path / f"l{level}_pass")
    assert len(records) == len(specimens_at(level))
    assert all(r["tests"]["pass"] is True for r in records.values())
    # no numeric test is invented at these levels
    assert all("flow_ml_remaining" not in r["tests"] for r in records.values())


@pytest.mark.parametrize("level", [4, 5, 6, 7])
def test_other_levels_all_fail_control(tmp_path, level):
    records, _ = run_capture(level, answers_for(specimens_at(level), verdict="fail"),
                             tmp_path / f"l{level}_fail")
    assert all(r["tests"]["pass"] is False for r in records.values())


@pytest.mark.parametrize("level", [4, 5, 6, 7])
def test_other_levels_single_failing_test_still_fails(tmp_path, level):
    """One failing applicable test fails the specimen — the conjunction holds."""
    overrides = {specimens_at(level)[0]["applicable_tests"][0]: "fail"}
    records, _ = run_capture(level, answers_for(specimens_at(level), overrides=overrides),
                             tmp_path / f"l{level}_one_fail")
    assert all(r["tests"]["pass"] is False for r in records.values())
    failing = next(iter(overrides))
    assert all(r["test_outcomes"][failing] == "fail" for r in records.values())


# ---------------------------------------------------------------------------
# emit / status read the same accounting back
# ---------------------------------------------------------------------------

def test_emit_counts_and_reports_the_syringe_readings(tmp_path, capsys):
    records_dir = tmp_path / "records"
    run_capture(3, answers_for(specimens_at(3), flow="9"), records_dir)
    run_capture(4, answers_for(specimens_at(4)[:5], overrides={"fork_pressure": "fail"}),
                records_dir, limit=5)
    manifest = tmp_path / "specimens.jsonl"
    rc = cap.cmd_emit(argparse.Namespace(out=str(records_dir), manifest=str(manifest),
                                         level=None, food=None, limit=None,
                                         est_min_per=2.0))
    assert rc == 0
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    assert len(rows) == 59 + 5
    assert sum(1 for r in rows if r["tests"]["pass"]) == 59
    out = capsys.readouterr().out
    assert "passed 59 · failed 5" in out
    assert "L3 syringe readings: 59 measured" in out


def test_emit_warns_loudly_about_pre_fix_records(tmp_path, capsys):
    """A record storing the reading as text was written by the buggy accounting.

    Its L3 verdict is void, and `emit` must say so instead of passing it on as a
    valid ground-truth row.
    """
    records_dir = tmp_path / "records"
    run_capture(3, answers_for(specimens_at(3), flow="9"), records_dir)
    # forge one pre-fix record: free-text reading, verdict forced to fail
    legacy = json.loads((records_dir / "bf01_steamed_white_rice_L3.json").read_text())
    legacy["specimen_id"] = "legacy_bf01_steamed_white_rice_L3"
    legacy["tests"] = {"spoon_tilt": "pass", "fork_drip": "pass",
                       "flow_ml_remaining": "9", "pass": False}
    (records_dir / "bf01_steamed_white_rice_L3.json").unlink()
    (records_dir / "legacy_bf01_steamed_white_rice_L3.json").write_text(
        json.dumps(legacy, indent=2))
    manifest = tmp_path / "specimens.jsonl"
    rc = cap.cmd_emit(argparse.Namespace(out=str(records_dir), manifest=str(manifest),
                                         level=None, food=None, limit=None,
                                         est_min_per=2.0))
    assert rc == 0
    out = capsys.readouterr().out
    assert "store flow_ml_remaining as text" in out
    assert "legacy_bf01_steamed_white_rice_L3" in out
    # the forged row still counts as a fail, it is not quietly promoted
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    assert sum(1 for r in rows if r["tests"]["pass"]) == 58


def test_status_reports_the_same_pass_count(tmp_path, capsys):
    # One mixed batch: 56 readings inside the window, 3 outside it. Records are
    # append-only and `run` resumes, so a second run into the same directory
    # would be a no-op and could not produce a mixed count.
    records_dir = tmp_path / "records"
    run_capture(3, answers_for(specimens_at(3),
                               flow_for=lambda s, i: "7" if i < 3 else "9"),
                records_dir)
    rc = cap.cmd_status(argparse.Namespace(out=str(records_dir), level=None, food=None,
                                           limit=None, manifest=None, est_min_per=2.0))
    assert rc == 0
    out = capsys.readouterr().out
    assert "passed        : 56" in out
    assert "failed        : 3" in out
