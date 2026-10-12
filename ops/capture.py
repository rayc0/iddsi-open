#!/usr/bin/env python3
"""capture.py — run the IDDSI-Open physical specimen capture and emit
schema-conformant event rows.

Why this exists: the 267 specimen cards are complete prose, which is exactly why
a long run drifts. At specimen 200 nobody re-reads section 2, and a mis-recorded
fork_pressure silently becomes a wrong ground-truth label that then trains a model
wrongly. This presents one card at a time, records the applicable tests as
structured fields, and writes the event row directly -- no transcription.

    capture.py plan  [--level L4] [--food BF07] [--limit 20] [--est-min-per 2.0]
    capture.py run   [--level L4] [--food BF07] [--limit 20] [--out capture/records]
    capture.py emit  [--out capture/records] [--manifest capture/specimens.jsonl]
    capture.py status

Records are append-only JSONL under capture/records/, one file per specimen, so a
run survives interruption and `run` resumes where it stopped.
"""
import argparse, json, pathlib, re, sys, time
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL = REPO / "capture/protocol.json"


def load():
    if not PROTOCOL.is_file():
        sys.exit(f"missing {PROTOCOL} — run: python3 ops/build_capture_protocol.py")
    return json.loads(PROTOCOL.read_text())


def spec_dir(args):
    return REPO / (args.out or "capture/records")


def done_ids(d):
    out = set()
    if d.is_dir():
        for f in d.glob("*.json"):
            out.add(f.stem)
    return out


def select(proto, args, for_run=False):
    sp = proto["specimens"]
    if args.level:
        sp = [s for s in sp if s["level"] == args.level]
    if args.food:
        sp = [s for s in sp if s["base_food_id"] == args.food.upper()]
    if for_run:
        d = spec_dir(args)
        done = done_ids(d)
        sp = [s for s in sp if s["specimen_id"] not in done]
    if args.limit:
        sp = sp[: args.limit]
    return sp


def cmd_plan(args):
    proto = load()
    sp = select(proto, args)
    mins = args.est_min_per * len(sp)
    print(f"specimens in scope : {len(sp)}  of {proto['count']} total")
    if sp:
        lv = {}
        for s in sp:
            lv[s["level_tag"]] = lv.get(s["level_tag"], 0) + 1
        print(f"by level          : {dict(sorted(lv.items()))}")
        est_h, est_m = divmod(int(mins), 60)
        print(f"estimated time    : ~{est_h}h {est_m:02d}m at {args.est_min_per} min each")
    remaining = select(proto, args, for_run=True)
    # Count actual record files, not total-minus-remaining: when a level filter is
    # applied those differ wildly (267-44=223 reported "already recorded" for a
    # directory that does not exist).
    already = len(done_ids(spec_dir(args)) & {s["specimen_id"] for s in sp + remaining})
    print(f"already recorded  : {already}  ·  to go: {len(remaining)}")
    print("\nFIRST 10:")
    for s in sp[:10]:
        print(f"  {s['specimen_id']:<44} {s['level_tag']}  tests={','.join(s['applicable_tests'])}")
    if len(sp) > 10:
        print(f"  ... and {len(sp)-10} more")
    return 0


# ---------------------------------------------------------------------------
# The L3 deciding test is a measurement, not a pass/fail judgement
# ---------------------------------------------------------------------------

#: The only protocol tests recorded as a number rather than a pass/fail word.
#: `flow_ml_remaining` appears in exactly one level's ``applicable_tests`` (L3,
#: 59 of the 267 specimens), so a defect in its accounting has a bounded but
#: total blast radius — see H-CAPTUREL3 in hf/HAZARD_LOG.md.
FLOW_TESTS = ("flow_ml_remaining",)

#: The L3 pass window for the 10 ml / 10 s syringe flow test, in ml REMAINING at
#: t=10 s (i.e. 0-2 ml discharged). Inclusive at both ends. Mirrored from the card
#: prose that ``ops/build_capture_protocol.py`` parsed into
#: ``capture/protocol.json``; verified identical across all 59 L3 cards, either as
#: "Pass = 8-10 ml remaining in the syringe" (12 cards) or through the fail modes
#: "residual <8.0 ml" = too thin and "10.0 ml remains with no measurable discharge
#: at 10 s" = too thick (the rest).
#:
#: One case a number alone cannot express: exactly 10.0 ml remaining with NO
#: measurable discharge is "too thick" on the card's own fail-mode text, even
#: though 10.0 sits inside the window. That judgement belongs in the notes field;
#: the window is the stated rule and is what is implemented here.
FLOW_WINDOW_ML = (8.0, 10.0)

#: Answers meaning "this applicable test was not performed" — never "it passed".
#: An empty answer is NOT here: a stray Enter on the deciding test is re-asked,
#: not read as a skip.
FLOW_SKIP_TOKENS = {"-", "na", "n/a", "none", "skip", "not done", "not performed"}


def parse_flow_ml(raw):
    """Parse a free-text syringe reading into a float number of ml.

    Returns ``None`` when the operator explicitly skips the test. Raises
    :class:`ValueError` for anything else, so an unparseable answer is re-asked
    instead of stored: this number is the L3 ground-truth decision and must never
    be guessed at by a parser.
    """
    text = str(raw).strip().lower()
    for unit in ("millilitres", "milliliters", "millilitre", "milliliter", "ml", "cc"):
        text = text.replace(unit, " ")
    text = text.strip().lstrip("≈~=:").strip()
    if text in FLOW_SKIP_TOKENS:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{raw!r} is not a reading in ml") from None


def test_outcome(name, value):
    """Normalise one recorded test value into ``'pass' | 'fail' | 'na'``.

    Pass/fail tests are stored as the constrained word the operator chose, so the
    value is already the verdict. The numeric flow test is stored as the parsed
    reading (or ``None`` when skipped), so its verdict has to be derived from the
    L3 window — comparing the raw answer against the string ``"pass"`` is exactly
    what recorded every L3 specimen as FAIL.

    A skipped flow test is a fail, not a pass: it is an applicable test that was
    not performed, and a specimen whose deciding test was not run cannot be
    certified at its level.
    """
    if name in FLOW_TESTS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "fail"
        lo, hi = FLOW_WINDOW_ML
        return "pass" if lo <= float(value) <= hi else "fail"
    return value


def _ask(prompt, choices=None):
    if choices:
        print(f"   {prompt}  [{'/'.join(choices)}]", end=" ")
        a = input().strip().lower()
        while a not in choices:
            print(f"   one of {'/'.join(choices)}: ", end="")
            a = input().strip().lower()
        return a
    print(f"   {prompt}", end=" ")
    return input().strip()


def _ask_flow_ml():
    """Ask for the L3 syringe reading, re-asking until it parses.

    Free text on purpose — the operator reads a syringe — but the value stored is
    the parsed number, and an answer that cannot be parsed is re-asked rather than
    silently becoming a fail.
    """
    lo, hi = FLOW_WINDOW_ML
    while True:
        raw = _ask(f"ml left in the 10 ml syringe at t=10s "
                   f"(L3 window is {lo:g}-{hi:g} ml; 'skip' = not performed):")
        try:
            return parse_flow_ml(raw)
        except ValueError as exc:
            print(f"   {exc} — enter a number of ml, or 'skip'.")


def cmd_run(args):
    proto = load()
    sp = select(proto, args, for_run=True)
    if not sp:
        print("nothing to do — every specimen in scope is already recorded")
        return 0
    d = spec_dir(args)
    d.mkdir(parents=True, exist_ok=True)
    print(f"capturing {len(sp)} specimens into {d}")
    print("At each specimen: follow PREP, run ONLY the applicable tests, record pass/fail.")
    print("A failed specimen is a valid result. Never rename a fail to another level.\n")

    for i, s in enumerate(sp, 1):
        print("=" * 78)
        print(f"[{i}/{len(sp)}]  {s['specimen_id']}   base_food={s['base_food_id']}  level={s['level_tag']}")
        print(f"  card: {s['card_path']}")
        print("-" * 78)
        print(f"PREP:\n  {s['prep']}\n")
        print(f"IDDSI TEST (applicable to {s['level_tag']}):\n  {s['iddsi_test']}\n")
        print(f"VISUAL QC:\n  {s['visual_qc']}\n")

        tests = {}
        for t in s["applicable_tests"]:
            if t in FLOW_TESTS:
                tests[t] = _ask_flow_ml()
            else:
                tests[t] = _ask(f"{t}: pass or fail", ["pass", "fail", "na"])
        # Present the card's own media checklist. This list is parsed out of the
        # SHOTS section by ops/build_capture_protocol.py and used to be read by
        # nobody, so a SHOTS header the parser could not place cost the operator
        # the checklist twice: once in the build, once here.
        expected = s.get("media_expected") or []
        if expected:
            print(f"SHOTS — this card requires {len(expected)} media file(s):")
            for f in expected:
                print(f"    {f}")
        else:
            print("SHOTS — no media filename parsed from this card's SHOTS "
                  "section; record what you captured.")
        shots = _ask("media files captured (comma-separated, or '-'):") or "-"
        captured = [x.strip() for x in shots.split(",") if x and x.strip() != "-"]
        missing = [f for f in expected if f not in captured]
        if missing:
            print(f"  WARNING: {len(missing)} of {len(expected)} required media "
                  f"file(s) not listed: {', '.join(missing)}")
        notes = _ask("deviations / notes (or '-'):") or ""

        # Judge each applicable test on its own kind of answer, then require every
        # one to be a pass (or an explicit 'na' where the test does not apply).
        # Before the H-CAPTUREL3 fix this line compared the raw answers against
        # ("pass", "na"), so the numeric L3 reading could never be in the set.
        outcomes = {t: test_outcome(t, v) for t, v in tests.items()}
        overall = all(v in ("pass", "na") for v in outcomes.values())
        rec = {
            "specimen_id": s["specimen_id"],
            "base_food_id": s["base_food_id"],
            "level": s["level"],
            "prep": {**s["prep_params"], "notes": notes},
            "tests": {**tests, "pass": overall},
            "test_outcomes": outcomes,
            "media": captured,
            "media_expected": expected,
            "media_missing": missing,
            "operator": "single-operator",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        (d / f"{s['specimen_id']}.json").write_text(json.dumps(rec, indent=2))
        print(f"  -> recorded: {'PASS' if overall else 'FAIL'}")
    print(f"\ndone. {len(done_ids(d))} specimens on record. Next: python3 ops/capture.py emit")
    return 0


def cmd_emit(args):
    proto = load()
    d = spec_dir(args)
    if not d.is_dir():
        sys.exit(f"no records directory at {d} — run `capture.py run` first")
    recs = []
    for i, f in enumerate(sorted(d.glob("*.json")), start=1):
        r = json.loads(f.read_text())
        media = r.get("media", [])
        recs.append({
            "event_id": f"EV-{i:04d}",
            "base_food_id": r["base_food_id"],
            "level": r["level"],
            "prep": r.get("prep", {}),
            "tests": r.get("tests", {}),
            "source": "real-tested",
            "operator": r.get("operator", "single-operator"),
            "specimen_id": r["specimen_id"],
            "media_count": len(media),
            "media": media,
            "recorded_at": r.get("recorded_at"),
        })
    dest = pathlib.Path(args.manifest or (REPO / "capture/specimens.jsonl"))
    with open(dest, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    passed = sum(1 for r in recs if r["tests"].get("pass"))
    lv = {}
    for r in recs:
        lv[r["level"]] = lv.get(r["level"], 0) + 1
    print(f"emitted {len(recs)} event rows -> {dest}")
    print(f"  passed {passed} · failed {len(recs)-passed}")
    print(f"  by level: {dict(sorted(lv.items()))}")
    # How many L3 rows are backed by an actual syringe reading? A level-3 label
    # whose deciding test was skipped is recorded as a fail, and this says so in
    # the same breath as the counts instead of leaving it to be inferred.
    readings = [r["tests"].get("flow_ml_remaining") for r in recs
                if "flow_ml_remaining" in r["tests"]]
    measured = [v for v in readings
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
    skipped = len(readings) - len(measured)
    if readings:
        print(f"  L3 syringe readings: {len(measured)} measured"
              + (f" · {skipped} skipped (recorded fail — the deciding test was "
                 f"not performed)" if skipped else ""))
    # Records written before the H-CAPTUREL3 fix stored the reading as free text
    # and force-failed every L3 specimen. Say so loudly rather than emitting them
    # as if they were valid: a stored verdict is ground truth and is never
    # rewritten silently.
    legacy = [r["specimen_id"] for r in recs
              if isinstance(r["tests"].get("flow_ml_remaining"), str)]
    if legacy:
        print(f"\n  WARNING: {len(legacy)} record(s) store flow_ml_remaining as text "
              f"— written by the pre-fix accounting, whose L3 verdicts are void: "
              f"{', '.join(legacy[:5])}"
              + (" ..." if len(legacy) > 5 else ""))
    if not passed:
        print("\n  WARNING: zero passed specimens — this manifest cannot train anything.")
    return 0


def cmd_status(args):
    proto = load()
    d = spec_dir(args)
    done = done_ids(d)
    print(f"records       : {len(done)} / {proto['count']}")
    if d.is_dir():
        pas = fail = 0
        for f in d.glob("*.json"):
            if json.loads(f.read_text()).get("tests", {}).get("pass"):
                pas += 1
            else:
                fail += 1
        print(f"passed        : {pas}")
        print(f"failed        : {fail}")
    else:
        print("records dir  : not created yet")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "emit", "status"):
        p = sub.add_parser(name)
        p.add_argument("--level", type=int, choices=[3, 4, 5, 6, 7])
        p.add_argument("--food")
        p.add_argument("--limit", type=int)
        p.add_argument("--out")
        p.add_argument("--est-min-per", type=float, default=2.0)
        p.add_argument("--manifest")
    args = ap.parse_args()
    return {"plan": cmd_plan, "run": cmd_run, "emit": cmd_emit,
            "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
