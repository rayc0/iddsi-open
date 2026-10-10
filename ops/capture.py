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
            if t == "flow_ml_remaining":
                tests[t] = _ask("ml left in the 10 ml syringe at t=10s (L3 window is 8-10):")
            else:
                tests[t] = _ask(f"{t}: pass or fail", ["pass", "fail", "na"])
        shots = _ask("media files captured (comma-separated, or '-'):") or "-"
        notes = _ask("deviations / notes (or '-'):") or ""

        overall = all(v in ("pass", "na") for v in tests.values())
        rec = {
            "specimen_id": s["specimen_id"],
            "base_food_id": s["base_food_id"],
            "level": s["level"],
            "prep": {**s["prep_params"], "notes": notes},
            "tests": {**tests, "pass": overall},
            "media": [x for x in shots.split(",") if x and x != "-"],
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
