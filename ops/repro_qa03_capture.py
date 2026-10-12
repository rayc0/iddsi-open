#!/usr/bin/env python3
"""repro_qa03_capture.py — reproduce (and then re-check) the qa03_capture defect.

The defect (HIGH severity, found by lane qa03_capture): `ops/capture.py` asks the
L3 deciding test `flow_ml_remaining` as free text ("ml left in the 10 ml syringe
at t=10s"), but the pass/fail accounting was

    overall = all(v in ("pass", "na") for v in tests.values())

so a valid numeric reading such as "9" is never in that set and EVERY L3
specimen was recorded as FAIL, whatever the operator typed. L3 is the only level
whose applicable tests include `flow_ml_remaining` (measured below), so the blast
radius is exactly the L3 block of the 267-specimen corpus: 59 of 267 specimens,
22.1% of the corpus, all mis-recorded in the conservative direction.

This script does not re-implement capture.py. It drives the real
`ops/capture.py:cmd_run` with a scripted operator, writes into a scratch
directory, and reads the resulting records back. Run it from the repo root:

    python3 ops/repro_qa03_capture.py                      # every scenario
    python3 ops/repro_qa03_capture.py --evidence out.json  # + JSON evidence
    python3 ops/repro_qa03_capture.py --verbose             # full transcript

Each scenario states the verdict the card rule requires and the script exits 1
if any specimen is recorded otherwise, so it doubles as a gate. `--evidence`
records the git HEAD it ran against, so a "before" file and an "after" file can
never be confused.
"""
from __future__ import annotations

import argparse
import builtins
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent

#: The L3 window stated on the card text in capture/protocol.json: "Pass = 8-10 ml
#: remaining in the syringe (i.e., 0-2 ml flowed out)". Mirrored here only so this
#: script can judge a reading; capture.py owns the rule.
L3_WINDOW = (8.0, 10.0)

#: name, level, scripted flow answer, per-test verdict overrides, expected verdict.
#: "pass" = every record in scope must be recorded pass; "fail" = every record must
#: be recorded fail. The L4-L7 controls exist so a future change to the shared
#: accounting cannot silently weaken another level.
SCENARIOS = [
    # --- L3: the deciding test is numeric -------------------------------------
    dict(name="l3_reading_9ml", level=3, flow="9", expect="pass"),
    dict(name="l3_reading_8ml_lower_bound", level=3, flow="8", expect="pass"),
    dict(name="l3_reading_10ml_upper_bound", level=3, flow="10", expect="pass"),
    dict(name="l3_reading_9ml_with_unit", level=3, flow="9 ml", expect="pass"),
    dict(name="l3_reading_7ml", level=3, flow="7", expect="fail"),
    dict(name="l3_reading_7p9ml", level=3, flow="7.9", expect="fail"),
    dict(name="l3_reading_10p1ml", level=3, flow="10.1", expect="fail"),
    dict(name="l3_reading_11ml", level=3, flow="11", expect="fail"),
    # a bad reading is re-asked, not silently recorded as a fail
    dict(name="l3_reading_reprompt_then_9ml", level=3, flow="not a number", flow_retry="9",
         expect="pass"),
    # the L3 conjunction must still require the OTHER L3 tests
    dict(name="l3_good_flow_but_fork_drip_fails", level=3, flow="9",
         overrides={"fork_drip": "fail"}, expect="fail"),
    dict(name="l3_good_flow_but_spoon_tilt_na", level=3, flow="9",
         overrides={"spoon_tilt": "na"}, expect="pass"),
    # --- controls: the other levels must be untouched -------------------------
    dict(name="l4_all_pass_control", level=4, expect="pass"),
    dict(name="l5_all_pass_control", level=5, expect="pass"),
    dict(name="l6_all_pass_control", level=6, expect="pass"),
    dict(name="l7_all_pass_control", level=7, expect="pass"),
    dict(name="l4_all_fail_control", level=4, verdict="fail", expect="fail"),
    dict(name="l4_fork_pressure_only_fail", level=4, overrides={"fork_pressure": "fail"},
         expect="fail"),
    dict(name="l7_fork_pressure_only_fail", level=7, overrides={"fork_pressure": "fail"},
         expect="fail"),
]


def load_capture_module(repo):
    """Import ops/capture.py as a module without requiring ops/ to be a package."""
    path = pathlib.Path(repo) / "ops" / "capture.py"
    spec = importlib.util.spec_from_file_location("iddsi_capture_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scripted_operator(answers):
    """A stand-in for stdin: pops scripted answers, records every prompt seen."""
    seen = []
    remaining = list(answers)

    def fake_input(prompt=""):
        seen.append(str(prompt))
        if not remaining:
            raise AssertionError(
                f"scripted operator ran out of answers at prompt {prompt!r} "
                f"after {len(seen)} prompts"
            )
        return remaining.pop(0)

    return fake_input, seen


def answer_script(specimens, flow, verdict, overrides, flow_retry):
    """The exact answer sequence cmd_run consumes for one specimen, in order."""
    out = []
    for s in specimens:
        for t in s["applicable_tests"]:
            if t == "flow_ml_remaining":
                out.append(flow if flow is not None else "na")
                if flow_retry is not None:
                    out.append(flow_retry)
            else:
                out.append((overrides or {}).get(t, verdict))
        out.append("-")   # media files captured
        out.append("-")   # deviations / notes
    return out


def run_scenario(cap, proto, sc, workdir, verbose=False):
    """Drive cmd_run for one scenario with a fully scripted operator."""
    specimens = [s for s in proto["specimens"] if s["level"] == sc["level"]]
    args = argparse.Namespace(
        level=sc["level"], food=None, limit=None, out=str(workdir), est_min_per=2.0,
        manifest=None,
    )
    answers = answer_script(specimens, sc.get("flow"), sc.get("verdict", "pass"),
                            sc.get("overrides"), sc.get("flow_retry"))
    fake, _seen = scripted_operator(answers)
    real_input = builtins.input
    builtins.input = fake
    # cmd_run prints the whole card per specimen; keep the report readable unless
    # the full transcript is wanted.
    sink = None if verbose else io.StringIO()
    real_stdout = sys.stdout
    if sink is not None:
        sys.stdout = sink
    try:
        cap.cmd_run(args)
    finally:
        sys.stdout = real_stdout
        builtins.input = real_input
    records = {f.stem: json.loads(f.read_text())
               for f in sorted(pathlib.Path(workdir).glob("*.json"))}
    return specimens, records


def summarise(specimens, records):
    passed = [k for k, r in records.items() if r["tests"].get("pass")]
    failed = [k for k, r in records.items() if not r["tests"].get("pass")]
    return {
        "specimens_in_scope": len(specimens),
        "records_written": len(records),
        "recorded_pass": len(passed),
        "recorded_fail": len(failed),
        "flow_values_recorded": sorted(
            {json.dumps(r["tests"].get("flow_ml_remaining")) for r in records.values()}
        ),
    }


def git_head(repo):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence", help="write the measured JSON evidence here")
    ap.add_argument("--workdir", help="scratch dir for records (default: a temp dir)")
    ap.add_argument("--verbose", action="store_true",
                   help="print the full per-specimen capture transcript")
    ap.add_argument("--code-commit",
                    help="commit id to record in the evidence when --repo is a "
                         "copy outside the git history (e.g. the pre-fix tree)")
    ap.add_argument("--repo", default=str(REPO),
                   help="repo holding ops/capture.py + capture/protocol.json "
                        "(default: this checkout). Point it at a copy of the "
                        "pre-fix tree to regenerate the 'before' evidence.")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    cap = load_capture_module(repo)
    proto = json.loads((repo / "capture" / "protocol.json").read_text())
    total = len(proto["specimens"])

    # Which levels actually ask flow_ml_remaining? Measured, not assumed.
    flow_specimens = [s for s in proto["specimens"]
                      if "flow_ml_remaining" in s["applicable_tests"]]
    flow_levels = sorted({s["level"] for s in flow_specimens})
    per_level = {}
    for s in proto["specimens"]:
        tag = s["level_tag"]
        per_level[tag] = per_level.get(tag, 0) + 1

    print("=" * 78)
    print("qa03_capture reproduction — ops/capture.py pass/fail accounting")
    print("=" * 78)
    head = args.code_commit or git_head(repo)
    print(f"code                     : {head}")
    print(f"corpus                   : {total} specimens, by level {per_level}")
    print(f"levels asking flow_ml_remaining : {flow_levels}  ({len(flow_specimens)} specimens"
          f" = {100.0 * len(flow_specimens) / total:.1f}% of the corpus)")
    print(f"L3 window under test     : {L3_WINDOW[0]:g}-{L3_WINDOW[1]:g} ml remaining at t=10s")

    root = pathlib.Path(args.workdir) if args.workdir else pathlib.Path(
        tempfile.mkdtemp(prefix="qa03_capture_"))
    root.mkdir(parents=True, exist_ok=True)

    evidence = {"code_commit": head, "repo": str(repo), "corpus_total": total, "per_level": per_level,
                "levels_asking_flow": flow_levels,
                "flow_specimen_share_pct": round(100.0 * len(flow_specimens) / total, 1),
                "l3_window_ml": list(L3_WINDOW), "runs": {}}

    problems = []
    print()
    for sc in SCENARIOS:
        d = root / sc["name"]
        d.mkdir(parents=True, exist_ok=True)
        specimens, records = run_scenario(cap, proto, sc, d, verbose=args.verbose)
        summary = summarise(specimens, records)
        evidence["runs"][sc["name"]] = {**summary, "expected": sc["expect"],
                                        "scripted_flow": sc.get("flow"),
                                        "overrides": sc.get("overrides")}
        want_pass = summary["records_written"] if sc["expect"] == "pass" else 0
        ok = summary["recorded_pass"] == want_pass
        if not ok:
            problems.append(
                f"{sc['name']}: expected {'all pass' if sc['expect'] == 'pass' else 'all fail'} "
                f"({want_pass} of {summary['records_written']}), got {summary['recorded_pass']} "
                f"pass / {summary['recorded_fail']} fail")
        print(f"[{'ok ' if ok else 'BAD'}] {sc['name']:<38} L{sc['level']} "
              f"flow={sc.get('flow')!r:<14} pass {summary['recorded_pass']:>3}"
              f" / fail {summary['recorded_fail']:>3}  (want {want_pass:>3} pass)")

    evidence["problems"] = problems
    print("\n" + "=" * 78)
    if problems:
        print(f"DEFECT PRESENT — {len(problems)} accounting problem(s):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("NO PROBLEM — every scenario recorded the verdict the card rule requires.")
    print(f"records scratch dir: {root}")

    if args.evidence:
        p = pathlib.Path(args.evidence)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(evidence, indent=2) + "\n")
        print(f"evidence written: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
