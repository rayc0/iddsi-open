#!/usr/bin/env python3
"""night_summary.py — build NIGHT_<date>.md from the IDDSI-Open night-run logs.

Stdlib-only. Reads (all optional — missing files are skipped with a note):
  - night_pilot.log            (armed / heartbeat / window open / pilot finished lines)
  - synth_pilot_night*.log     (generator stdout/stderr: errors, Tracebacks, exit codes)
  - flux_test.log              (FLUX smoke-test output, if present)
  - data/synth_pilot_L4/generation_log.jsonl  (authoritative accept/reject record)

Usage:
  night_summary.py [--logs-dir DIR] [--out PATH] [--date YYYY-MM-DD]

Defaults: --logs-dir ~/iddsi, --out <logs-dir>/reports/NIGHT_<date>.md,
--date = latest date seen in the logs (else today).

The report contains: accept rate (from generation_log.jsonl, cross-checked
against log lines), per-attempt timings when present, error lines, and the
night_pilot lifecycle (armed -> window open -> pilot finished -> done).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import date, datetime

# ---------------------------------------------------------------------------
# Log line patterns (matched against the real formats observed on Spark)
# ---------------------------------------------------------------------------

# night_pilot.log:
#   2026-09-01 07:09:52 armed; waiting for 23:00 local (now 07:09)
#   2026-09-01 07:09:52 heartbeat H=07
#   2026-09-01 23:09:53 window open; pausing deepseek-v4
#   2026-09-01 23:19:01 pilot finished; jpgs on disk: 72; log tail: RuntimeError: ... exit=1
#   2026-09-01 23:19:01 restoring deepseek-v4
#   2026-09-01 23:21:32 done
RE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})\s+(.*)$")
RE_ARMED = re.compile(r"armed; waiting for (\S+)")
RE_HEARTBEAT = re.compile(r"heartbeat H=(\d+)")
RE_WINDOW = re.compile(r"window open")
RE_FINISHED = re.compile(
    r"pilot finished; jpgs on disk:\s*(\d+); log tail:\s*(.*?)\s*exit=(\d+)"
)
RE_DONE = re.compile(r"\bdone\b")
RE_RESTORING = re.compile(r"restoring")

# synth_pilot_night.log / flux_test.log:
RE_ERROR_LINE = re.compile(
    r"(Traceback \(most recent call last\)|^\s*(?:\w+)?Error|RuntimeError|"
    r"Exception|FAILED|failed|Couldn't connect|^exit=\d+)",
    re.IGNORECASE,
)
RE_ACCEPTED_N_OF_M = re.compile(r"[Aa]ccepted (?:only )?(\d+)/(\d+) candidates")
RE_ATTEMPTS = re.compile(r"after (\d+) attempts")
# generic per-image timing shapes, e.g. "image 12 took 34.5s", "12.8s/image",
# "elapsed 12.8s", "took 12.8 sec"
RE_TIMING = re.compile(
    r"(?:took|elapsed|in)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)\b"
    r"|([0-9]+(?:\.[0-9]+)?)\s*s/image"
    r"|([0-9]+(?:\.[0-9]+)?)s/it"
)


def _read_lines(path):
    """Read a log file tolerantly; split on \\r too (tqdm progress bars)."""
    try:
        with open(path, "r", errors="replace") as fh:
            raw = fh.read()
    except OSError:
        return []
    lines = []
    for chunk in raw.splitlines():
        lines.extend(chunk.split("\r"))
    return [ln.strip() for ln in lines if ln.strip()]


def parse_night_pilot(path):
    """Extract the night_pilot lifecycle. Returns dict of facts."""
    facts = {
        "path": path,
        "armed": [],        # list of (ts, wait_for)
        "heartbeats": 0,
        "first_ts": None,
        "last_ts": None,
        "window_open": None,
        "finished": None,   # (ts, jpgs, tail_msg, exit_code)
        "restoring": None,
        "done": None,
        "dates": set(),
    }
    for ln in _read_lines(path):
        m = RE_TS.match(ln)
        if not m:
            continue
        d, t, rest = m.group(1), m.group(2), m.group(3)
        ts = "%s %s" % (d, t)
        facts["dates"].add(d)
        if facts["first_ts"] is None:
            facts["first_ts"] = ts
        facts["last_ts"] = ts
        if (a := RE_ARMED.search(rest)):
            facts["armed"].append((ts, a.group(1)))
        elif RE_HEARTBEAT.search(rest):
            facts["heartbeats"] += 1
        elif RE_WINDOW.search(rest):
            facts["window_open"] = ts
        elif (f := RE_FINISHED.search(rest)):
            facts["finished"] = (ts, int(f.group(1)), f.group(2), int(f.group(3)))
        elif RE_RESTORING.search(rest):
            facts["restoring"] = ts
        elif RE_DONE.search(rest):
            facts["done"] = ts
    return facts


def parse_generation_log(path):
    """Parse generation_log.jsonl. Returns stats dict (tolerant of bad lines)."""
    stats = {
        "path": path,
        "total": 0,
        "accepted": 0,
        "rejected": 0,
        "levels": {},          # level_declared -> [accepted, total]
        "models": {},          # generator_model -> count
        "run_ids": set(),
        "bad_lines": 0,
        "reject_reasons": {},  # normalized reason -> count
    }
    try:
        fh = open(path, "r", errors="replace")
    except OSError:
        return stats
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_lines"] += 1
                continue
            stats["total"] += 1
            acc = bool(rec.get("accepted"))
            if acc:
                stats["accepted"] += 1
            else:
                stats["rejected"] += 1
            lv = rec.get("level_declared")
            if lv is not None:
                slot = stats["levels"].setdefault(str(lv), [0, 0])
                slot[1] += 1
                if acc:
                    slot[0] += 1
            gm = rec.get("generator_model")
            if gm:
                stats["models"][gm] = stats["models"].get(gm, 0) + 1
            if rec.get("run_id"):
                stats["run_ids"].add(rec["run_id"])
            judge = rec.get("judge") or {}
            for reason in judge.get("reasons") or []:
                if not isinstance(reason, str):
                    continue
                # keep the short categorical reasons, skip long prose
                if len(reason) <= 80:
                    stats["reject_reasons"][reason] = (
                        stats["reject_reasons"].get(reason, 0) + 1
                    )
    return stats


def parse_synth_log(path):
    """Extract errors, accept-rate mentions, and timings from a synth log."""
    out = {
        "path": path,
        "errors": [],          # trimmed error-ish lines (deduped, capped)
        "accepted_nm": None,   # (accepted, candidates) from log text
        "attempts": None,
        "exit_code": None,
        "timings": [],         # floats, seconds
    }
    seen = set()
    for ln in _read_lines(path):
        if (m := RE_ACCEPTED_N_OF_M.search(ln)):
            out["accepted_nm"] = (int(m.group(1)), int(m.group(2)))
        if (m := RE_ATTEMPTS.search(ln)):
            out["attempts"] = int(m.group(1))
        if ln.startswith("exit="):
            try:
                out["exit_code"] = int(ln.split("=", 1)[1])
            except ValueError:
                pass
        if RE_ERROR_LINE.search(ln):
            key = ln[:120]
            if key not in seen and len(out["errors"]) < 40:
                seen.add(key)
                out["errors"].append(ln[:300])
        for tm in RE_TIMING.finditer(ln):
            for g in tm.groups():
                if g is not None:
                    try:
                        out["timings"].append(float(g))
                    except ValueError:
                        pass
                    break
    return out


def _fmt_pct(n, d):
    return "n/a" if d == 0 else "%.1f%%" % (100.0 * n / d)


def build_report(logs_dir, report_date, sources):
    """Compose the markdown report. `sources` maps label -> parsed dict/None."""
    lines = []
    lines.append("# NIGHT %s — IDDSI-Open night run summary" % report_date)
    lines.append("")
    lines.append("_Generated %s by ops/night_summary.py from `%s`._"
                 % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), logs_dir))
    lines.append("")

    npf = sources.get("night_pilot")
    gen = sources.get("generation_log")
    synths = sources.get("synth_logs") or []
    flux = sources.get("flux_test")

    # --- Lifecycle ---------------------------------------------------------
    lines.append("## Night pilot lifecycle")
    lines.append("")
    if npf is None:
        lines.append("- `night_pilot.log` not found — skipped.")
    else:
        if npf["armed"]:
            for ts, wait in npf["armed"]:
                lines.append("- **armed** %s (waiting for %s)" % (ts, wait))
        if npf["window_open"]:
            lines.append("- **window open** %s" % npf["window_open"])
        lines.append("- heartbeats: %d" % npf["heartbeats"])
        if npf["finished"]:
            ts, jpgs, tail, code = npf["finished"]
            lines.append("- **pilot finished** %s — jpgs on disk: %d, exit=%d"
                         % (ts, jpgs, code))
            if tail:
                lines.append("  - log tail: `%s`" % tail[:200])
        if npf["restoring"]:
            lines.append("- **restoring** %s" % npf["restoring"])
        if npf["done"]:
            lines.append("- **done** %s" % npf["done"])
        if npf["first_ts"] and npf["last_ts"]:
            lines.append("- log span: %s → %s" % (npf["first_ts"], npf["last_ts"]))
    lines.append("")

    # --- Accept rate --------------------------------------------------------
    lines.append("## Accept rate")
    lines.append("")
    if gen is None or gen["total"] == 0:
        lines.append("- `generation_log.jsonl` missing or empty — no "
                     "authoritative accept rate.")
    else:
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append("| attempts | %d |" % gen["total"])
        lines.append("| accepted | %d |" % gen["accepted"])
        lines.append("| rejected | %d |" % gen["rejected"])
        lines.append("| accept rate | **%s** |"
                     % _fmt_pct(gen["accepted"], gen["total"]))
        if gen["bad_lines"]:
            lines.append("| unparseable jsonl lines | %d |" % gen["bad_lines"])
        lines.append("")
        if gen["levels"]:
            lines.append("| level | accepted | attempts | rate |")
            lines.append("|---|---|---|---|")
            for lv in sorted(gen["levels"]):
                a, t = gen["levels"][lv]
                lines.append("| L%s | %d | %d | %s |"
                             % (lv, a, t, _fmt_pct(a, t)))
            lines.append("")
        if gen["models"]:
            lines.append("- generator model(s): "
                         + ", ".join("`%s` (%d)" % (m, c)
                                     for m, c in sorted(gen["models"].items())))
        lines.append("- run id(s): "
                     + ", ".join("`%s`" % r for r in sorted(gen["run_ids"])))
    # cross-check against log-line claims
    for s in synths:
        if s["accepted_nm"]:
            a, m = s["accepted_nm"]
            lines.append("- log-line cross-check (%s): accepted %d/%d (%s)"
                         % (os.path.basename(s["path"]), a, m, _fmt_pct(a, m)))
    lines.append("")

    # --- Reject reasons -----------------------------------------------------
    if gen and gen["reject_reasons"]:
        lines.append("## Top reject reasons (judge, categorical)")
        lines.append("")
        lines.append("| reason | count |")
        lines.append("|---|---|")
        for reason, cnt in sorted(gen["reject_reasons"].items(),
                                  key=lambda kv: -kv[1])[:10]:
            lines.append("| %s | %d |" % (reason.replace("|", "\\|"), cnt))
        lines.append("")

    # --- Timings ------------------------------------------------------------
    lines.append("## Timings")
    lines.append("")
    any_timing = False
    for s in synths + ([flux] if flux else []):
        if s and s["timings"]:
            any_timing = True
            ts = s["timings"]
            lines.append("- `%s`: %d timing sample(s), mean %.2fs, "
                         "min %.2fs, max %.2fs"
                         % (os.path.basename(s["path"]), len(ts),
                            sum(ts) / len(ts), min(ts), max(ts)))
    if not any_timing:
        lines.append("- No per-image timing lines found in the logs "
                     "(tqdm progress output is not parsed for timings).")
    lines.append("")

    # --- Errors -------------------------------------------------------------
    lines.append("## Errors")
    lines.append("")
    any_err = False
    for s in synths + ([flux] if flux else []):
        if s and (s["errors"] or s["exit_code"] not in (None, 0)):
            any_err = True
            lines.append("### `%s`" % os.path.basename(s["path"]))
            if s["exit_code"] is not None:
                lines.append("- exit code: **%d**" % s["exit_code"])
            for e in s["errors"][:15]:
                lines.append("- `%s`" % e.replace("`", "'"))
            lines.append("")
    if not any_err:
        lines.append("- No error lines found.")
        lines.append("")

    # --- Missing sources note ------------------------------------------------
    missing = [lbl for lbl, v in sources.items()
               if v is None and lbl != "synth_logs"]
    if not synths:
        missing.append("synth_pilot_night*.log")
    if missing:
        lines.append("## Missing inputs (skipped)")
        lines.append("")
        for lbl in missing:
            lines.append("- %s" % lbl)
        lines.append("")

    return "\n".join(lines) + "\n"


def pick_date(sources, override):
    if override:
        return override
    dates = set()
    npf = sources.get("night_pilot")
    if npf:
        dates |= npf["dates"]
    if dates:
        return max(dates)
    return date.today().isoformat()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs-dir", default=os.path.expanduser("~/iddsi"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD report date")
    args = ap.parse_args(argv)

    logs_dir = os.path.expanduser(args.logs_dir)
    sources = {}

    np_path = os.path.join(logs_dir, "night_pilot.log")
    sources["night_pilot"] = (parse_night_pilot(np_path)
                              if os.path.isfile(np_path) else None)

    synth_paths = sorted(glob.glob(os.path.join(logs_dir, "synth_pilot_night*.log")))
    sources["synth_logs"] = [parse_synth_log(p) for p in synth_paths]

    fx_path = os.path.join(logs_dir, "flux_test.log")
    sources["flux_test"] = (parse_synth_log(fx_path)
                            if os.path.isfile(fx_path) else None)

    gen_path = os.path.join(logs_dir, "data", "synth_pilot_L4",
                            "generation_log.jsonl")
    sources["generation_log"] = (parse_generation_log(gen_path)
                                 if os.path.isfile(gen_path) else None)

    report_date = pick_date(sources, args.date)
    out_path = args.out or os.path.join(logs_dir, "reports",
                                        "NIGHT_%s.md" % report_date)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    report = build_report(logs_dir, report_date, sources)
    with open(out_path, "w") as fh:
        fh.write(report)
    print("wrote %s (%d bytes)" % (out_path, len(report)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
