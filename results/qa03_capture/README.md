# qa03_capture — the L3 flow-test accounting defect

Defect (HIGH severity, found by lane `qa03_capture`): `ops/capture.py` asked the L3
deciding test `flow_ml_remaining` as free text but computed the specimen verdict as
`all(v in ("pass", "na") for v in tests.values())`, so a numeric reading could never
be in that set and every L3 specimen was recorded as FAIL.

Scope, measured from `capture/protocol.json`: `flow_ml_remaining` is in the
`applicable_tests` of exactly one level — **L3, 59 of the 267 specimens (22.1% of the
corpus)**. L4/L5/L6/L7 are pass/fail/na words only.

## Files

| file | what it is |
|---|---|
| `repro_before_fix.json` / `.log` | measured on the **pre-fix** `ops/capture.py` (commit `90008cb`), 18 scenarios |
| `repro_after_fix.json` / `.log` | measured on the **fixed** `ops/capture.py`, same 18 scenarios |

Each JSON records the `code_commit` it ran against, so the two can never be confused.

## Reproduce both halves

```bash
# BEFORE — extract the pre-fix module, then point the repro at it
BEFORE=$(mktemp -d); mkdir -p "$BEFORE/ops" "$BEFORE/capture"
git show 90008cb:ops/capture.py > "$BEFORE/ops/capture.py"
cp capture/protocol.json "$BEFORE/capture/"
python3 ops/repro_qa03_capture.py --repo "$BEFORE" \
        --code-commit 90008cb --evidence results/qa03_capture/repro_before_fix.json
# exit 1, 6 problems: 0 pass / 59 fail with a reading of 9 ml

# AFTER — this checkout
python3 ops/repro_qa03_capture.py --evidence results/qa03_capture/repro_after_fix.json
# exit 0, no problems
```

## Measured, before → after (59 L3 specimens per run)

| scripted reading | card rule | before | after |
|---|---|---|---|
| `9` ml | pass | 0 pass / 59 fail | 59 pass / 0 fail |
| `8` ml (lower bound) | pass | 0 / 59 | 59 / 0 |
| `10` ml (upper bound) | pass | 0 / 59 | 59 / 0 |
| `9 ml` | pass | 0 / 59 | 59 / 0 |
| `7` ml | fail | 0 / 59 | 0 / 59 |
| `7.9` ml | fail | 0 / 59 | 0 / 59 |
| `10.1` ml | fail | 0 / 59 | 0 / 59 |
| `11` ml | fail | 0 / 59 | 0 / 59 |
| unparseable, then `9` | pass | 0 / 59 | 59 / 0 |
| `9` ml + fork_drip fail | fail | 0 / 59 | 0 / 59 |
| `9` ml + spoon_tilt na | pass | 0 / 59 | 59 / 0 |

Controls, identical before and after: L4 58, L5 55, L6 51, L7 44 specimens — all-pass
records all pass, all-fail records all fail, and one failing applicable test fails the
specimen.

Gate: `ops/tests/test_capture_flow.py` — 67 tests, `python -m pytest ops/tests -q`
(89 passed in `ops/tests` overall: 67 new + 22 pre-existing).

Hazard entry: **H-CAPTUREL3** in `hf/HAZARD_LOG.md`.

No physical specimen has been captured with this tool (`capture/records/` does not
exist), so no published label or figure moves.
