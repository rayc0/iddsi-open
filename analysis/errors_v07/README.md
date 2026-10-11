# v0.7 per-level error analysis (lane an02_errors)

Reproducible per-level error analysis of the best v0.7 seed. **Read
[`outputs/findings.md`](outputs/findings.md) for the findings** — every number in it
is asserted against the JSON in `outputs/` by `verify_findings.py`.

## Reproduce

```bash
# artifacts are the per-seed test NPZ + calibration + published metrics,
# copied out of /work/runs/v07/siglip2_v07_fullcoverage_seed*/
mkdir -p /work/lane_runs/an02_errors/artifacts
for s in 17 29 43 71 101; do
  docker cp iddsi-train:/work/runs/v07/siglip2_v07_fullcoverage_seed$s/test_predictions.npz \
    /work/lane_runs/an02_errors/artifacts/v07_seed${s}_test_predictions.npz
  docker cp iddsi-train:/work/runs/v07/siglip2_v07_fullcoverage_seed$s/metrics.json \
    /work/lane_runs/an02_errors/artifacts/v07_seed${s}_metrics.published.json
  docker cp iddsi-train:/work/runs/v07/siglip2_v07_fullcoverage_seed$s/calibration.json \
    /work/lane_runs/an02_errors/artifacts/v07_seed${s}_calibration.json
done

docker exec -u 1000:1000 -w /work/lanes/an02_errors/trainer iddsi-train \
  python /work/lanes/an02_errors/analysis/errors_v07/analyze_errors.py
docker exec -u 1000:1000 -w /work/lanes/an02_errors/trainer iddsi-train \
  python /work/lanes/an02_errors/analysis/errors_v07/analyze_structure.py
docker exec -u 1000:1000 iddsi-train \
  python /work/lanes/an02_errors/analysis/errors_v07/verify_findings.py
```

CPU-only; no GPU gate needed. Run as uid 1000 so nothing lands root-owned.

## Files

| file | what |
|---|---|
| `analyze_errors.py` | seed selection, confusion matrices, adjacent-pair and over/under bias analysis, group axes, cross-seed stability |
| `analyze_structure.py` | emitted-level census, abstention analysis, harvest-bucket vs consensus labels, bootstrap CIs, headline-metric reproduction |
| `verify_findings.py` | 121 assertions tying every claim in `findings.md` to a value in `outputs/*.json` |
| `outputs/` | the JSON/CSV/MD this lane produced (raw NPZ stays in `/work/lane_runs/an02_errors/artifacts/`) |

## Inputs (read-only; nothing sealed was opened)

`/work/data/manifest_v07_weighted.jsonl`, `/work/data/harvest_{d1,d1b}/events.jsonl`,
`/work/data/realweak_pilot/events.jsonl`,
`/work/lanes/an02_errors/trainer/train/model.py` (`coral_decode`).
