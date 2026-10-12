# Abstention-threshold sweep on the v0.7 VALIDATION split (lane fix02_threshold)

> Research demonstration for culinary education. Estimates visual similarity to IDDSI
> descriptors; does not perform official IDDSI tests, assess swallowing, determine
> suitability for any person, or decide whether food is safe to consume. Never replaces
> clinician-prescribed texture or physical IDDSI testing. This project is not the official
> IDDSI website and is not endorsed by IDDSI.

## Question

Wave 1 measured that on the v0.7 held-out **test** split (n=108) no seed emits L5 or L6 as a
decision — per-level recall L5 0.00, L6 0.00 (`analysis/errors_v07/outputs/structural_seed43.json`).
Two mechanisms produce that symptom:

1. the CORAL rank decode never **produces** L5/L6 (a representation failure), or
2. the decode produces them but every such decode falls below the confidence threshold
   (an abstention-gate failure).

This lane sweeps the confidence threshold over **every** value achievable on the
**validation** split and reports coverage, per-level recall (L5 and L6 especially),
dangerous-direction FNR and ECE at each point.

## Headline answer

**Yes for reachability, no for recall.** On validation the shipped decode produces L5
(4–5 of 110 events per seed) and L6 (11–15) *before* abstention, and lowering the threshold
below the CORAL interior mass ceiling makes both appear as decisions — at **no cost in
coverage** (coverage rises, it does not fall). But **L5 recall is 0.000 at every threshold
for every seed**, because not one of the six true-L5 validation events is ever decoded as
L5. The cost of reachability is dangerous-direction FNR, not coverage.

Read [`/work/lane_runs/fix02_threshold/threshold_sweep_val.md`](../../../lane_runs/fix02_threshold/threshold_sweep_val.md)
for the full tables.

## Reproduce

The container `iddsi-train` had lost CUDA when this lane ran, so the validation forward
pass runs on CPU. That is validated in step 4 against a GPU-produced dump: the cut-point
logits are **bit-identical** to `/work/lane_runs/an03_calibration/val_predictions_seed43.npz`
(max abs difference 0.0 over the 110x4 matrix, 0 of 110 decoded-level disagreements).

```bash
# 1. rebuild the CALIBRATED validation probabilities for all five v0.7 seeds
docker exec -u 1000:1000 -w /work/lanes2/fix02_threshold/trainer iddsi-train \
  python /work/lanes2/fix02_threshold/analysis/threshold_sweep/dump_val_predictions.py \
    --device cpu --no-amp --suffix _cpu

# 2. sweep every achievable threshold on VALIDATION only
docker exec -u 1000:1000 -w /work/lanes2/fix02_threshold/trainer iddsi-train \
  python /work/lanes2/fix02_threshold/analysis/threshold_sweep/sweep_thresholds.py --suffix _cpu

# 3. render the report
docker exec -u 1000:1000 -w /work/lanes2/fix02_threshold/trainer iddsi-train \
  python /work/lanes2/fix02_threshold/analysis/threshold_sweep/render_report.py

# 4. CPU-vs-GPU cross-check, then assert every claim in the report against the JSON
docker exec -u 1000:1000 -w /work/lanes2/fix02_threshold/trainer iddsi-train \
  python /work/lanes2/fix02_threshold/analysis/threshold_sweep/crosscheck_cpu_vs_gpu.py
docker exec -u 1000:1000 -w /work/lanes2/fix02_threshold/trainer iddsi-train \
  python /work/lanes2/fix02_threshold/analysis/threshold_sweep/verify_findings.py
```

CPU-only, no GPU gate needed. Run as uid 1000 so nothing lands root-owned.

## Files

| file | what |
|---|---|
| `dump_val_predictions.py` | rebuilds the calibrated validation probabilities per seed from the shipped `best.pt`; re-fits the temperature and re-selects the threshold as a self-check |
| `sweep_thresholds.py` | the exhaustive per-threshold sweep: coverage, per-level recall with denominators, three dangerous-FNR definitions, ECE (all and covered), pre-abstention decode census, interior reachability, emitted-L5/L6 composition |
| `render_report.py` | renders `threshold_sweep_val.md` from the sweep JSON |
| `crosscheck_cpu_vs_gpu.py` | compares the CPU forward pass against the GPU-produced reference |
| `verify_findings.py` | 133 assertions tying every claim in the report to a value in the JSON/NPZ |
| `outputs/` | copies of the artifacts committed with this lane |

## Inputs (read-only; nothing sealed was opened)

- `/work/data/manifest_v07_weighted.jsonl` — **`split == "val"` rows only** (110 events)
- `/work/runs/v07/siglip2_v07_fullcoverage_seed{17,29,43,71,101}/best.pt`
- `/work/runs/v07/siglip2_v07_fullcoverage_seed{17,29,43,71,101}/calibration.json`
- `/work/lane_runs/an03_calibration/val_predictions_seed43.npz` — GPU cross-check reference

The held-out test split is never loaded by any script in this directory.

## Outputs (canonical location)

Everything lands under `/work/lane_runs/fix02_threshold/`:

| file | what |
|---|---|
| `threshold_sweep_val.md` | the report |
| `threshold_sweep_val.json` | the full sweep: every candidate threshold x every metric, per seed |
| `threshold_sweep_val.csv` | the round-number grid (0.00-1.00) as flat rows |
| `threshold_sweep_val_answer.json` | the per-seed reachability answer and boundary points |
| `val_predictions_seed{17,29,43,71,101}_cpu.npz` | calibrated validation probabilities + raw cut-point logits |
| `val_dump_summary_cpu.json` | temperature/threshold reproduction deltas per seed |
| `verify_log.txt` | the 133/133 assertion log |
| `val_predictions_seed{17,29,43,71,101}_cpu.npz` | calibrated validation probabilities + raw cut-point logits, per seed |
| `crosscheck_cpu_vs_gpu.json` | the CPU-vs-GPU logit/decode comparison |

## Method notes

- **The threshold grid is exhaustive, not sampled.** Acceptance is
  `confidence >= threshold`, so every threshold between two consecutive distinct confidence
  values gives an identical acceptance set. The grid is
  `{0.0} ∪ {distinct confidences} ∪ {nextafter(max)} ∪ {shipped threshold}` = 113 candidates
  per seed.
- **Three dangerous-direction FNR definitions are reported, each with its denominator.**
  `dangerous_fnr_abstention_detector` (accepted dangerous / all dangerous),
  `dangerous_underclassification_fnr_covered` (accepted dangerous / all covered — the
  shipped `eval.metrics` name), and `dangerous_underclassification_rate_all` (accepted
  dangerous / all N). H-ABSTBLIND records that the last is the only one that improves when
  the model abstains.
- **ECE** is `eval.metrics.expected_calibration_error`, 10 equal-width bins on the CORAL
  rank-decode confidence, on the 110 validation events. ECE (all) is threshold-free and is
  repeated in every row so the tables are self-contained.
- **The decode is temperature-invariant** (`P(y>k) > 0.5` is `cutpoint_logit_k > 0`, and a
  positive temperature preserves the sign), verified at 0 of 110 disagreements between T=1
  and the shipped T for every seed — so the pre-abstention census is a property of the
  checkpoint, not of the calibration.
- **Code provenance is pinned.** Every script inserts this lane's `trainer/` at `sys.path[0]`,
  shadowing the container's editable install of another lane's clone, and prints the sha256
  of each module it imports into the output JSON.

## Interpretation boundary

Research demonstration for culinary education. Estimates visual similarity to IDDSI
descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability
for any person, or decide whether food is safe to consume. Never replaces
clinician-prescribed texture or physical IDDSI testing. This project is not the official
IDDSI website and is not endorsed by IDDSI. Training labels are VLM weak labels, not
physical IDDSI tests and not clinician-validated; every figure above is measured against the
inter-judge label noise recorded in `hf/HAZARD_LOG.md` (H-NOISE, H-QWENBIAS) and means
nothing about real-world accuracy. See the new hazard entry **H-THRGATE** in that log.
