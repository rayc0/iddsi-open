---
pretty_name: IDDSI Flow-Test Synthetic Stress Bench (n=100)
license: other
task_categories:
  - video-classification
task_ids:
  - video-classification
language:
  - en
tags:
  - iddsi
  - dysphagia
  - flow-test
  - syringe
  - synthetic
  - benchmark
  - abstention
  - opencv
size_categories:
  - n<1K
---

# IDDSI Flow-Test synthetic stress bench (n=100)

> **Synthetic pipeline-development benchmark only. Not real-video validation
> and not a safety benchmark.** Every clip in this dataset is drawn by
> `flowtest/synthetic.py` from a documented parameter grid. It contains no
> harvested, licensed or patient media.

## One line

A 100-case synthetic stress bench for grading the 10 mL / 10 s IDDSI syringe
Flow Test from video — regenerate it deterministically in one command, with no
licence questions and no real footage.

## What this is

`flowtest` is a deterministic OpenCV pipeline that measures the residual volume
in a filmed **10 mL / 10 second syringe Flow Test**, detects the calibrated
61.5 mm barrel, verifies absolute scale from an 85.60 × 53.98 mm ID-1 card,
finds the release cue, tracks the meniscus, and reads it exactly 10 seconds
after release. It uses no learned model, no GPU and no network service.

This dataset is the benchmark for that grader. It is **synthetic by design**:
the grader's job is photogrammetry on a known geometry, so a synthetic clip has
a known ground-truth residual volume and can be generated to arbitrary
precision. That makes the bench exactly reproducible and free of any
third-party licence.

**The bench is a stress test, not an accuracy claim.** On these 100 cases the
grader **abstains on 84 and grades 16**. That is the intended behaviour: the
grid deliberately includes bubble-contaminated, tilted, low-light and
boundary-adjacent cases, and a grader that answered all 100 would be a grader
with no quality control.

## Reproduce it

```bash
git clone https://github.com/rayc0/iddsi-open && cd iddsi-open/flowtest
python -m pip install -e '.[test]'
python -m flowtest.bench.run_bench --n 100 --out results_n100
```

Deterministic by construction: case `i` uses grid combo `i % len(grid)` with
`SyntheticSpec(seed = 1000 + i)`.

## Grid

| Parameter | Values |
|---|---|
| `residual_ml` | 0.5, 2.5, 4.25, 6.0, 8.4, 10.0 |
| `tilt_deg` | 0.0, 2.0, 4.0 |
| `bubbles` | false, true |
| `lighting_factor` | 1.0, 0.75, 1.15 |
| fixed | `fps 8.0`, `640 × 480` |

108 combinations. Level boundaries are 1 / 4 / 8 mL with a ±0.5 mL abstention
margin, so `residual_ml = 0.5`, `4.25` and `8.4` sit inside the margin and are
expected to abstain as `near_level_boundary`.

## Headline numbers — stated with their denominators

Measured on the shipped bench (`bench_summary.json`) and reproduced by me on
2026-10-12 in the project container (OpenCV 4.11.0, NumPy 1.26.4, Python
3.12.3):

| Quantity | Shipped file | My reproduction |
|---|---|---|
| n | 100 | 100 |
| graded | 16 | 16 |
| abstained | 84 | 84 |
| abstention rate | 0.84 | 0.84 |
| MAE, **all estimates** (mL) | 0.6224 | 0.6251 |
| MAE, **graded only** (mL) | 0.0651 | 0.0491 |
| n with a numeric estimate | 97 | 100 |

**Quote the graded-only figure only with its denominator.** "MAE 0.065 mL on a
100-video bench" is 16 of 100 cases. The honest sentence is: *"on the 16 of 100
synthetic stress cases it grades, the grader reads residual volume to ~0.05 mL;
on all 100 the error is ~0.62 mL."*

**The numbers are not bit-reproducible across OpenCV builds.** Same code, same
seeds, different build: graded-only MAE moved 0.0651 → 0.0491 and "n with
estimate" moved 97 → 100. Codec differences change the meniscus contrast, which
changes which cases pass the confidence gate. Treat the bench as a regression
check on a pinned environment, not as a cross-platform constant.

## Abstention reasons on this bench

| Code | Count |
|---|---|
| `near_level_boundary` | 47 |
| `bubbles_or_lumps_detected` | 46 |
| `wrong_or_unverified_syringe` | 32 |
| `meniscus_unclear` | 11 |

Every abstention carries machine-readable codes and a human-readable corrective
capture action. The canonical registry is `flowtest/abstention.py`
(`ABSTENTION_CODES`).

## Intended use

- Research and culinary-education use only.
- A regression bench for anyone modifying the grader.
- A worked example of an abstention-first measurement tool: the output is
  either a number with its quality checks attached, or a reason to re-film.

## Out of scope

- **Not real-video validation.** No real captures have been evaluated.
  Transparent liquids, reflections, curved menisci, foam, labels, patterned
  backgrounds, rolling shutter, autofocus, camera motion and nonstandard
  syringes may all defeat these heuristics.
- **Not a safety benchmark.** This dataset says nothing about whether any food
  or liquid is safe for any person.
- **Not IDDSI-endorsed.** This project is independent of the International
  Dysphagia Diet Standardisation Initiative and of every prior-art group named
  in the repository.

## Disclaimer

> Research demonstration for culinary education. Estimates visual similarity to
> IDDSI descriptors; does not perform official IDDSI tests, assess swallowing,
> determine suitability for any person, or decide whether food is safe to
> consume.

## Licence

The generated clips and summaries are produced entirely by project code and
carry no third-party content. Ship under the same Apache-2.0 licence as the
code. (`license: other` in the front matter above is a placeholder — set the
real value when creating the repo. `UNVERIFIED-HERE`: the Hub's current licence
key list could not be checked from this sandbox.)
