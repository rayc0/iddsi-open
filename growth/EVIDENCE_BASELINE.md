# Measured baseline — 2026-10-12 (lane hf04_growth_plan)

Every number below was produced by me in this run. Nothing here is copied from
another lane or restated from a prior release note. Command + output file are
named for each.

## A. GitHub discovery surface (the repo people actually land on)

Source: `GET https://api.github.com/repos/rayc0/iddsi-open`, 2026-10-12.

| Field | Measured value |
|---|---|
| `description` | `None` — **empty** |
| `homepage` | `None` — **empty** |
| `topics` | `[]` — **empty** |
| `stargazers_count` | 0 |
| `forks_count` | 0 |
| `watchers_count` | 0 |
| `open_issues_count` | 0 |
| `license.spdx_id` | `NOASSERTION` (GitHub reports "Other" despite a 10 374-byte Apache-2.0 `LICENSE`) |
| `has_pages` | `False` |
| `has_wiki` | `True` |
| `created_at` / `pushed_at` | 2026-09-15 / 2026-10-11T15:24:53Z |
| `default_branch` / `size` | `main` / 1 886 KB |

`README.md` contains **exactly one** external link (the HF repo, line 6) and
**zero** badges. There is no `.github/` directory in the tree
(`git ls-files` → no `.github/**`), so no issue templates, no
`FUNDING`/`CITATION` discovery, no CI badge.

## B. Network reachability from this sandbox (decides what I could verify)

| Target | Result |
|---|---|
| `api.github.com` | 200 |
| `pypi.org` | 200 |
| `arxiv.org` / `export.arxiv.org` API | 200 |
| `doi.org/10.3389/fnut.2026.1829703` (prior-art citation) | 200 |
| `iddsi.org` | 301 |
| `paperswithcode.com` | **302 → `huggingface.co/papers/trending`** |
| `huggingface.co` (any path, incl. `/api/models/lingualeap`) | **000 — unreachable** |

Two consequences that shape this whole plan:

1. **Papers with Code no longer exists as a separate site.** It 302-redirects to
   HF Papers. "PwC linkage" must be planned as *HF Papers + arXiv*.
2. **I could not read or publish anything on Hugging Face.** Every HF-specific
   mechanic below is therefore marked `UNVERIFIED-HERE` and must be re-checked
   by whoever executes the step from a network that reaches huggingface.co.

## C. PyPI — the only channel that produces counted downloads

`GET https://pypi.org/pypi/<name>/json`, 2026-10-12, all **404 = available**:

- `iddsi-flowtest`
- `iddsi-open`
- `iddsi-open-flowtest`
- `iddsi-flowtest-grader`

`flowtest/pyproject.toml` already declares `name = "iddsi-flowtest"`,
`version = "0.1.0"`, deps `numpy>=1.24` + `opencv-python-headless>=4.8`,
`requires-python >=3.10`. The name is free and the packaging metadata exists.
**Nothing has been published.**

## D. Does the code actually work? (measured in container `iddsi-train`)

| Suite | cwd | Result |
|---|---|---|
| `flowtest` | `/work/lanes/hf04_growth_plan/flowtest` | 60 items → **58 passed, 1 skipped, 1 FAILED** |
| `trainer` | `/work/lanes/hf04_growth_plan/trainer` | 37 items → 37 passed |
| `ops` | `/work/lanes/hf04_growth_plan/ops` | 22 items → 22 passed in 0.33 s |
| `tools` | `/work/lanes/hf04_growth_plan/tools` | 4 items → 3 passed, 1 skipped |

Output: `lane_runs/hf04_growth_plan/flowtest_pytest.txt` (pytest 8.1.1,
Python 3.12.3).

The single failure in the repository is in the one component this plan wants to
promote:

```
FAILED tests/test_w9_fixes.py::test_fiducial_optional_mode_grades_uncalibrated
  abstention_reasons = ['bubbles_or_lumps_detected', 'uncalibrated_scale_mode']
  status = 'abstain'   (test expected 'graded')
  estimated_residual_ml = 9.723, boundary_distance_ml = 1.723
```

`flowtest/README.md` documents `--fiducial-optional` as "lets the grader proceed
without the card". The test asserts it does. It does not. **Publishing a PyPI
package or a Space while its own documented mode has a red test is the single
most avoidable way to lose a first-time user.**

## E. The one headline number is not reproducible and is quoted out of context

Reproduced by me: `python -m flowtest.bench.run_bench --n 100 --out
lane_runs/hf04_growth_plan/bench_n100_repro`, container `iddsi-train`,
cv2 4.11.0 / numpy 1.26.4 / Python 3.12.3.

| Quantity | Shipped `flowtest/flowtest/bench/results_n100/bench_summary.md` | My repro |
|---|---|---|
| n | 100 | 100 |
| graded | 16 | 16 |
| abstained | 84 | 84 |
| abstention rate | 0.84 | 0.84 |
| **MAE mL (all estimates)** | 0.6224 | **0.6251** |
| **MAE mL (graded only)** | **0.0651** | **0.0491** |
| n with estimate | 97 | 100 |

Three separate problems, all measured:

1. **"MAE 0.065 ml on a 100-video bench" is the graded-only number over 16 of
   100 videos.** The all-estimates number on the same bench is 0.62 mL. The
   top-level `README.md`, `hf/model_README.md` and `paper/paper.md` (§3.4,
   §6, §7) all quote the 0.065 figure without the 16/100 qualifier.
2. **It is not reproducible.** Same code, same seed rule (`seed 1000 + i`, 108
   grid combos, `i % len(grid)`), different OpenCV build → graded-only MAE moves
   0.0651 → 0.0491 and "n with estimate" moves 97 → 100.
3. **The bench is synthetic.** `bench_summary.json`'s own `scope` field reads
   `"synthetic pipeline-development benchmark only; not real-video validation
   and not a safety benchmark"`. `flowtest/README.md` §"Honest limitations"
   already says "No real-video validation has been performed… Do not quote the
   synthetic MAE as product accuracy." The three public-facing documents
   contradict that.

The claim that survives all three filters is narrow and still strong:
*"a deterministic, dependency-light OpenCV grader for the 10 mL / 10 s IDDSI
Flow Test that abstains on 84 of 100 synthetic stress cases and, on the 16 it
grades, reads residual volume to ~0.05 mL."* That is the sentence to publish.
It is more interesting than the current one, and it cannot be walked back.

## F. The citable paper is five versions behind the card it must match

- `paper/paper.md` — 486 lines, dated **"7 September 2026"**, `subtitle:
  "Technical note - v0 research preview"`. Every result cell is v0/v0.1:
  weighted kappa 0.42, macro-F1 0.358, dangerous under-classification 33%,
  inter-judge agreement 31.0% on n=575, flow MAE 0.065 ml.
- `hf/model_README.md` — 441 lines, carries **v0.2, v0.3, v0.5, v0.6 and v0.7**
  sections, including the v0.6 finding "the gate outcome depends on the
  yardstick, not the model" and the v0.5 cross-family consensus result.

Nothing from v0.2→v0.7 appears in the paper. A reader who opens the paper and
then the card sees a document that its own project has superseded. arXiv is
irreversible: once a preprint is posted it is public and indexed permanently,
so **posting the current `paper.md` would permanently attach a stale, weaker
set of claims to the project's name.**

Related defect: `paper/sections/related_work.md` (131 lines) and
`paper/sections/limitations.md` (157 lines) are **orphaned**.
`paper/build.sh` runs `pandoc paper.md …` and `paper.md` has no include
directive (grep: no matches for `include`). 288 lines of written prose never
reach `paper.html` / `paper.pdf`.

## G. Metric naming will not survive a reviewer diff

`results/v07_release/*_metrics.json` (5 seeds, held-out n=108 each):

| seed | macro-F1 | κ weighted (covered) | duc FNR (covered) | `…_rate_all` | coverage | ECE all |
|---|---|---|---|---|---|---|
| 101 | 0.3399 | 0.8604 | 0.0353 | 0.0278 | 0.7870 | 0.0644 |
| 17 | 0.3594 | 0.8498 | 0.0562 | 0.0463 | 0.8241 | 0.0240 |
| 29 | 0.3317 | 0.8504 | 0.0349 | 0.0278 | 0.7963 | 0.0774 |
| 43 | 0.3949 | 0.8507 | 0.0435 | 0.0370 | 0.8519 | 0.0301 |
| 71 | 0.3193 | 0.8375 | 0.0353 | 0.0278 | 0.7870 | 0.0447 |
| **range** | 0.3193–0.3949 | 0.8375–0.8604 | 0.0349–0.0562 | **0.0278–0.0463** | 0.7870–0.8519 | 0.0240–0.0774 |

`trainer/eval/metrics.py:77` defines that fourth column as
`((raw_predictions < labels) & accepted).mean()` — the numerator is gated on
acceptance while the denominator is all N. `hf/model_README.md` itself names
this defect **H-ABSTBLIND** in its v0.3 section and publishes a corrected
all-N rate instead (v0.3: 0.176–0.282; v0.7 table: 0.093–0.130).

So the same words, "under-classification, all-N", currently resolve to three
different numbers depending on whether you read the paper (33%), the card's
corrected v0.7 row (0.093–0.130) or the seed JSONs (0.0278–0.0463). Any one of
them is defensible in isolation. All three together read as either confusion or
gaming. **A metric dictionary is a prerequisite for a citable paper, not a
nice-to-have.**

## H. arXiv has essentially no IDDSI literature (this is the opportunity)

`export.arxiv.org/api/query`, metadata search, 2026-10-12:

| Query | `totalResults` |
|---|---|
| `all:"IDDSI"` | **0** |
| `all:"International Dysphagia Diet Standardisation Initiative"` | **0** |
| `abs:"dysphagia"` | **15** |
| `abs:"food texture" AND abs:"classification"` | **0** |
| `all:"dysphagia" AND all:"texture"` | 1 |
| `all:"texture-modified" AND all:"food"` | 1 |

The community's literature is in clinical and food-science journals
(`fnut.2026.1829703` resolves 200), not on arXiv. A cs.CV/cs.LG preprint on
IDDSI texture vision would face no arXiv competition at all — but it also means
arXiv visibility alone will not reach the people who cite this work. The
citation channel and the download channel are **different channels** here.
