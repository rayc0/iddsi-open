# HOW_TO_CONSUME.md — the documented paths that consume this release

Written 2026-10-12. Every command in this file was run on the project's own box
before this file was committed; §5 holds the pasted output. Nothing here is a
plan, and nothing here is a claim about use that did not happen.

**Read this first.** This release publishes **no model weights**. The release
shape was locked on 2026-09-07 (code + eval results + the flow-test grader +
the measured negative finding), the photo-classification head fails the
project's own pre-registered gates (dangerous under-classification 33%), and
`ops/stage_release.sh` hard-refuses to stage a `*.pt` / `*.safetensors` /
`*.bin` file. So every path below consumes **code and eval evidence**, not a
classifier. There is no command in this repository that downloads a model that
can be pointed at a plate of food, and none should be written.

The release is not a swallowing-safety tool, never issues a verdict, and makes
no dietary, clinical or patient claim. Culinary-education and research use
only, with confirmation by the applicable physical IDDSI test.

---

## 1. What you are consuming

| Artifact | Where it lives | What it is |
|---|---|---|
| `results/fix1_siglip2_coral_seed17/test_predictions.npz` | Hub + GitHub | 109 held-out events: CORAL probabilities, labels, event ids |
| `results/fix1_siglip2_coral_seed17/metrics.json` | Hub + GitHub | the published metrics + the validation-calibrated threshold |
| `results/fix1_siglip2_coral_seed17/resolved_config.yaml` | Hub + GitHub | the exact resolved training config (architecture, phases, calibration) |
| `results/fix1_siglip2_coral_seed17/calibration.json` | Hub + GitHub | temperature, coverage, under-FNR and *why* the gate was not met |
| `flowtest/`, `trainer/`, `crowd/`, `harvest/`, `synth/`, `dataschema/`, `tools/` | Hub + GitHub | the code, including the flow-test grader that actually works |

Hub repo: `lingualeap/iddsi-open-siglip2-v0`. Pinned revision: **`v0.7`**.
`main` carries unreleased launch-prep commits — a reproducible consumption
never floats on it.

Every fetch writes a receipt (`hub_receipt.json`) naming the endpoint that
served the bytes, the resolved commit, and the SHA-256 of each file, so any
number you compute can be traced to the exact artifact it came from.

## 2. Endpoint selection — your mirror on a CN network, the real endpoint elsewhere

One function decides, `eval.hub_source.resolve_endpoint`, in this order:

1. an explicit `endpoint=` / `--hub-endpoint` / `hub.endpoint:` in the config;
2. `IDDSI_HF_ENDPOINT` (project-specific, wins over the generic variable);
3. `HF_ENDPOINT` (the generic `huggingface_hub` variable);
4. probe `https://huggingface.co` — **the real endpoint, used outside CN**;
5. probe `https://<your-mirror>` — **the CN mirror, used on this project's box**;
6. otherwise raise. **There is no silent local fallback**: a run that cannot
   reach a Hub endpoint fails, it does not quietly read a local file and call
   it consumption.

Measured on this box (2026-10-12, from inside the `iddsi-train` container):

| endpoint | reachable from here? | evidence |
|---|---|---|
| `https://huggingface.co` | **no** — SNI-blocked, TLS reset | `_probe(REAL_ENDPOINT)` → `False` |
| `https://<your-mirror>` | **yes** | `_probe(MIRROR_ENDPOINT)` → `True` |

So the auto-resolution lands on the mirror here and on the real endpoint on a
GitHub Actions runner, without either machine needing a tunnel. The container
image already exports `HF_ENDPOINT=https://<your-mirror>`, which is why a run
inside it reports `provenance=env:HF_ENDPOINT`.

```bash
# Pin one explicitly (overrides every probe):
export IDDSI_HF_ENDPOINT=https://<your-mirror>      # CN network
unset IDDSI_HF_ENDPOINT HF_ENDPOINT                 # elsewhere: probes huggingface.co first
```

## 3. Install

```bash
python -m pip install -e "trainer[hub]"    # + PyYAML, numpy, scikit-learn, Pillow (core deps)
```

The `[hub]` extra is `huggingface_hub>=0.24`; it is optional so a training-only
install does not pull the Hub client, but every command in this file needs it.

## 4. The paths

### (a) The capture/eval pipeline — reproduce the published metrics from Hub bytes

```bash
python tools/consume_release.py \
  --config configs/consume/hub_release.yaml \
  --step eval \
  --output-dir out/eval
```

Equivalent, as a bare CLI with no config file:

```bash
cd trainer
python -m eval.cli \
  --predictions 'hub:results/fix1_siglip2_coral_seed17/test_predictions.npz' \
  --threshold   'hub:results/fix1_siglip2_coral_seed17/metrics.json' \
  --output-dir out/eval
```

Both pull the two artifacts from the Hub, evaluate them with the released
threshold, and write `metrics.json`, `report.md` and `hub_receipt.json`.

### (a) The capture pipeline — resolve the released model source

```bash
python tools/consume_release.py \
  --config configs/consume/hub_release.yaml \
  --step capture \
  --output-dir out/capture
```

This pulls the released `resolved_config.yaml` **and the Hub file list**, then
reports the architecture, the calibrated threshold, and — measured from that
file list, not assumed — how many weight files the release publishes. The
Gradio capture app uses the same function; point it at the Hub with one
variable:

```bash
CROWD_MODEL_SOURCE='hub:results/fix1_siglip2_coral_seed17/resolved_config.yaml' \
  python crowd/app.py
```

`CROWD_MODEL_SOURCE` may also hold a local checkpoint path, and unset it keeps
the historical offline behaviour — a deployment that never opted into the Hub
is not silently made network-dependent.

### (b) The CI smoke test

```yaml
# .github/workflows/hub-smoke.yml — runs the same command, on both endpoints
- name: Consume the release from the Hub
  env:
    IDDSI_HF_ENDPOINT: ${{ matrix.endpoint }}   # "" = auto-resolve
    IDDSI_HUB_CACHE: ${{ runner.temp }}/hf-cache-${{ matrix.endpoint_name }}
  run: |
    python tools/consume_release.py \
      --config configs/consume/hub_release.yaml \
      --step smoke \
      --output-dir out
```

Three jobs: `unit` (offline tests of endpoint precedence and receipt honesty),
`smoke` (both endpoint legs), and `endpoints-agree` (the two legs must have
served byte-identical artifacts — a mirror answering from a stale cache fails
the run). Locally, the identical command is:

```bash
python tools/consume_release.py \
  --config configs/consume/hub_release.yaml \
  --step smoke \
  --output-dir out/smoke
```

`--step smoke` runs the eval step, runs the capture step, then asserts every
`expect:` entry and three release-shape checks: a receipt must exist, it must
not be entirely cache hits, and it must resolve a commit. If the Hub ever
starts publishing weights, the smoke test **fails** until the docs are
updated — a smoke test that cannot fail is not a test.

### The config snippet

```yaml
# configs/consume/hub_release.yaml
hub:
  repo: lingualeap/iddsi-open-siglip2-v0
  revision: v0.7          # pinned tag; never `main`
  endpoint: ""            # "" = auto-resolve (§2). No local fallback exists.
  cache_dir: ""           # "" = the huggingface_hub default cache
  require_network: true   # re-download so the endpoint provably served the bytes
artifacts:
  predictions: hub:results/fix1_siglip2_coral_seed17/test_predictions.npz
  threshold:   hub:results/fix1_siglip2_coral_seed17/metrics.json
  model_source: hub:results/fix1_siglip2_coral_seed17/resolved_config.yaml
  calibration: hub:results/fix1_siglip2_coral_seed17/calibration.json
eval:
  bootstrap_samples: 1000
  bootstrap_seed: 2026
expect:
  n: 109
  macro_f1_with_abstentions: 0.3578
  weighted_kappa_covered: 0.4227
  coverage: 0.8807
```

A `hub:` reference is `hub:<rfilename>` (repo and revision from `hub:`) or the
explicit `hub:<repo>@<revision>:<rfilename>`.

## 5. What these paths actually produced (pasted, not summarised)

Run 2026-10-12, inside the `iddsi-train` container, cache under
`/work/lane_runs/rel05_consume/hf_cache`, revision `v0.7`:

```
$ python tools/consume_release.py --config configs/consume/hub_release.yaml --step smoke --output-dir /work/lane_runs/rel05_consume/smoke_run1
consume: config=/work/lanes2/rel05_consume/configs/consume/hub_release.yaml step=smoke repo=lingualeap/iddsi-open-siglip2-v0@v0.7 endpoint=(auto) cache_dir=(default) require_network=True
hub: repo=lingualeap/iddsi-open-siglip2-v0@v0.7 commit=af40a912ed8c57e1f11518c1063164e173ce5dca endpoint=https://<your-mirror> provenance=env:HF_ENDPOINT files=2 bytes=17006
{"report": "/work/lane_runs/rel05_consume/smoke_run1/eval/report.md", "metrics": "/work/lane_runs/rel05_consume/smoke_run1/eval/metrics.json", "hub_receipt": "/work/lane_runs/rel05_consume/smoke_run1/eval/hub_receipt.json"}
hub model source: lingualeap/iddsi-open-siglip2-v0@v0.7 (commit af40a912ed8c57e1f11518c1063164e173ce5dca) via https://<your-mirror> — env:HF_ENDPOINT
weights published for lingualeap/iddsi-open-siglip2-v0@v0.7: False (0 file(s) matching .pt, .pth, .safetensors, .bin, .ckpt)
released architecture: {"dropout": 0.1, "kind": "siglip2", "name": "/work/models_local/siglip2-base-patch16-224", "pretrained": true}
capture estimate: status=placeholder_no_weights predicted_level=None is_placeholder=True
capture detail: No weights are published for this release (wk1 kill rule; dangerous under-classification 33%). The estimate below is the uniform placeholder, not a model output. Provenance — hub model source: lingualeap/iddsi-open-siglip2-v0@v0.7 (commit af40a912ed8c57e1f11518c1063164e173ce5dca) via https://<your-mirror> — env:HF_ENDPOINT.
smoke test PASSED: 9 assertions + release-shape checks
wrote /work/lane_runs/rel05_consume/smoke_run1/consume_summary.json
```

The fail-closed behaviour, same command with a dead endpoint pinned:

```
$ IDDSI_HF_ENDPOINT=http://127.0.0.1:1 python tools/consume_release.py --step smoke ...
CONSUME FAILED: could not pull lingualeap/iddsi-open-siglip2-v0@v0.7 via the auto-resolved endpoint: ValueError: Force download failed due to the above error.
(exit 2)
```

Full logs and every JSON receipt: `/work/lane_runs/rel05_consume/` in the
project's own lane output (see `runs/`).

## 6. What the numbers mean, and what they do not

The metrics reproduced by path (a) are the **fix1 seed-17** run on **109
held-out events**, with the validation-calibrated threshold `0.36494785003441793`:

| metric | value | denominator |
|---|---:|---|
| `macro_f1_with_abstentions` | 0.3578 | all 109 events; abstentions counted as misses |
| `weighted_kappa_covered` | 0.4227 | the 96 covered (non-abstained) events |
| `dangerous_underclassification_fnr_covered` | 0.3333 | covered events whose truth is a higher level |
| `dangerous_underclassification_rate_all` | 0.2936 | all 109 events |
| `coverage` | 0.8807 | 96 of 109 events |
| `abstention_rate` | 0.1193 | 13 of 109 events |
| `ece_all` / `ece_covered` | 0.0906 / 0.0856 | all / covered events |

Three things a consumer must not do with them:

- **Do not read them as accuracy.** They are measured against a label set the
  project built itself from two VLM judges who agree with each other on 31.0%
  of the same images (n=575 pairs, κ_unweighted 0.1062, κ_quadratic 0.2766).
  See hazard **H-YARDSTICK** in `hf/HAZARD_LOG.md`.
- **Do not mix the kappas.** `weighted_kappa_covered` (0.4227 here) is computed
  on covered events only. `results/v07_release/bias_matrix.json` holds kappa
  over **all** rows including abstentions — a different quantity. Hazard
  **H-YARDEVID**.
- **Do not read the v0.7 range as this run.** The published 0.838–0.860 is the
  v0.7 five-seed range on 108 events against the v0.7 full-coverage consensus.
  The 109-event fix1 numbers above are a different split and a different
  yardstick. Denominators are not interchangeable.

`L5` and `L6` are never emitted as a decision in any of the five v0.7 seeds
(`analysis/errors_v07/outputs/structural_seed43.json`). A consumer reproducing
these metrics will not see them either; that is the finding, not a bug in the
path.

## 7. Known limits of these paths

- **The real endpoint was not exercised from this box.** `huggingface.co` is
  SNI-blocked here (measured, §2). The `real` leg of `hub-smoke.yml` runs it on
  a GitHub-hosted runner, which is outside CN; that workflow cannot be executed
  from this machine, so its green status is not claimed in this file. Every
  other command here was run and its output pasted.
- **`require_network: true` re-downloads ~17 KB per run.** That is the point —
  a receipt built from cache hits names an endpoint that never answered
  (hazard **H-HUBCACHE**). Set it false only when you do not need proof of
  consumption.
- **The capture path still returns a placeholder.** It will until weights are
  published, which the wk1 kill rule says they should not be. When that
  changes, point `CROWD_MODEL_SOURCE` at the published checkpoint and the same
  code path loads it.
- **`downloads` on the Hub repo is not inflated by these runs.** Measured
  2026-10-12 from the Hub API: `downloads: 0`, `likes: 0`, repo sha
  `af40a912ed8c57e1f11518c1063164e173ce5dca`, last modified 2026-10-10T18:25:37Z.
  Self-generated pulls that do not correspond to real use are fraud against the
  Hub's terms and against this project's governance; the growth plan says so
  explicitly (`hf/GROWTH_PLAN.md`, §3 L6).

## 8. Repo map for this path

| file | role |
|---|---|
| `trainer/eval/hub_source.py` | the single implementation of "pull the release from the Hub": endpoint resolution, `hub:` refs, receipts, the consume-config loader, the `expect:` checker |
| `trainer/eval/cli.py` | `python -m eval.cli`; `run_eval()` is the shared implementation the smoke test calls |
| `crowd/inference.py` | the capture pipeline's model-source resolution (`CROWD_MODEL_SOURCE`) |
| `tools/consume_release.py` | `--step eval \\| capture \\| smoke` |
| `configs/consume/hub_release.yaml` | the config snippet all three read |
| `.github/workflows/hub-smoke.yml` | the CI smoke test, both endpoints |
| `trainer/tests/test_hub_consume.py` | offline tests: endpoint precedence, receipt honesty, assertions |
| `crowd/tests/test_inference_hub.py` | offline tests: capture-path wiring |
