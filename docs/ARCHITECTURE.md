# IDDSI-Open — Architecture

> Consolidated 2026-09-02 by work package W34 from the fleet runboard, the DGX fleet board, and the verified `~/iddsi/reports/*.md` reports on the Spark machine. Every fact below is traceable to one of those sources; unknowns are marked **TBD**.
>
> Operational commands (how to connect, docker exec flags, fixes) live in [RUNBOOK.md](RUNBOOK.md).

## 1. What IDDSI-Open is

From `RUNBOARD.md` (frozen claims, 2026-08-31):

- A **guided IDDSI test assistant**: 45° photo + 8–12 s fork-press/spoon-tilt clip → level compatibility (L3–L7) with a mandatory "unclear — perform physical test" abstention.
- An **open-source 10 ml / 10 s Flow-Test video grader** (L0–L4), deterministic CV pipeline.
- An **open release**: dataset (physically tested labels) + model weights (SigLIP-2 head; Qwen3-VL-2B LoRA for explanation only) + HF Space + arXiv note.
- Demo posture: research / culinary-education; no patient profile; no safe/unsafe; teaches the test. Not IDDSI-endorsed. Never claim "first IDDSI AI / clinically validated / safe-to-eat".

Development runs as an AI-agent fleet; Raymond's involvement is limited to 2 approvals (HF account, external sends). Fleet coordination files: `_CC_iddsi_open_runboard.md` and `_CC_iddsi_dgx_fleet.md` in `00AIprojects/00aggray/` (OneDrive).

## 2. Repository map (`~/Projects/iddsi-open/` on the Mac, mirrored to Spark `~/iddsi/repo/`)

| Package | What it does | Evidence |
|---|---|---|
| `flowtest/` | Deterministic CV pipeline for the IDDSI Flow Test: `flowtest.grade` (grade a syringe video → residual ml / level / abstain reasons), synthetic MAE bench (`flowtest.bench.run_bench`, 100-video bench), rotation/orientation handling, ASCII-safe paths, fiducial-optional mode, clear abstention codes. 18→34 tests. | W4, W9, W13 reports |
| `dataschema/` | Event schema (`schema/`), tools and tests for dataset manifests — no installable package (no `pyproject.toml`); used via `PYTHONPATH`. `docs/` holds the dataset card + datasheet (EN + 繁中). | W3 §2, W10 |
| `trainer/` | SigLIP-2 ordinal trainer + resnet18 baseline: `train/cli.py`, `eval/cli.py` (confusion / weighted-κ / dangerous-direction FNR / calibration / bootstrap CIs), `fixtures/generate.py` (synthetic fixtures), `explain/`. 16 tests. | W3, W14 |
| `synth/` | Synthetic-image generation: FLUX/sdxl prompt bank × Cantonese 软餐 dish names (99 dishes, L3–L7, ~79K–158K distinct prompts/level after dedupe, zero cross-level descriptor leakage) + Qwen3-VL consistency judge; `synth.export_prompts` for prompt review without GPU. 19 tests. | W15, runboard log |
| `harvest/` | CC-licensed real-photo collector (Wikimedia, nutrition5k): download, licence check, pHash dedup, attribution manifest, weak labelling with `--labeler defer` (labels deferred to a later GPU pass; `harvest.relabel` runs the Qwen3-VL labeler over deferred manifests). 10 tests. | W8, W17 spec |
| `crowd/` | Gradio opt-in capture app (the demo Space): photo path → trainer inference stub, flow-test video path → `flowtest.grade`, EN + 繁中 UI with the research-preview disclaimer verbatim from `r5_reg.md`. 5 tests. | W11, W19 spec |
| `paper/` | arXiv/technical-note skeleton: `paper.md` + `references.bib` + `build.sh` (pandoc 3.9, typst engine); every result cell = `TBD`. Prior works cited only from `r5_prior.md`. | W12 |
| `ops/` | Spark operator scripts: `cleanup_rsync_temps.sh`, `night_summary.py` (+tests). | W25 spec, verified on Mac |
| `docs/` | Long-form docs: `REGULATORY_POSTURE.md` (W27), this `ARCHITECTURE.md`, `RUNBOOK.md` (W34). | verified |
| `drafts/` | Human-facing drafts: `snap_to_iddsi_page_copy.md`, `hku_outreach_email.md`, `intern_capture_brief.md`. | verified on Mac |
| root `r5_*.md`, `RUNBOARD.md` | Project analysis basis (feasibility / prior work / regulatory / solution / value) and the runboard. | verified |

## 3. Data lanes

Four provenance lanes, each tagged in the dataset schema (W20 loader design; W10 cards):

1. **`synthetic` — synthetic-first.** FLUX.1-schnell (fallback: `stabilityai/sdxl-turbo`, 4-step) on the Spark GPU × the `synth/` prompt bank × Cantonese soft-meal dishes, filtered by a Qwen3-VL-2B consistency judge. Pipeline-dev only; excluded from any released "physically tested" split. First night pilot (2026-09-01, sdxl-turbo 2-step): 0/24 accepted after 72 generations — the judge was **correct** (rice grains in "puree", a literal phone rendered in the food); the prompt bank was subsequently fixed (camera facet rephrased, negatives added) and the interim generator moved to sdxl 4-step @1024².
2. **`real-weak` — harvested CC images with deferred weak labels.** Three-stage design (W8): **Mac harvest** (network access — Wikimedia is blocked CN-side, Spark cannot harvest) → **deferred manifests** (`--labeler defer`; attribution manifest kept per image) → **GPU labelling pass on Spark** (`python -m harvest.relabel`). Pilot: 100 accepted / 357 examined; licences CC BY 4.0 ×95, CC BY-SA ×4, CC0 ×1; pHash dedup caught 68 duplicates. Lives in `~/iddsi/data/realweak_pilot/` on Spark.
3. **`real-crowd` — opt-in contributions** through the demo Space (`crowd/` app + storage, private HF dataset repo, default OFF).
4. **`real-tested` — LinguaLeap staff tests.** Ground truth = official IDDSI test methods performed **on camera** by kitchen/intern staff + recipe-declared level; AI reads the test result from the clip, human spot-check 10%. Card wording: "tests performed by LinguaLeap staff, not clinicians". (RD/SLP hiring was cancelled by Raymond 2026-08-31; RD/SLP = future validation tier only.)

Release v0 = synthetic-first + real-weak + the `增稠剂.mp4` seed; the card states no clinician testing.

## 4. Machines and roles

| Machine | Role |
|---|---|
| Mac (this machine) | Repo authoring, all pytest runs, network-dependent work (Wikimedia harvest, HF uploads, FLUX download via aria2c), fleet coordination. |
| DGX Spark (`tun@100.69.203.92`, Tailscale) | GPU training, synthetic generation, model downloads, deferred-label GPU passes. Training container + other (untouchable) service containers. |

### Spark container `iddsi-train` (verified `compose.yml`, 2026-09-02)

- Image `nvcr.m.daocloud.io/nvidia/pytorch:25.04-py3` (torch 2.7.0a0+nv25.04, transformers 5.3.0, CUDA 12.9 runtime / driver 580.142 / CUDA 13.0, NVIDIA GB10, sm_121).
- `command: sleep infinity`, `gpus: all`, `network_mode: host`, `shm_size: 16g`, `restart: unless-stopped` (durable since 2026-09-01 07:30; survives reboots — the original docker-run container died in the 2026-08-31 reboot, which motivated the compose swap, W1/MIGRATE.md).
- Volumes: `/home/tun/iddsi` (host) → `/work` (container); `/srv/spark/models` → `/models` (read-only; hosts the other services' models, e.g. deepseek-v4-flash 98 GB).
- Environment: `HF_HOME=/work/hf`, `HF_ENDPOINT=https://hf-mirror.com`, empty proxy vars, `NO_PROXY=*`.
- `~/iddsi/bootstrap.sh` (run inside the container after any recreate): pip deps + the cv2 fix + verification.

### Other containers on Spark — DO NOT TOUCH

`deepseek-v4` (llama-server on :8090, ~98–100 GB unified-memory resident when running), `ollama`, `open-webui-*`, `vault`, plus whisper/tts/llamacpp-vision processes. Only `deepseek-v4` may be **stopped/started** (never removed), and only per the night-window protocol below.

### GPU memory reality (GB10 unified memory, 130.7 GB total)

With deepseek-v4 + qwen2.5-vl + whisper + tts resident, free headroom is ~3 GB and **every** CUDA allocation OOMs (W1/W2 verified, dmesg `NV_ERR_NO_MEMORY`). All GPU work therefore runs in windows where `deepseek-v4` is paused (see RUNBOOK).

## 5. Models on disk (verified 2026-09-02)

`/work/hf/hub` (= host `~/iddsi/hf/hub`, HF_HOME cache) contains:

- `timm/resnet18.a1_in1k` (baseline)
- `google/siglip2-base-patch16-224` (backbone)
- `Qwen/Qwen3-VL-2B-Instruct` (judge / explanation)
- `black-forest-labs/FLUX.1-schnell` (generator; **gated repo** — license accepted on the `raymondchau` HF account)
- `stabilityai/sdxl-turbo` (ungated fallback generator)

`/work/models_local/` (host `~/iddsi/models_local/`):

- `siglip2-base-patch16-224/` — **the canonical training path**. The upstream checkpoint ships a sparse v1-style config that `Siglip2VisionModel.from_pretrained` cannot load on transformers 5.3.0; W14's one-time conversion (Conv2d patch weight → Linear, `num_patches=196`) produces this local copy. Idempotent step 0 of `run_pipeline.sh`.
- `FLUX.1-schnell/` — shipped from the Mac by rsync over Tailscale (32 GB; as of 2026-09-02 00:50 the main `flux1-schnell.safetensors` was still partial at 15.2/23.8 GB, `ae.safetensors` complete; T5 + CLIP shards pending). Diffusers loads it **by local path** — no HF gate at runtime. The night runner size-checks it and falls back to sdxl 4-step if incomplete.

`/srv/spark/models` → `/models` (read-only mount): the other services' models (deepseek-v4 etc.) — not for this project.

## 6. Model / training architecture

- **Backbone:** SigLIP-2 (`google/siglip2-base-patch16-224`, via the converted `/work/models_local` copy) with an ordinal head over IDDSI levels; frozen-encoder head training verified on GPU (W3, W14). Baseline: `timm/resnet18.a1_in1k`.
- **Explanation/judge:** Qwen3-VL-2B-Instruct (LoRA planned for explanation; used as the synthetic-consistency judge via `--judge-backend transformers`).
- **Eval harness** (`trainer/eval/`): per-level confusion, macro-F1 with abstentions counted as misses, weighted κ on covered events, dangerous-direction (under-classification) FNR, calibration (temperature + threshold selected on **validation only**), bootstrap CIs, risk–coverage. Inter-rater module for human baselines (W13).
- **Flow-test grader** (`flowtest/`): deterministic CV (barrel detect → normalise → release detect → meniscus → volume → quality gates → level or abstain). Synthetic-bench MAE 0.0626–0.065 mL; on the real 2021 handheld seed video it correctly abstains — real phone capture needs the v0 capture protocol (tripod, portrait ≥1080p, fiducial card, record from pull-back; W4 §6).
- **HF release scaffold (all PRIVATE, org `lingualeap`, never public):** dataset `lingualeap/iddsi-open-v0`, model `lingualeap/iddsi-open-siglip2-v0`, space `lingualeap/iddsi-open-demo` (created sdk=static because Gradio Spaces under an org need a paid HF Team plan — billing decision for Raymond).

## 7. Standing engineering rules (learned the hard way)

1. **Verification by glob:** a download is never "done" because a log says so (`DL_DONE` lied twice; an sdxl snapshot was "OK" with the weights absent). Verify by globbing the expected snapshot/weight files on disk. (Runboard 2026-08-31 07:45, 05:00.)
2. **Offline env when cached:** export `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for training runs — otherwise `AutoImageProcessor.from_pretrained` retried the hub for 13m39s despite a full cache. (W3 §6.)
3. **HF mirror always:** `huggingface.co` DNS is poisoned inside the container (resolves to a Facebook IPv6) — always `HF_ENDPOINT=https://hf-mirror.com` and empty proxy env. (Fleet board; W2.)
4. **cv2/FFMPEG trap:** the NVIDIA image's cv2 has no FFMPEG; the fix is `/work/fix_cv2.sh` / `bootstrap.sh` (quarantine + force-reinstall opencv-python-headless 4.11). (W4, W14; runboard 2026-09-01 07:30 for the RECORD trap.)
5. **Night window for GPU:** deepseek-v4 owns the day; GPU work lives in the 23:00–08:00 window with deepseek paused and **always restored after**. (Runboard 2026-09-01 07:45.)
6. **pkill-in-ssh trap:** a `pkill <pattern>` inside an ssh command kills the ssh session itself — always bracket-escape the pattern (`pkill -f "[r]estore"`). (Runboard 2026-08-31 10:56.)
7. **Guard patterns must heartbeat:** a "restore guard" that stays alive with an empty log failed silently; any long-running watcher must heartbeat-log. (Runboard 2026-08-31 10:56, 2026-09-01 07:45.)
