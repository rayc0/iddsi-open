# IDDSI-Open — Spark Operator Runbook

> Consolidated 2026-09-02 by work package W34 from the fleet board, the runboard, `~/iddsi/compose.yml` / `bootstrap.sh` / `night_pilot.sh` / `fix_cv2.sh` (read live on Spark), and the verified WP reports in `~/iddsi/reports/`. System/data context: [ARCHITECTURE.md](ARCHITECTURE.md).

## 1. Connect

```bash
# Spark over Tailscale (BatchMode OK — key auth is set up)
ssh -o BatchMode=yes tun@100.69.203.92
```

Key host paths:

| Host path | Container path | What |
|---|---|---|
| `~/iddsi` | `/work` | All project state (repo, hf cache, data, runs, reports, scripts) |
| `~/iddsi/hf` | `/work/hf` | `HF_HOME` (hub cache) |
| `~/iddsi/repo` | `/work/repo` | rsync mirror of the Mac repo |
| `~/iddsi/data` | `/work/data` | `seed/`, `realweak_pilot/`, `synth_pilot_L4*/` |
| `~/iddsi/runs` | `/work/runs` | pipeline run outputs (`<timestamp>/` per run) |
| `~/iddsi/reports` | — | all work-package reports |
| `~/iddsi/models_local` | `/work/models_local` | converted SigLIP-2, shipped FLUX.1-schnell |
| `/srv/spark/models` | `/models` (read-only) | other services' models — not ours |

Repo sync from the Mac (always with these excludes — root-owned `__pycache__` on Spark causes rsync exit 23):

```bash
rsync -az --no-times --exclude .git --exclude __pycache__ --exclude .pytest_cache \
    ~/Projects/iddsi-open/ tun@100.69.203.92:~/iddsi/repo/
```

## 2. The docker exec environment flags (MANDATORY on every exec)

The container's default proxy env breaks huggingface.co, and HF DNS inside the container is poisoned (huggingface.co resolves to a Facebook IPv6 — direct calls hang). Every `docker exec` that touches the network or the HF cache must pass the full flag set:

```bash
docker exec \
    -e HTTP_PROXY= -e HTTPS_PROXY= -e http_proxy= -e https_proxy= \
    -e NO_PROXY='*' \
    -e HF_HOME=/work/hf \
    -e HF_ENDPOINT=https://hf-mirror.com \
    iddsi-train bash -c "<command>"
```

Notes (W2):

- Root inside the container **cannot** reach huggingface.co directly (host TPROXY OUTPUT rule bypasses the proxy for root only); uid 1000 can. Either way, downloads must go through `HF_ENDPOINT=https://hf-mirror.com`.
- `compose.yml` now bakes all of these env vars in, but they take effect only at the next container recreate — pass them explicitly until then (and in any case for one-off execs).
- For training runs with models already cached, also export `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` — without them `from_pretrained` retried the hub for 13m39s despite a complete cache (W3).

## 3. Container lifecycle

`iddsi-train` runs under `~/iddsi/compose.yml` (image `nvcr.m.daocloud.io/nvidia/pytorch:25.04-py3`, `restart: unless-stopped`, host network, 16g shm, `sleep infinity`). It survives reboots. After any recreate, re-run the bootstrap:

```bash
# Recreate (only when sanctioned — never while fleet lanes are live):
ssh tun@100.69.203.92 'cd ~/iddsi && docker stop iddsi-train && docker rm iddsi-train && docker compose -f compose.yml up -d'

# Then, inside the container (deps + cv2 fix + verification, idempotent):
docker exec -it iddsi-train bash /work/bootstrap.sh

# Sanity:
docker exec iddsi-train nvidia-smi
docker exec iddsi-train python -c "import torch; print(torch.cuda.is_available())"
```

`/work` is a host bind mount — nothing is lost on recreate.

## 4. The cv2/FFMPEG fix

**Trap:** the NVIDIA pytorch:25.04 image ships a source-built `opencv` 4.10 whose cv2 has **no FFMPEG** (`cv2.VideoCapture(...).isOpened() == False` for every video, `cv2.VideoWriter` mp4v fails). It collides with pip's `opencv-python-headless` — two dist-info RECORDs both claim `cv2/`, and the NVIDIA build wins.

**Fix (already applied in the live container; `bootstrap.sh` embeds it for fresh containers):** `/work/fix_cv2.sh`

- Quarantines the NVIDIA cv2 (moves `cv2/` + `opencv*` from `/usr/local/lib/python3.12/dist-packages` into `/work/_quarantine_cv2/` — nothing deleted).
- `pip install opencv-python-headless==4.11.0.86` from the Tsinghua PyPI mirror (`-i https://pypi.tuna.tsinghua.edu.cn/simple`).
- **RECORD trap (2026-09-01):** after the quarantine, pip's surviving RECORD says "already satisfied" while the import is broken → bootstrap force-reinstalls: `pip install --force-reinstall --no-deps --no-cache-dir opencv-python-headless==4.11.0.86`.
- Verifies `FFMPEG: YES` and that both seed videos open.

Check the current state:

```bash
docker exec iddsi-train python -c "import cv2; print(cv2.__version__); print([l.strip() for l in cv2.getBuildInformation().splitlines() if l.strip().startswith('FFMPEG')])"
# expect: 4.11.0 / ['FFMPEG:                      YES']
```

## 5. Night window (GPU scheduling)

**Pattern (standing decision, 2026-09-01):** the GPU is scheduled at night; `deepseek-v4` owns the day. Daytime free memory is ~3–4 GB (deepseek ~98–100 GB + voice stack ≈ 114/121 GB) — every CUDA alloc OOMs. GPU work lives in the **23:00–08:00 HKT** window with deepseek paused.

Reference implementation: `~/iddsi/night_pilot.sh` (v2, setsid-detached, heartbeat-logged every 10 min):

1. Waits for 23:00 local (heartbeat every 10 min — a silent alive-but-empty-log guard failed before; always heartbeat).
2. `docker stop deepseek-v4` → 5 s settle.
3. **Size-checks FLUX** (`~/iddsi/models_local/FLUX.1-schnell/flux1-schnell.safetensors` > 23,000,000,000 bytes AND `text_encoder_2/model-00002-of-00002.safetensors` non-empty): FLUX 4-step if complete, else `stabilityai/sdxl-turbo` 4-step fallback.
4. FLUX GPU load test (512², 4 steps → `~/iddsi/flux_test.log`), then the synth pilot (`python -m synth.generate … --judge-backend transformers`, all with `-e HF_HUB_OFFLINE=1`).
5. Verifies outputs **on disk** (jpg count + accepted rows in `generation_log.jsonl`).
6. `docker start deepseek-v4` → wait ~150 s → `curl -s http://127.0.0.1:8090/healthz`.

### Starting GPU work manually

```bash
docker stop deepseek-v4          # frees ~98 GB of unified memory
# ... GPU work (check free memory first: docker exec iddsi-train python -c "import torch; print(torch.cuda.mem_get_info())") ...
```

### Restoring deepseek-v4 afterwards (MANDATORY)

```bash
docker start deepseek-v4
sleep 150
curl -s -m 10 http://127.0.0.1:8090/healthz   # expect a healthy response after ~2 min
```

Never restore mid-training; never remove the container; never touch `ollama` / `open-webui-*` / `vault` / whisper / tts.

## 6. FLUX gating

- `black-forest-labs/FLUX.1-schnell` is a **gated HF repo** (license accepted on the `raymondchau` account). Run FLUX **generation only when GPU work is sanctioned** — i.e. inside the night window with deepseek paused; never as a daytime ad-hoc job.
- Runtime loads FLUX **by local path** (`/work/models_local/FLUX.1-schnell`) — no HF gate at runtime.
- The Mac→Spark ship is a detached rsync (`~/flux_ship.sh`, log `~/flux_ship.log` on the Mac, look for `FLUX_SHIPPED`). Before any FLUX run, size-check the main safetensors (≥ ~23.8 GB expected; see night_pilot.sh's check) — the ship has been partial for days and stale `.flux1-schnell.safetensors.*` rsync temp files sit next to it (delete only after `FLUX_SHIPPED`, per `ops/cleanup_rsync_temps.sh`).

## 7. Downloads: verify by glob, never by log

Standing rule: **no download is "done" until the expected files are globbed on disk.** Twice a log line said complete (`DL_DONE`, "OK sdxl") while weights were absent (DNS flake mid-run, resume bug). Verify e.g.:

```bash
ssh tun@100.69.203.92 'ls -la ~/iddsi/hf/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*/ ; \
  find ~/iddsi/hf/hub -name "*.incomplete" | head'
```

Download anything new via the mirror with the §2 env flags (do not start duplicates — check `pgrep -af snapshot_download` first).

## 8. Running the tests

All suites, inside the container (59 passed as of W7; suite counts grow as lanes land):

```bash
docker exec -e HTTP_PROXY= -e HTTPS_PROXY= -e http_proxy= -e https_proxy= \
    -e NO_PROXY='*' -e HF_HOME=/work/hf -e HF_ENDPOINT=https://hf-mirror.com \
    iddsi-train bash -c "cd /work/repo && python3 -m pytest -q flowtest dataschema trainer synth harvest crowd"
```

On the Mac (repo root): `python -m pytest -q <dir>` per package.

## 9. One-command pipeline

`bash ~/iddsi/run_pipeline.sh` (on the host) runs inside the container: SigLIP-2 checkpoint conversion (idempotent step 0 — see below) → synthetic fixtures → tiny_cnn CPU smoke → SigLIP-2 head 1 epoch (GPU) → resnet18 baseline 1 epoch (GPU) → eval reports → flowtest synthetic MAE bench. Outputs land in `~/iddsi/runs/<timestamp>/`. Reference run: `~/iddsi/runs/20260831_030025/` (flow MAE 0.0626 ml, 5/5 graded).

**SigLIP-2 conversion fact:** `google/siglip2-base-patch16-224` ships a sparse v1-style config that cannot load on transformers 5.3.0 — always train from the converted local copy `/work/models_local/siglip2-base-patch16-224`, NOT the HF cache id.

## 10. Common traps → fixes (quick table)

| Trap | Symptom | Fix |
|---|---|---|
| Proxy env in exec | HF/network calls hang or fail | §2 flag set on every `docker exec` |
| HF DNS poisoned in-container | huggingface.co resolves to a Facebook IPv6; direct calls hang | `HF_ENDPOINT=https://hf-mirror.com` |
| NVIDIA cv2 without FFMPEG | `VideoCapture`/`VideoWriter` fail on every video | §4 fix_cv2 / bootstrap.sh (quarantine + force-reinstall 4.11) |
| pip RECORD survives quarantine | "already satisfied" but `import cv2` broken | `pip install --force-reinstall --no-deps opencv-python-headless==4.11.0.86` |
| Daytime CUDA OOM | `NV_ERR_NO_MEMORY` on any alloc | deepseek-v4 owns the day — schedule GPU in the 23:00–08:00 window (§5) |
| False-complete downloads | log says done, weights absent | verify by globbing files on disk (§7) |
| Hub re-check despite cache | 13m39s stall in `from_pretrained` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` |
| `pkill` inside ssh kills the ssh itself | session dies mid-command | bracket-escape the pattern: `pkill -f "[r]estore"` |
| rsync exit 23 to Spark | root-owned `__pycache__` dirs | `--exclude __pycache__ --exclude .pytest_cache` (and `--no-times`) |
| Silent watcher failure | guard process alive, log empty, nothing fired | heartbeat-log every few minutes (night_pilot.sh pattern) |
| Wikimedia blocked from Spark | harvest downloads fail CN-side | harvest runs on the **Mac**; labels deferred → GPU pass on Spark (`python -m harvest.relabel --dataset /work/data/realweak_pilot`) |

## 11. Rules of engagement on the Spark machine

- Never delete anything; quarantine instead (cv2 pattern above).
- Never touch `deepseek-v4` beyond stop/start per §5, and never `ollama` / `open-webui-*` / `vault`.
- No network-exposed services.
- Write work-package reports to `~/iddsi/reports/<WP>.md`; verified facts (pasted command output) only, never claims.
