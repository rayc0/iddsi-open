# 🏃 RUNBOARD — IDDSI-Open (Y06 milestone receipt) · started 2026-08-31
Owner: AI (Raymond: "i dun wanna coordinate anything for u, u fix everything yourself") · Raymond touches: 2 approvals only (HF account, external sends)
Analysis basis: `_expert_iddsi_2026-08-31/` (5 sol lanes) · Decision `recvtNRzB8LkSa` · PENDING A334

## Product (frozen claims)
- Guided IDDSI test assistant: 45° photo + 8–12 s fork-press/spoon-tilt clip → level compatibility (L3–L7) with mandatory "unclear — perform physical test" abstention.
- Open-source 10 ml / 10 s Flow-Test video grader (L0–L4), deterministic CV pipeline.
- Open dataset (physically tested labels) + model weights (SigLIP-2 head; Qwen3-VL-2B LoRA for explanation only) + HF Space + arXiv note.
- Claim: "first publicly downloadable IDDSI-specific smartphone dataset + fine-tuned model under explicit reuse licence, physically tested Chinese/Cantonese soft-meal labels" · "first open-source smartphone grader for the IDDSI Flow Test". NEVER "first IDDSI AI / clinically validated / safe-to-eat".
- Demo posture: research / culinary-education; no patient profile; no safe/unsafe; teaches the test. Not IDDSI-endorsed.

## Phases
| Phase | What | Needs Raymond? | Status |
|---|---|---|---|
| A0 | Spark training container (pytorch:25.04 + transformers/peft/timm/opencv/datasets) | no | 🔄 2026-08-31 |
| A1 | Repo `~/Projects/iddsi-open`: data schema, capture-protocol spec, flow-test CV pipeline, SigLIP-2 ordinal trainer, eval harness (confusion/κ/FNR/calibration) | no | 🔄 sol lanes dispatched |
| A2 | Bootstrap data for PIPELINE DEV ONLY: synthetic + CC images at declared levels, `增稠剂.mp4` flow seed — tagged `synthetic`, EXCLUDED from any released "physically tested" split | no | ⏳ |
| A3 | Drafts: (a) softened seniordeli snap-to-iddsi page copy, (b) Karen Chan outreach email, (c) intern capture brief (15-min sessions, per-level shot list, syringe protocol) | drafts only | ⏳ |
| B1 | HF account/org (LinguaLeap) + token on this Mac | **YES — only Raymond can create/verify** | 🔴 ask in A334 |
| B2 | Real capture: interns shoot ≥2,000 preps + ≥1,000 flow videos in 15-min sessions | approve brief (c) → Ada via Raymond's own channel | ⏳ |
| B3 | Ground truth: RD + SLP physical tests (Y06 Outsourcing/Consultancy budget HK$80K) | approve engagement | ⏳ |
| B4 | HKU blind validation (Karen Chan lab) | approve email (b) | ⏳ |
| C | Train on real data → gates wk1–wk4 → dataset/model/Space/arXiv → first post | no (post text = draft for approval) | ⏳ |

## Gates (from sol)
wk1 expert photo-only weighted κ ≥0.70 else kill photo-classification · wk2 external macro-F1 ≥0.75, dangerous under-classification ≤5% at ≤30% abstention · wk3 flow ≥95% non-boundary agreement, MAE ≤0.25 ml · wk4 blinded ≥300-event audit. Fail → "Y06 research preview" only.

## Log
- 2026-08-31 — Spark checked (2.7 TB free, pytorch:25.04 image present, no torch on host). No HF token on this Mac. In-house imagery ≈140 files; only shoot = `Work/0seniordeli/增稠剂.mp4`. Runboard created; A0/A1 started.
