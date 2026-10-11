---
title: IDDSI-Open — Visual Texture-Level Research Demo
emoji: 🍚
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.23.2
app_file: app.py
pinned: false
license: apache-2.0
short_description: Estimate visual similarity to IDDSI L3–L7 texture levels from a food photo. Research demo, not a swallowing-safety tool.
---

# IDDSI-Open — visual texture-level research demo

> **⚠️ PLACEHOLDER HEAD — no trained weights are shipped.** The v0.7 SigLIP-2
> CORAL weights are **withheld** (see [the release card](../hf/model_README.md),
> "Why the weights are withheld"), so this Space runs a clearly-labelled
> **uniform placeholder** and reports **no IDDSI level at all**. It abstains
> with `unclear — perform physical test`. The interface, the CORAL ordinal
> decode and the fail-closed abstention path are all real and tested; the
> numbers are not.

Upload a food photograph and the demo returns a probability distribution over
**IDDSI texture levels L3–L7**, estimated from appearance alone.

## What this is

A culinary-education / QA **research demonstration**. It estimates *visual
similarity* to IDDSI texture descriptors from a single photograph.

## What this is NOT

- **Not a swallowing-safety classifier.** It does not assess swallowing,
  aspiration risk, or suitability for any person.
- **Not an IDDSI test.** It performs no official IDDSI test and measures no
  flow, hardness, adhesiveness or cohesiveness.
- **Not a verdict.** It never says food is safe or unsafe to consume, and never
  tells anyone what texture to prepare, modify or prescribe.
- **Not IDDSI-endorsed.** This project is independent of IDDSI.

**Disclaimer (English).** Research demonstration for culinary education.
Estimates visual similarity to IDDSI descriptors; does not perform official
IDDSI tests, assess swallowing, determine suitability for any person, or decide
whether food is safe to consume.

**免責聲明（繁體中文）。** 烹飪教學研究示範。本工具僅估算食物外觀與 IDDSI
描述之間的視覺相似度；並不執行 IDDSI 官方測試、不評估吞嚥能力、不判斷食物
是否適合任何人士，亦不決定食物是否安全食用。

## Why the weights are withheld

The photo-classification head fails the project's own pre-registered safety
gates. Dangerous under-classification — predicting a *softer* texture than
truth — was measured at **33%** (covered) in v0.1, and although abstention now
intercepts most of the model's own dangerous calls, the pre-registered
**wk1 gate (weighted κ ≥ 0.70) fires on every v0.7 seed** (best 0.860 against
the v0.7 labels, and the v0.6 finding
[H-YARDSTICK](../hf/HAZARD_LOG.md) is that a gate outcome is a property of the
yardstick rather than of the model). A downloadable classifier would be used
for verdicts whatever a card says, and a disclaimer does not cure a claim.
Withholding the weights is the mitigation.

Labels are machine weak labels from two independent model families, which agree
on an IDDSI level from a web photograph about **three times in four**
(251/337 = 74.5% on the doubled-coverage sample). They are *consistent*, not
*correct*. The physically-tested desk-specimen protocol — 300 specimens, one
operator, fully specified in `docs/CAPTURE_PROTOCOL.md` — has never been run and
remains the only route to ground truth.

## Running the weighted model in this Space anyway

The app already implements the weighted path. Set one environment variable (or
drop a checkpoint next to `app.py`) and the PLACEHOLDER banner disappears:

| variable | meaning |
|---|---|
| `IDDSI_SPACE_CHECKPOINT` | path to a `.pt` payload `{"metadata": {"model_kind", "model_name", "levels"}, "state_dict": ...}` |
| `IDDSI_SPACE_BACKBONE` | path or HF id of the SigLIP-2 tower (default `/work/models_local/siglip2-base-patch16-224`) |
| `IDDSI_SPACE_ABSTAIN_THRESHOLD` | confidence below which the demo abstains (default `0.5189`, the most conservative of the five v0.7 validation calibrations) |

Accepted checkpoint format (same one `crowd/inference.py` loads):

```python
payload = {
    "metadata": {"model_kind": "siglip2", "model_name": "...", "levels": [3, 4, 5, 6, 7]},
    "state_dict": model.state_dict(),
}
torch.save(payload, "best.pt")
```

## Decode rule — why argmax is not used

CORAL reconstructs class probabilities as **differences of adjacent sigmoids**,
so interior classes are capped near 0.17 while both tails keep their full mass.
Argmax over that distribution provably cannot return L4, L5 or L6 for *any*
input; the first trained run emitted only L3, L7 and abstain. This Space
therefore decodes with the ordinal rank rule:

```
level = #{k : P(y>k) > 0.5}
```

See hazard **H-CORAL** in [`hf/HAZARD_LOG.md`](../hf/HAZARD_LOG.md).

## Intended use

- Research preview for reproducible methods work on IDDSI-related vision.
- Culinary-education use only, with confirmation by the applicable physical
  IDDSI test.

## Out of scope

- No safe or unsafe verdicts. The demo never issues a verdict.
- No dietary, clinical, or patient claims.
- Not for assessing swallowing, aspiration risk, or suitability for any person.
- Not for prescribing or modifying a texture.
- Not IDDSI-endorsed.

## Licence

Apache-2.0, matching the rest of the release.
