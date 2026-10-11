# HF model card — proposed YAML front matter

`UNVERIFIED-HERE`: huggingface.co is unreachable from this sandbox
(`https://huggingface.co/api/models/lingualeap` → 000, measured 2026-10-12).
The keys below are the standard Hub model-card metadata fields; **re-verify
each one against current Hub documentation before applying.**

## Current state (`hf/model_README.md`, lines 1-8)

```yaml
---
language: en
pipeline_tag: image-classification
tags:
- iddsi
- food-texture
- research-preview
- siglip2
license: apache-2.0
---
```

## Proposed

```yaml
---
language: en
pipeline_tag: image-classification
library_name: transformers        # UNVERIFIED-HERE: only if the Hub still
                                  # accepts this value for a repo with no
                                  # weights; otherwise omit.
base_model: google/siglip2-base-patch16-224
tags:
- iddsi
- dysphagia
- food-texture
- texture-modified-food
- ordinal-classification
- coral
- calibrated-abstention
- negative-result
- research-preview
- siglip2
- flow-test
license: apache-2.0
arxiv: XXXX.XXXXX                 # fill in once the preprint exists (#8).
                                  # This is the single highest-value key: it is
                                  # what links the repo into HF Papers, which
                                  # is where paperswithcode.com now redirects.
datasets: lingualeap/iddsi-flowtest-synthetic-bench   # once #4 exists
pretty_name: IDDSI-Open v0.7
---
```

## Judgement call: keep `pipeline_tag: image-classification`

This tag puts the repo on the image-classification task page, where visitors
expect downloadable weights. Recommendation: **keep it.** Removing it costs
discovery, and the repo genuinely is an image-classification research artefact.
The mitigation is on the card body, not the metadata.

## Required card-body addition — "What you can actually download today"

Insert immediately after the `## What this is` section's bullet list. This is the
block that converts the one thing a visitor will be disappointed by into three
clicks that count:

```markdown
## What you can actually download today

This repo ships **no model weights** (see above) and **no image dataset**
(licence grounds). What does ship, and what is genuinely useful:

| Artefact | Where | What it does |
|---|---|---|
| `iddsi-flowtest` | `pip install iddsi-flowtest` | Deterministic OpenCV grader for the 10 mL / 10 s IDDSI Flow Test. No ML, no GPU, no network. |
| Synthetic flow-test bench | `<dataset repo>` | 100-case stress bench, regenerated deterministically in one command. No harvested media, no licence questions. |
| Demo Space | `<space repo>` | Generate or upload a Flow-Test clip and see the reading and the abstention reasons. |
| Eval results + code | this repo | Every number in this card, with per-seed JSON and bootstrap CIs. |

The weights are withheld because the photo-classification head failed its
pre-registered gates. That is a deliberate decision, not an oversight — see
"Why the weights are withheld" above.
```

## SEO note on the title

The current title is "IDDSI-Open research preview — HONEST release card".
"HONEST" is doing real work in this project's culture and should not be
removed, but it is not a search term. The `pretty_name` field above is where
the plain, searchable name belongs; the H1 can keep its voice.
