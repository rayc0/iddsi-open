# GitHub repo settings — copy-paste ready

Measured empty today (2026-10-12, `api.github.com/repos/rayc0/iddsi-open`):
`description = None`, `homepage = None`, `topics = []`, 0 stars, 0 forks.

Apply at **Settings → General** (description, homepage, topics) — or via
`gh repo edit`. Nothing here changes repo content.

## Description (GitHub caps at 350 chars; keep it under 200)

```
Open-source smartphone grader for the 10 mL / 10 s IDDSI Flow Test, plus a
measured negative result on photo classification for IDDSI texture levels L3-L7.
Research preview for culinary education; weights and dataset withheld.
```

## Homepage

```
https://huggingface.co/lingualeap/iddsi-open-siglip2-v0
```

Swap this for the arXiv abstract URL once #8 lands — an arXiv URL is the more
durable landing page and is what reviewers look for.

## Topics (12)

GitHub topic rules: lowercase, letters/digits/hyphens, no spaces, max 50 chars.

```
iddsi
dysphagia
food-texture
texture-modified-food
computer-vision
opencv
ordinal-classification
calibrated-abstention
negative-result
culinary-education
research-preview
siglip2
```

Rationale, in the order a browser would use them:

- `iddsi`, `dysphagia`, `texture-modified-food` — the clinical/food-science
  search terms. This is the audience that cites.
- `computer-vision`, `opencv`, `siglip2` — the ML audience that stars and forks.
- `ordinal-classification`, `calibrated-abstention`, `negative-result` — the
  methods audience. `negative-result` is deliberately included: the v0.6
  "gate outcome is a property of the yardstick" finding is the most citable
  thing in the repo and should be findable by topic.
- `culinary-education`, `research-preview` — signals scope and prevents the
  clinical misreading before anyone opens the card.

Do **not** add: `swallowing`, `safety`, `medical`, `clinical`, `healthcare`.
They misdescribe the release and would attract exactly the audience the
regulatory posture (§5 of the paper) is engineered to exclude.

## `CITATION.cff`

See `CITATION.cff` in this directory. Drop it at the repo root. This is what
makes GitHub render the "Cite this repository" button — currently absent
because the file does not exist.

## Badge row for `README.md`

Currently `README.md` has one external link and zero badges. Insert directly
under the H1. Every URL below must be replaced with the real one as each
channel goes live; do not ship a badge pointing at a 404.

```markdown
[![PyPI version](https://img.shields.io/pypi/v/iddsi-flowtest.svg)](https://pypi.org/project/iddsi-flowtest/)
[![PyPI downloads](https://img.shields.io/pypi/dm/iddsi-flowtest.svg)](https://pypi.org/project/iddsi-flowtest/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![flowtest tests](https://github.com/rayc0/iddsi-open/actions/workflows/flowtest.yml/badge.svg)](https://github.com/rayc0/iddsi-open/actions/workflows/flowtest.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

## Enable GitHub Pages

`has_pages = False` today. `docs/capture_kit_A4.pdf` already exists, so the
repo has publishable documentation; a Pages site is not required, but turning
Pages on makes the repo's PDFs linkable at a stable URL, which is what a
reviewer wants when a citation needs to point at a specific artefact. Low
priority — do this after #5's higher-value items.

## Licence detection

`license.spdx_id = NOASSERTION` while `LICENSE` is a 10 374-byte Apache-2.0
text. Most likely stale detection. Re-check after the next push; if it persists,
add `SPDX-License-Identifier: Apache-2.0` as the first line of `LICENSE`.
