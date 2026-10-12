# HF_METADATA.md — exact Hub metadata for `lingualeap/iddsi-open-siglip2-v0`

Companion to `hf/model_README.md` (the card source). This file specifies the **exact**
YAML front matter and tag list to set on the Hub repository, the reasoning for each
value, the 10 target search phrases with the audit that confirms each appears in the
card, and the legitimacy guardrails this optimisation stays inside.

Everything here is **legitimate discoverability only**: correct metadata, honest tags,
accurate prose. No download inflation, no fake engagement, no keyword stuffing that
misrepresents what the model does.

Lane `rel01_modelcard`, 2026-10-12, nonce `6f9e400bf8dc`. The card is 826 lines /
~51.7k characters; `hf/verify_card_claims.py` passes on it **10/10 search phrases,
17/17 disclaimer markers, 16/16 hazard entries, 67/67 claim checks** (see §7).

---

## 1. The front matter — paste this verbatim at the top of the Hub README

The Hub repo README **is** `hf/model_README.md`. The first 27 lines of that file are the
front matter; they are the very first thing in the file, with no blank line before the
opening `---` (verified: line 1 and line 27 are both `---`).

```yaml
---
language: en
license: apache-2.0
pipeline_tag: image-classification
library_name: transformers
base_model: google/siglip2-base-patch16-224
tags:
- image-classification
- food
- food-texture
- food-image-classification
- texture
- texture-analysis
- dysphagia
- iddsi
- iddsi-texture-levels
- dietary
- texture-modified-diet
- ordinal-classification
- coral
- siglip2
- calibrated-abstention
- selective-classification
- culinary-education
- research-preview
- not-for-clinical-use
---
```

`hf/verify_card_claims.py` parses the front matter out of `hf/model_README.md` and
asserts it is byte-identical to the block above; a mismatch fails the run. Field order
does not matter to the parser, but keep it as written so the file's line numbers stay
meaningful.

### Field-by-field

| field | value | why this value |
|---|---|---|
| `pipeline_tag` | `image-classification` | The task is multi-class image classification over 5 ordered IDDSI levels. `image-classification` is the correct Hub pipeline tag for a fine-tuned classifier head; `zero-shot-image-classification` would be wrong (there is no text prompt at inference). Setting it also makes the repo appear under the Image-Classification task filter. |
| `library_name` | `transformers` | The SigLIP-2 tower is loaded through `transformers` (`trainer/train/model.py`, `OrdinalImageModel._siglip_tower`), and the training/eval code is a `transformers` + `torch` stack. `transformers` is the value that makes the Hub show the framework chip and route the repo to the transformers filter. |
| `base_model` | `google/siglip2-base-patch16-224` | Exact upstream checkpoint ID. This creates the base-model relation on the Hub, so anyone browsing `google/siglip2-base-patch16-224`'s "models trained from this" list finds this repo. |
| `language` | `en` | The card, code comments and output strings are English. Vision-only model, so language is metadata only — no multilingual claim is made. |
| `license` | `apache-2.0` | Matches the repo `LICENSE`. Applies to **code**; no weights are published, so no weights licence is granted. The card states this explicitly. |
| `tags` | see §2 | 19 tags across four facets: task, domain, method, and an honesty marker. |

### Fields deliberately omitted, and why

| field | decision | reason |
|---|---|---|
| `model-index` | **not used** | A `model-index` block renders as a benchmark results table on the Hub. Every metric here is conditional on a label set the project built itself (hazard **H-YARDSTICK**), and a rendered results block would read as a benchmark win. The metrics live in the card body as markdown tables with their caveats attached. See §5 for an optional snippet if a future release ever wants it. |
| `datasets` | **not used** | No dataset is published — the harvested corpus is withheld on licence grounds (`hf/HAZARD_LOG.md`, H-LICENCE). Adding a `datasets:` field would imply a published dataset that does not exist. |
| `arxiv` / `doi` | **not used** | No paper or DOI exists for this release. Do not add one until it does. |
| `mask_token` | **not used** | Not a masked-LM. |
| `inference: false` | **optional, not set** | Consider setting `inference: false` in the repo settings once weights are published, so the Hub does not advertise an Inference API for a repo whose whole point is that the weights are withheld. Irrelevant while there are no weights at all. |

---

## 2. The tag list, grouped by the facet it serves

**Task facet (2)**

| tag | serves |
|---|---|
| `image-classification` | the Hub's canonical task tag; also the `pipeline_tag`. Must match. |
| `food-image-classification` | long-tail task+domain phrase; the intersection nobody else occupies. |

**Domain facet (10)**

| tag | serves |
|---|---|
| `food` | the broadest domain chip; the single highest-traffic food tag. |
| `food-texture` | the domain, stated as the project states it. |
| `texture` | drops the "food" qualifier for people browsing texture analysis. |
| `texture-analysis` | the CV-side name for the same domain. |
| `dysphagia` | the clinical-adjacent domain the IDDSI scale comes from. See the guardrail in §4. |
| `iddsi` | the framework name; the highest-precision tag in the set. |
| `iddsi-texture-levels` | the specific sub-domain phrase. |
| `texture-modified-diet` | the dietetics-side name for the practice the scale serves. |
| `dietary` | the broader dietary/nutrition browsing facet. |
| `culinary-education` | the use case the release is actually scoped to. |

**Method facet (5)**

| tag | serves |
|---|---|
| `ordinal-classification` | the correct task framing; attracts the ordinal-methods audience. |
| `coral` | the specific head (Cao, Mirjalili & Raschka). Rare, high-precision. |
| `siglip2` | the backbone family; matches the `base_model` relation. |
| `calibrated-abstention` | the abstention policy; attracts the selective-prediction audience. |
| `selective-classification` | the research area the AUGRC numbers belong to. |

**Honesty marker (2)**

| tag | serves |
|---|---|
| `research-preview` | signals maturity honestly. |
| `not-for-clinical-use` | puts the limitation in the tag chips, where it is visible before the card is even opened. |

**Total: 19 tags** — task 2, domain 10, method 5, honesty 2. The block in §1 is
authoritative; the verifier re-parses it from `hf/model_README.md` and fails on any
drift.

### Tags deliberately NOT added

| rejected tag | why |
|---|---|
| `swallowing-safety`, `aspiration`, `dysphagia-screening`, `clinical` | These would promise a capability the release explicitly disclaims. The project is **not** a swallowing-safety classifier and must never be described as one. Tagging it as one would be misrepresentation, however much traffic it attracted. |
| `sota`, `state-of-the-art`, `best` | Unsupported. The card's own conclusion is that the labels are the ceiling. |
| `dataset` | No dataset ships. |
| `zero-shot-image-classification` | Wrong mechanism; there is no prompt. |
| `medical`, `healthcare`, `patient` | Would put the repo in a category the release is engineered to stay out of (SaMD class). |

---

## 3. The 10 search phrases, and the audit that each appears in the card

These are phrases a person would actually type — into the Hub search box, or into a
search engine. Each is confirmed present in `hf/model_README.md` by
`hf/verify_card_claims.py`, which normalises case and treats hyphens and em-dashes as
spaces before matching, and reports the line numbers it matched on. The bar is **at
least 2 occurrences** — once in real prose and once in the signpost list — so no phrase
is carried by the signpost block alone.

| # | search phrase | occurrences | lines | where it lands in the card |
|---|---|---|---|---|
| 1 | `iddsi texture level` | 13 | 16, 29, 46, 83, 91, 126, 809 | H1 title (29), the safety banner's audience line (46), "What this model does" where the scale is introduced (83) and where the question is posed (91), the `iddsi-texture-levels` tag in the front matter (16), the signpost list (126), and the BibTeX citation (809) |
| 2 | `food texture classification` | 2 | 94, 126 | "What this model does" — the sentence framing the project's question (94); and the signpost list (126) |
| 3 | `food image classification model` | 2 | 50, 127 | the one-line summary immediately under the safety banner (50); and the signpost list (127) |
| 4 | `texture modified diet` | 5 | 18, 83, 95, 127, 185 | "What this model does" — where the IDDSI scale is introduced (83) and where the framing question lands (95); the `texture-modified-diet` tag (18); the domain-vocabulary list (185); and the signpost list (127) |
| 5 | `dysphagia food texture` | 3 | 128, 186, 190 | "Intended use and domain" — where the origin of the scale is explained (190); the domain-vocabulary list (186); and the signpost list (128) |
| 6 | `siglip2 food` | 2 | 54, 128 | the one-line summary (54); and the signpost list (128) |
| 7 | `ordinal classification` | 4 | 19, 128, 144, 550 | "Model description" — the task row of the architecture table (144); the Delgado & Tibau objection in the limitations (550); the `ordinal-classification` tag (19); and the signpost list (128) |
| 8 | `thickened liquid` | 4 | 87, 129, 187, 682 | "What this model does" — the L0–L4 end of the scale (87); the domain-vocabulary list (187); "The flow-test grader" (682); and the signpost list (129) |
| 9 | `culinary education` | 6 | 24, 47, 129, 178, 188, 799 | the safety banner (47), "Intended use and domain" (178), the domain-vocabulary list (188), "Never-claims" (799), the `culinary-education` tag (24), and the signpost list (129) |
| 10 | `iddsi level from a photograph` | 3 | 129, 511–512, 669 | "Bias, risks and limitations" — the irreducible-noise finding, where the phrase is split across a line break at 511–512 (lines 511–512); the same sentence repeated in "Training data" under the coverage-closure paragraph (669); and the signpost list (129) |

Each phrase appears at least once in the **body prose** and at least once in the
signpost list, so the card reads naturally to a human and is not a keyword block. Several
phrases additionally match a hyphenated **tag** in the front matter (`iddsi-texture-levels`,
`texture-modified-diet`, `ordinal-classification`, `culinary-education`), because the
verifier normalises hyphens to spaces — the Hub indexes those tags for filtering, which
is the same surface from the other direction. The verifier reports both the matched
surface form and the line numbers; run output:
`lane_runs/rel01_modelcard/search_phrase_audit.json` and
`lane_runs/rel01_modelcard/card_claims_verification.md`.

### Phrases that are covered by tags but not written into the card

`food`, `texture`, `texture-analysis`, `dysphagia`, `iddsi`, `dietary`, `coral`,
`selective-classification`, `research-preview`, `not-for-clinical-use`. These are single
words or tag-shaped tokens rather than things a person types as a sentence, so they live
in the tag list (which the Hub indexes for filtering) rather than being forced into
prose. (`dysphagia` and `culinary education` do appear in prose, in the honest
"where this scale comes from" and "who this is for" contexts respectively.)

---

## 4. Legitimacy guardrails — what this optimisation did NOT do

Recorded so a reviewer can check the boundary rather than take it on trust.

- **No download inflation, no engagement farming, no fake stars, no reciprocal-link
  schemes.** Nothing in this card or this metadata file asks for or rewards any of that.
- **No misrepresentation.** The repo does not ship weights and the card says so in the
  first blockquote, above the fold, before any metric. The tags do not claim a clinical
  or safety capability the release disclaims.
- **No hidden keyword stuffing.** The ten target phrases appear in real sentences. The
  one list-shaped block ("Why you may have found this card") exists to tell an arriving
  reader what the limitation is, not to game a crawler.
- **No gate, hazard entry or disclaimer was weakened, deleted or silenced.** The
  verifier checks 16 named hazard entries from `hf/HAZARD_LOG.md` are still referenced
  in the card, plus the two pre-registered gates, the abstention-never-a-verdict rule,
  the not-IDDSI-endorsed statement, the SaMD/out-of-scope statement, the
  research-plus-culinary-education scope, the 31.0% label-noise ceiling, and the prior-art
  never-claim. Any absence fails the run. This pass **added** one hazard entry
  (**H-BENCH**, §6) rather than removing any.
- **Every number is measured and sourced.** `hf/verify_card_claims.py` re-derives the
  headline ranges from the primary artefacts (per-seed metrics JSON, the stored
  `test_predictions.npz` per seed, the v0.7 consensus manifest, the inter-judge
  cross-tab, the flow-test bench summary) rather than trusting prose. Three figures from
  earlier internal drafts were **removed** from the card because no file in the
  repository backs them; the card's provenance section says so (§6).

---

## 5. Optional: a `model-index` block, if a future release wants one

Not recommended for this release (see §1). If a future version ever has a
physically-measured anchor and wants a rendered results table, the shape would be:

```yaml
model-index:
  - name: iddsi-open-siglip2-v0
    results:
      - task:
          type: image-classification
          name: "IDDSI texture level, L3-L7 (ordinal, with abstention)"
        dataset:
          name: "IDDSI-Open held-out test split"
          type: internal
        metrics:
          - type: "weighted_kappa"
            value: 0.860
            name: "weighted kappa (covered subset, n=108, seed 43 of 5)"
            verified: false
```

Do **not** add it until the caveat can be dropped. A benchmark table whose footnotes
say "measured against labels we built ourselves" is worse than a prose table that says
the same thing, because the table is what gets scraped.

---

## 6. What this lane changed, and the three numbers it removed

`hf/model_README.md` was restructured for discoverability: a search-shaped H1, a
repo/licence/base-model line, a safety banner as the first blockquote above the fold, a
table of contents, "What this model does", "Why you may have found this card", "Model
description", "Intended use and domain" with an explicit domain-vocabulary list, and a
single cross-yardstick results section. The front matter went from 4 tags to the 19
above, and gained `library_name` and `base_model`.

The restructuring **preserved** everything the wave-1d pass had corrected, and repaired
four things that had been lost or were wrong:

1. **The benchmark-framing correction was missing.** The discoverability draft predated
   the 2026-10-12 benchmark-framing pass and had reverted to the two-reason
   "that comparison flatters us" paragraph. The card now carries the ten-point
   non-comparability list, the covered/all-events/abstain-as-miss/argmax/macro bases
   printed side by side with their decode rules named, the always-L7 baselines, the
   label-noise band, and the explicit "uninformative in both directions" conclusion.
   The hazard entry that documents this (**H-BENCH**) was declared in
   `docs/BENCHMARK_FRAMING.md` §6 and cited by the card, but **was never written** — a
   dangling cross-reference in a log whose purpose is that every claim be checkable. It
   is now written into `hf/HAZARD_LOG.md`, with every figure re-measured from the named
   artefact.
2. **A denominator error was fixed.** The draft printed *"152 of 684 events (29.8%) were
   unusable"*. 152/684 = **22.2%**; 29.8% is 152/510, a conditional rate over the
   both-judged subset. The card now states both, names each denominator, and gives the
   rule for which to quote — the rule the previous card had already written down and the
   draft had then broken.
3. **A mixed-denominator table was replaced.** The previous v0.5 results table quoted
   coverage 0.732 – 0.845 from the **n=142** per-seed metrics
   (`results/v05_release/consensus_seed*_metrics.json`) beside exact-match and MAE rows
   taken from the **n=108** `head_to_head2.json` — two denominators in one table. The
   cross-yardstick tables are now single-artefact per column, and each states its n and
   its label set.
4. **The v0.2 abstention finding was restored** with its per-seed range corrected:
   28 – 36 dangerous under-calls over 102 test events, of which the model *answers*
   wrongly on only 2 – 8, because abstention intercepts 75.0 – 93.9% of its own
   dangerous calls (`results/v02_judgelabels/honest_metrics.json`). The previous card
   said "roughly 85%"; the measured per-seed range is 75.0 – 93.9%, mean 81.3%.

**Three numbers were removed because no artefact in the repository backs them**, per the
house rule that every claim be reproducible from a named file. The card's provenance
section lists them explicitly rather than dropping them silently:

- the per-corpus harvest tallies as a total of 713 (superseded by manifest-verifiable
  counts);
- the v0.1 manifest's 704 labelled rows / 20 duplicate `event_id` values (the v0.1
  manifest is not in the repository);
- the 30-image pilot's *"100% of disagreements with Qwen-8B were exactly ±1 IDDSI level
  (19 of 19)"* and *"78.6% of items had at least two of three labellers agreeing"*.
  These appear in **no** artefact in the repo — not in `results/regen_20260916/`, not in
  `results/v05_release/`, and not in the H-LABELCEIL hazard entry. The pilot's 40.0% /
  42.9% agreement figures *are* carried, because H-LABELCEIL states them.

---

## 7. How to apply this to the Hub repo

**Option A — the card is the source of truth (recommended).** `hf/model_README.md` is
already the complete README including front matter. Copy it to the repo root as
`README.md`:

```bash
# from the lane clone
cp hf/model_README.md /path/to/iddsi-open-siglip2-v0/README.md
huggingface-cli upload lingualeap/iddsi-open-siglip2-v0 README.md README.md \
  --repo-type model --commit-message "model card: metadata + discoverability pass"
```

**Option B — set the metadata in the Hub UI.** Repo → *Edit model card* → paste the
front matter block from §1 at the top, keep the body from `hf/model_README.md`. The tag
chips are derived from the YAML `tags:` list, so they need no separate UI step.

**After uploading, check:**

1. Repo page shows the **Image-Classification** pipeline chip and the **transformers**
   framework chip.
2. The base-model relation links to `google/siglip2-base-patch16-224`.
3. All 19 tag chips render.
4. The safety banner renders as the first blockquote, above the fold.
5. The repo appears under the filters: Task → Image Classification; Libraries →
   Transformers; and in tag searches for `iddsi`, `food`, `texture`, `dysphagia`,
   `dietary`.

---

## 8. Verification

```bash
docker exec -w /work/lanes2/rel01_modelcard -e LANE_RUNS=/work/lane_runs/rel01_modelcard \
  iddsi-train python hf/verify_card_claims.py
```

The run exits non-zero if any of the ten search phrases is missing from the card, if any
disclaimer/gate/hazard marker is missing, if a headline range cannot be re-derived from
the artefact it is attributed to, or if the card's front matter drifts from the block in
§1. Outputs:

- `lane_runs/rel01_modelcard/card_claims_verification.json` — per-claim measured vs
  quoted values, with the source artefact named for each.
- `lane_runs/rel01_modelcard/card_claims_verification.md` — the same as a table.
- `lane_runs/rel01_modelcard/search_phrase_audit.json` — phrase → matched surface form
  and line numbers.
- `lane_runs/rel01_modelcard/hazard_coverage_audit.json` — hazard id → referenced lines
  in the card.

Latest run: **OK — 10/10 search phrases present, 17/17 disclaimer markers present,
16/16 hazard entries referenced, 67/67 claim checks passed.**
