# Growing genuine downloads — analysis and plan

Written 2026-10-12. Every claim below was verified this session against the live
repos; nothing here is estimated from memory.

## 1. What a "download" even is for this project

The Hub repo `lingualeap/iddsi-open-siglip2-v0` does **not** ship model weights.
The release shape was locked on 2026-09-07 (code + eval results + flow-test
grader + the measured negative finding) and `ops/stage_release.sh` hard-refuses to
stage a `*.pt` / `*.safetensors` / `*.bin` file. So every download of this repo is
a download of the card, the code, and the eval evidence — i.e. people downloading
*the finding*, not a classifier.

That is not a weakness to work around. It changes the strategy: the thing worth
promoting is the **negative result and the harness**, which is exactly what a
researcher forks and a journalist cites.

## 2. Verified current state (2026-10-12)

| Asset | State |
|---|---|
| Hub repo `lingualeap/iddsi-open-siglip2-v0` | public, tags v0.2–v0.7 |
| Hub model card front matter | `language: en`, `pipeline_tag: image-classification`, tags `[iddsi, food-texture, research-preview, siglip2]`, `license: apache-2.0` |
| GitHub `rayc0/iddsi-open` | public, **0 stars, 0 forks, no description, no topics** |
| GitHub `rayc0/dysphagia-knowledge-hub` (softmeal.org) | public, 1,000+ articles, 11 languages, `llms.txt`, CC BY 4.0, explicitly built to be AI/RAG-readable — **contains no reference to this model** |
| `paper/` | exists (`paper.md`, `paper.pdf`, `references.bib`) |
| Weights | withheld by locked decision (dangerous under-classification 33%) |

## 3. Ranked levers, highest value first

### L1 — Cross-link from softmeal.org (biggest, cheapest, already ours)
The dysphagia hub is the only asset here that already receives search traffic in
11 languages, and it is deliberately structured for AI answer engines. Its IDDSI
framework pages are the highest-intent pages on the whole topic. A
correctly-framed link from those pages to this research preview is the single
largest available source of genuine, qualified traffic.

Framing rule (non-negotiable, and consistent with the withholding decision): it
must read as *research about how well a model can do this*, never as *a tool you
can use*. Anything that invites a caregiver to trust a photo classifier is both a
safety problem and, on a patient-facing site, a reputational one.

### L2 — Model-card search surface on the Hub
The card is thin on the terms people actually search. The front matter should
carry the task, the base model, the domain and the languages; the body should
name the real search phrases (IDDSI level, food texture classification,
texture-modified diet, thickened liquids, dysphagia diet, siglip2). Hub search
indexes tags and card text — this is the cheapest lever on the list and it is
entirely within our control.

### L3 — GitHub repo metadata (done this session)
Topics and description were empty; both are indexed by Google and by GitHub
search. Set 2026-10-12: description + 10 topics including `huggingface`,
`image-classification`, `iddsi`, `dysphagia`, `food-texture`, `siglip2`,
`ordinal-classification`, `food-science`.

### L4 — Citation loop: paper ↔ Hub ↔ GitHub
`paper/` already exists. Publishing it (arXiv) and linking it from the Hub card
and the GitHub README creates the loop that Papers with Code and Google Scholar
both crawl. For a negative-result paper the citation is the distribution
mechanism — nobody downloads a repo they cannot cite.

### L5 — A Hub Space that shows the *evidence*, not a classifier
With weights withheld, a "classify your food" demo would be dishonest. A Space
that renders the confusion matrix, the risk–coverage curve and the yardstick
finding is honest, is genuinely interesting to the audience that matters, and
gives the Hub a discovery surface that links back to the repo.

### L6 — Real self-use (the honest reading of "make more download ourselves")
The safe way to generate our own downloads is to *use the model in real
workloads*: the capture/eval pipeline, a CI smoke test, the explanation CLI. Each
real run pulls from the Hub. That is genuine use and it is also the only version
of self-generated downloads that survives scrutiny.

**Explicitly out of scope:** scripting repeated pulls to inflate the counter,
buying engagement, or any representation of the model as a safety tool. Download
counts that do not correspond to real use are fraud against the Hub's terms and
against the project's own governance.

## 4. What this session already shipped

- `.github/workflows/publish-hf.yml` — publishes the release from a GitHub
  Actions runner. The Mac and the DGX are both SNI-blocked from
  `huggingface.co` (TCP connects, TLS resets), so the runner is the egress and
  **no VPN is needed on either machine**. Verified: run
  [`38159141750`](https://github.com/rayc0/iddsi-open/actions/runs/38159141750)
  succeeded, 510 files staged, 0 weights/media, published.
- GitHub description + 10 topics on `rayc0/iddsi-open`.
- This plan.

## 5. Next actions, in order

1. L1: add a research-preview section to the dysphagia hub's IDDSI pages (needs
   the site's editorial framing — highest value, do first).
2. L2: rewrite the Hub card front matter and search surface (lane `hf01`).
3. L4: decide whether `paper/` goes to arXiv, then wire the three-way links.
4. L5: build the evidence Space (lane `hf02`).
5. L6: land the consume-from-Hub paths (lane `hf03`).
