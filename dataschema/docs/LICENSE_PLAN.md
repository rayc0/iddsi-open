# Licensing plan

This is a release plan, not legal advice and not a substitute for confirming that every contributed asset can be distributed under the stated terms.

## Intended licences

| Artifact | Intended licence | Release requirement |
|---|---|---|
| Original dataset records and releasable media | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) | Include the unmodified official licence text/link, attribution instructions, source/provenance, and consent/release confirmation. |
| Schema, validator, tests, capture documents, and other code | Apache License 2.0 | Include `LICENSE` and any required `NOTICE`; retain copyright and notices in redistributed source. |
| Released model weights | Apache License 2.0, where legally permitted | Verify base-model terms, training-data rights, and third-party restrictions before attaching Apache-2.0. Record base model, commit/version, and dependencies in the model card. |

Recommended attribution for the original data (final organization/release URL to be inserted): “IDDSI-Open dataset, LinguaLeap Tech Limited (HK), version [VERSION], [URL], licensed CC BY-NC-SA 4.0.”

## Separation and release checks

1. Keep data/media in a clearly marked data package with `LICENSE-DATA` containing the official CC BY-NC-SA 4.0 text or canonical link. Keep code and eligible weight packages under Apache-2.0 with their own `LICENSE`/`NOTICE`.
2. Do not imply that Apache-2.0 grants rights in the dataset, IDDSI trademarks/materials, third-party media, personal data, or a base model governed by separate terms.
3. Maintain a provenance ledger for every media item. Exclude assets with incompatible, absent, or ambiguous redistribution permission. Label synthetic items `source=synthetic`; do not represent them as physically tested.
4. Confirm that participant consent expressly covers public, noncommercial, share-alike redistribution of the media. Consent records remain private.
5. Review whether trained weights are legally separable from the CC BY-NC-SA dataset in each planned jurisdiction. Until counsel/release review confirms this and base-model terms are compatible, describe Apache-2.0 for weights as intended, not already granted.
6. Preserve the required IDDSI attribution/trademark notices and state that LinguaLeap’s tests are not IDDSI certification or endorsement.
7. Record each release’s dataset version, manifest checksum, exclusions, and exact licence files. Never silently relicense earlier media.

The NonCommercial restriction means the open data is not OSI “open source” in the code sense. Public materials should say “publicly downloadable under an explicit reuse licence” and name CC BY-NC-SA 4.0, rather than implying unrestricted commercial reuse.
