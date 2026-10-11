#!/usr/bin/env python3
"""Corpus-hygiene audit for the release model card, measured from the shipped
artefacts only.

Inputs (read-only):
  /work/data/{realweak_pilot,harvest_d1,harvest_d1b}/attribution.jsonl   licence ledgers
  /work/data/{realweak_pilot,harvest_d1,harvest_d1b}/events.jsonl        provenance rows
  /work/data/manifest_v07_weighted.jsonl                                 the v0.7 manifest
  /work/lanes/rl04_modelcard/results/v05_release/unusable_event_ids.txt  labelability sweep output
  /work/lanes/rl04_modelcard/results/v03_groupdisjoint/dedup_report.json perceptual dedup

Release rule applied (harvest/source_licenses.json + harvest/README.md
"Release invariant"): fail closed. A row is release-clean only if its provider
is per-asset or dataset-level verified (wikimedia / nutrition5k), its licence is
in the allowlist and is not a ShareAlike family incompatible with the BY-NC-SA
main release, and its attribution fields are complete.

Writes /work/lane_runs/rl04_modelcard/corpus_hygiene_measured.json.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

DATA = pathlib.Path("/work/data")
OUT = pathlib.Path("/work/lane_runs/rl04_modelcard/corpus_hygiene_measured.json")
SETS = ["realweak_pilot", "harvest_d1", "harvest_d1b"]

# harvest/source_licenses.json "status" field, by provider id.
UNVERIFIED_PROVIDERS = {"open-images", "openverse"}  # discovery_only_pending_per_asset_verification
SA_FAMILY = ("CC BY-SA",)
ALLOWED = (
    "CC0", "CC0 1.0", "CC BY 2.0", "CC BY 2.5", "CC BY 3.0", "CC BY 4.0",
    "CC BY-SA 2.0", "CC BY-SA 2.5", "CC BY-SA 3.0", "CC BY-SA 4.0",
)


def main() -> None:
    ledgers = {}
    for s in SETS:
        rows = [json.loads(l) for l in open(DATA / s / "attribution.jsonl") if l.strip()]
        ledgers[s] = rows

    all_rows = [r for s in SETS for r in ledgers[s]]
    event_ids = [r["event_id"] for r in all_rows]

    report: dict = {
        "harvest_rows_per_set": {s: len(ledgers[s]) for s in SETS},
        "harvest_rows_total": len(all_rows),
        "unique_event_ids": len(set(event_ids)),
        "duplicate_event_id_rows": len(event_ids) - len(set(event_ids)),
        "provider_counts": dict(Counter(r["provider"] for r in all_rows)),
        "licence_counts": dict(Counter(r["license_name"] for r in all_rows)),
    }

    # ---- the three exclusion classes the release builder applies ----
    unverified = [r for r in all_rows if r["provider"] in UNVERIFIED_PROVIDERS]
    sa = [r for r in all_rows if r["license_name"].startswith(SA_FAMILY)]
    sa_not_unverified = [r for r in sa if r["provider"] not in UNVERIFIED_PROVIDERS]
    dataset_level_only = [r for r in all_rows if r["provider"] == "nutrition5k"]

    report["unverified_open_images_openverse_rows"] = len(unverified)
    report["unverified_by_provider"] = dict(Counter(r["provider"] for r in unverified))
    report["sharealike_rows"] = len(sa)
    report["sharealike_rows_not_already_unverified"] = len(sa_not_unverified)
    report["sharealike_by_provider"] = dict(Counter(r["provider"] for r in sa))
    report["dataset_level_grant_rows"] = len(dataset_level_only)

    # ---- what would actually ship under the fail-closed rule ----
    incomplete_attribution = [
        r for r in all_rows
        if not (r.get("creator") and r.get("landing_url") and r.get("license_name") and r.get("license_url"))
    ]
    clean = [
        r for r in all_rows
        if r["provider"] not in UNVERIFIED_PROVIDERS
        and not r["license_name"].startswith(SA_FAMILY)
        and r["license_name"] in ALLOWED
        and r not in incomplete_attribution
    ]
    report["rows_with_incomplete_attribution"] = len(incomplete_attribution)
    report["release_clean_rows"] = len(clean)
    report["release_clean_unique_events"] = len(set(r["event_id"] for r in clean))
    report["release_clean_by_provider"] = dict(Counter(r["provider"] for r in clean))
    report["release_clean_by_licence"] = dict(Counter(r["license_name"] for r in clean))

    # ---- labelability sweep: how much of the corpus can carry a level at all ----
    unusable = [l.strip() for l in open(
        "/work/lanes/rl04_modelcard/results/v05_release/unusable_event_ids.txt") if l.strip()]
    sweep = json.load(open("/work/lanes/rl04_modelcard/results/v05_release/labelability_sweep.json"))
    report["labelability_sweep_raw"] = sweep
    report["unusable_events_listed"] = len(unusable)
    report["unusable_unique"] = len(set(unusable))
    report["unusable_fraction_of_684"] = len(set(unusable)) / 684
    report["cross_family_is_there_food_agreement"] = sweep["cross_food_agree"] / sweep["both"]
    report["cross_family_levellable_agreement"] = sweep["cross_lev_agree"] / sweep["both"]
    report["sweep_events_both_families"] = sweep["both"]

    # ---- perceptual dedup: near-duplicate leakage the id field never saw ----
    dedup = json.load(open("/work/lanes/rl04_modelcard/results/v03_groupdisjoint/dedup_report.json"))
    report["dedup_images_hashed"] = dedup["images_hashed"]
    report["dedup_near_duplicate_pairs"] = dedup["near_duplicate_pairs"]
    report["dedup_exact_byte_duplicate_groups"] = dedup["exact_byte_duplicate_groups"]
    report["dedup_groups_with_more_than_one_member"] = dedup["groups_with_more_than_one_member"]
    report["dedup_groups_spanning_multiple_splits_in_v02"] = dedup["groups_spanning_multiple_splits_in_v02"]
    report["dedup_leaky_groups_sample"] = dedup["leaky_groups_sample"]

    # ---- v0.7 manifest shape ----
    man = [json.loads(l) for l in open(DATA / "manifest_v07_weighted.jsonl") if l.strip()]
    report["v07_manifest_rows"] = len(man)
    report["v07_manifest_unique_events"] = len(set(r["event_id"] for r in man))
    report["v07_manifest_unique_groups"] = len(set(r["group_id"] for r in man))
    report["v07_split_counts"] = dict(Counter(r["split"] for r in man))
    report["v07_level_counts"] = {f"L{k}": v for k, v in sorted(Counter(r["level"] for r in man).items())}
    report["v07_level_fraction"] = {f"L{k}": v / len(man) for k, v in sorted(Counter(r["level"] for r in man).items())}
    report["v07_source_corpus_counts"] = dict(Counter(r["source_corpus"] for r in man))
    report["v07_label_source_counts"] = dict(Counter(r["label_source"] for r in man))
    report["v07_n_labellers_counts"] = dict(Counter(r["n_labellers"] for r in man))
    report["v07_agreement_counts"] = dict(Counter(r["agreement"] for r in man))
    report["v07_events_with_two_judgements"] = sum(1 for r in man if r["n_labellers"] >= 2)
    report["v07_events_unanimous"] = sum(1 for r in man if r["agreement"] == "2/2")
    report["v07_events_split"] = sum(1 for r in man if r["agreement"] == "1/2")
    report["v07_events_single_judgement"] = sum(1 for r in man if r["n_labellers"] == 1)
    report["v07_unanimous_fraction_of_double"] = (
        report["v07_events_unanimous"] / report["v07_events_with_two_judgements"]
    )
    test = [r for r in man if r["split"] == "test"]
    report["v07_test_n"] = len(test)
    report["v07_test_level_counts"] = {f"L{k}": v for k, v in sorted(Counter(r["level"] for r in test).items())}
    report["v07_test_source_corpus_counts"] = dict(Counter(r["source_corpus"] for r in test))
    report["v07_test_n_labellers_counts"] = dict(Counter(r["n_labellers"] for r in test))
    report["v07_train_n"] = sum(1 for r in man if r["split"] == "train")
    report["v07_val_n"] = sum(1 for r in man if r["split"] == "val")

    # ---- is the split actually group-disjoint? measure it, do not assume it ----
    by_group: dict = {}
    for r in man:
        by_group.setdefault(r["group_id"], []).append(r)
    report["v07_groups_spanning_multiple_splits"] = sum(
        1 for rs in by_group.values() if len({x["split"] for x in rs}) > 1
    )
    report["v07_multi_member_groups"] = sum(1 for rs in by_group.values() if len(rs) > 1)
    report["v07_events_in_multiple_splits"] = sum(
        1 for rs in by_group.values() if len({x["split"] for x in rs}) > 1
    )
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        ga = {r["group_id"] for r in man if r["split"] == a}
        gb = {r["group_id"] for r in man if r["split"] == b}
        report[f"v07_group_overlap_{a}_{b}"] = len(ga & gb)

    OUT.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}\n")
    for k in [
        "harvest_rows_total", "unique_event_ids", "duplicate_event_id_rows",
        "unverified_open_images_openverse_rows", "unverified_by_provider",
        "sharealike_rows", "sharealike_rows_not_already_unverified", "sharealike_by_provider",
        "dataset_level_grant_rows", "release_clean_rows", "release_clean_unique_events",
        "release_clean_by_provider", "release_clean_by_licence",
        "unusable_events_listed", "unusable_unique", "unusable_fraction_of_684",
        "cross_family_is_there_food_agreement", "cross_family_levellable_agreement",
        "dedup_near_duplicate_pairs", "dedup_groups_with_more_than_one_member",
        "dedup_groups_spanning_multiple_splits_in_v02",
        "v07_manifest_rows", "v07_manifest_unique_events", "v07_manifest_unique_groups",
        "v07_split_counts", "v07_level_counts", "v07_level_fraction",
        "v07_events_with_two_judgements", "v07_events_unanimous", "v07_events_split",
        "v07_events_single_judgement", "v07_unanimous_fraction_of_double",
        "v07_test_n", "v07_test_level_counts", "v07_test_source_corpus_counts",
        "v07_train_n", "v07_val_n",
    ]:
        print(f"  {k}: {report[k]}")


if __name__ == "__main__":
    main()
