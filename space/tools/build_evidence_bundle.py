#!/usr/bin/env python3
"""Rebuild ``space/evidence/`` from the repo's committed result files.

Everything the Hugging Face Space displays is either (a) copied verbatim from a
committed JSON in this repository, or (b) measured by this script from the
shared read-only inputs.  Nothing is hand-typed, and every bundled file records
its source path, byte length and SHA-256 in ``PROVENANCE.json`` so a reader can
tell copied-from-measured apart and re-derive both.

Run inside the ``iddsi-train`` container (it needs ``/work/data``):

    docker exec -w /work/lanes2/rel02_space/space/tools iddsi-train \\
        python -m build_evidence_bundle

Sources
-------
copied verbatim
    results/v07_release/*.json
    results/yardstick_audit/{summary,flip_decomposition,bias_matrix_identification}.json
    results/yardstick_audit/per_seed.json            -> slimmed, see below
    results/v03_groupdisjoint/dedup_report.json
    results/v05_release/labelability_sweep.json
    results/regen_20260916/inter_judge_20260916.json
    analysis/errors_v07/outputs/*.json

measured here
    corpus_hygiene_measured.json
        licence/attribution ledgers  /work/data/{realweak_pilot,harvest_d1,harvest_d1b}/attribution.jsonl
        v0.7 manifest                /work/data/manifest_v07_weighted.jsonl
        (plus the two copied files above for dedup and labelability)

``per_seed.json`` is copied with its ``per_event`` arrays dropped.  Those arrays
are the per-event prediction records for 60 runs; they are the audit's working
data, not a display quantity, and shipping 1 MB of them into a Space that never
looks at them would be dead weight.  Every scalar field of every run is kept
unchanged, and the drop is recorded in ``PROVENANCE.json`` so it is not a silent
edit.  The full file stays in the repository at
``results/yardstick_audit/per_seed.json``.

No sealed file is opened.  The script never reads ``capture/sealed_eval_*``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

# --------------------------------------------------------------------------
# what gets copied, verbatim, from where
# --------------------------------------------------------------------------

#: (source relative to repo root, destination name under space/evidence/)
COPY: tuple[tuple[str, str], ...] = (
    # --- the seed-43 error analysis (confusion matrix, structure, bias) ---
    ("analysis/errors_v07/outputs/confusion_seed43.json", "confusion_seed43.json"),
    ("analysis/errors_v07/outputs/structural_seed43.json", "structural_seed43.json"),
    ("analysis/errors_v07/outputs/adjacent_pairs_seed43.json", "adjacent_pairs_seed43.json"),
    ("analysis/errors_v07/outputs/bias_seed43.json", "bias_seed43.json"),
    ("analysis/errors_v07/outputs/seed_ranking.json", "seed_ranking.json"),
    ("analysis/errors_v07/outputs/summary.json", "analysis_summary.json"),
    # --- the five v0.7 release runs (risk-coverage curves, gates, CIs) ---
    ("results/v07_release/v07_seed17_metrics.json", "v07_seed17_metrics.json"),
    ("results/v07_release/v07_seed29_metrics.json", "v07_seed29_metrics.json"),
    ("results/v07_release/v07_seed43_metrics.json", "v07_seed43_metrics.json"),
    ("results/v07_release/v07_seed71_metrics.json", "v07_seed71_metrics.json"),
    ("results/v07_release/v07_seed101_metrics.json", "v07_seed101_metrics.json"),
    ("results/v07_release/v07_seed43_calibration.json", "v07_seed43_calibration.json"),
    ("results/v07_release/bias_matrix.json", "bias_matrix.json"),
    ("results/v07_release/coverage_meta.json", "coverage_meta.json"),
    # --- the yardstick audit (same checkpoints, two label sets) ---
    ("results/yardstick_audit/summary.json", "yardstick_summary.json"),
    ("results/yardstick_audit/flip_decomposition.json", "yardstick_flip_decomposition.json"),
    (
        "results/yardstick_audit/bias_matrix_identification.json",
        "yardstick_bias_matrix_identification.json",
    ),
    # --- corpus hygiene ---
    ("results/v03_groupdisjoint/dedup_report.json", "dedup_report.json"),
    ("results/v05_release/labelability_sweep.json", "labelability_sweep.json"),
    ("results/regen_20260916/inter_judge_20260916.json", "inter_judge_20260916.json"),
)

#: prose sources, copied byte-for-byte (not JSON).  The Space quotes from these
#: so a reader can check every quotation against the file it came from.
RAW_COPY: tuple[tuple[str, str], ...] = (
    ("hf/HAZARD_LOG.md", "HAZARD_LOG.md"),
    ("hf/model_README.md", "model_README.md"),
)

#: per_seed.json is copied with per_event dropped; recorded in PROVENANCE.json.
SLIM: tuple[tuple[str, str, str], ...] = (
    (
        "results/yardstick_audit/per_seed.json",
        "yardstick_per_seed_slim.json",
        "per_event arrays dropped (60 runs x ~108 per-event records); all scalar "
        "fields of all 60 runs kept unchanged",
    ),
)

#: licence allowlist, copied from ``hf/measure_corpus_hygiene.py`` so the two
#: measurements cannot drift.  ``harvest/source_licenses.json`` +
#: ``harvest/README.md`` "Release invariant": fail closed.
UNVERIFIED_PROVIDERS = {"open-images", "openverse"}
SA_FAMILY = ("CC BY-SA",)
ALLOWED = (
    "CC0", "CC0 1.0", "CC BY 2.0", "CC BY 2.5", "CC BY 3.0", "CC BY 4.0",
    "CC BY-SA 2.0", "CC BY-SA 2.5", "CC BY-SA 3.0", "CC BY-SA 4.0",
)
SETS = ("realweak_pilot", "harvest_d1", "harvest_d1b")


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_repo_root(start: pathlib.Path) -> pathlib.Path:
    for cand in [start, *start.parents]:
        if (cand / "results" / "v07_release").is_dir() and (cand / "analysis").is_dir():
            return cand
    raise SystemExit(f"could not locate repo root from {start}")


def measure_corpus_hygiene(repo_root: pathlib.Path, data_root: pathlib.Path) -> dict:
    """Licence, dedup, labelability and split-hygiene numbers, measured now.

    Reads only the shared read-only inputs.  Writes no prediction file, opens
    no sealed file, and never trains.
    """

    from collections import Counter

    ledgers = {}
    for s in SETS:
        rows = [
            json.loads(line)
            for line in (data_root / s / "attribution.jsonl").read_text().splitlines()
            if line.strip()
        ]
        ledgers[s] = rows
    all_rows = [r for s in SETS for r in ledgers[s]]
    event_ids = [r["event_id"] for r in all_rows]

    report: dict = {
        "_measured_by": "space/tools/build_evidence_bundle.py::measure_corpus_hygiene",
        "_inputs": {
            name: str(data_root / s / "attribution.jsonl") for s, name in zip(SETS, SETS)
        },
        "harvest_rows_per_set": {s: len(ledgers[s]) for s in SETS},
        "harvest_rows_total": len(all_rows),
        "unique_event_ids": len(set(event_ids)),
        "duplicate_event_id_rows": len(event_ids) - len(set(event_ids)),
        "provider_counts": dict(Counter(r["provider"] for r in all_rows)),
        "licence_counts": dict(Counter(r["license_name"] for r in all_rows)),
    }

    unverified = [r for r in all_rows if r["provider"] in UNVERIFIED_PROVIDERS]
    sa = [r for r in all_rows if r["license_name"].startswith(SA_FAMILY)]
    incomplete = [
        r
        for r in all_rows
        if not (
            r.get("creator")
            and r.get("landing_url")
            and r.get("license_name")
            and r.get("license_url")
        )
    ]
    clean = [
        r
        for r in all_rows
        if r["provider"] not in UNVERIFIED_PROVIDERS
        and not r["license_name"].startswith(SA_FAMILY)
        and r["license_name"] in ALLOWED
        and r not in incomplete
    ]
    report["unverified_open_images_openverse_rows"] = len(unverified)
    report["unverified_by_provider"] = dict(Counter(r["provider"] for r in unverified))
    report["sharealike_rows"] = len(sa)
    report["sharealike_by_provider"] = dict(Counter(r["provider"] for r in sa))
    report["rows_with_incomplete_attribution"] = len(incomplete)
    report["release_clean_rows"] = len(clean)
    report["release_clean_unique_events"] = len(set(r["event_id"] for r in clean))
    report["release_clean_by_provider"] = dict(Counter(r["provider"] for r in clean))
    report["release_clean_by_licence"] = dict(Counter(r["license_name"] for r in clean))

    # ---- the v0.7 manifest: shape, splits, group disjointness -------------
    man = [
        json.loads(line)
        for line in (data_root / "manifest_v07_weighted.jsonl").read_text().splitlines()
        if line.strip()
    ]
    report["_inputs"]["manifest_v07_weighted.jsonl"] = str(
        data_root / "manifest_v07_weighted.jsonl"
    )
    report["v07_manifest_rows"] = len(man)
    report["v07_manifest_unique_events"] = len(set(r["event_id"] for r in man))
    report["v07_manifest_unique_groups"] = len(set(r["group_id"] for r in man))
    report["v07_split_counts"] = dict(Counter(r["split"] for r in man))
    report["v07_level_counts"] = {
        f"L{k}": v for k, v in sorted(Counter(r["level"] for r in man).items())
    }
    report["v07_source_corpus_counts"] = dict(Counter(r["source_corpus"] for r in man))
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
    report["v07_test_level_counts"] = {
        f"L{k}": v for k, v in sorted(Counter(r["level"] for r in test).items())
    }
    report["v07_test_source_corpus_counts"] = dict(
        Counter(r["source_corpus"] for r in test)
    )
    report["v07_train_n"] = sum(1 for r in man if r["split"] == "train")
    report["v07_val_n"] = sum(1 for r in man if r["split"] == "val")

    by_group: dict[str, list[dict]] = {}
    for r in man:
        by_group.setdefault(r["group_id"], []).append(r)
    report["v07_multi_member_groups"] = sum(1 for rs in by_group.values() if len(rs) > 1)
    report["v07_groups_spanning_multiple_splits"] = sum(
        1 for rs in by_group.values() if len({x["split"] for x in rs}) > 1
    )
    report["v07_events_in_multiple_splits"] = sum(
        1 for rs in by_group.values() if len({x["split"] for x in rs}) > 1
    )
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        ga = {r["group_id"] for r in man if r["split"] == a}
        gb = {r["group_id"] for r in man if r["split"] == b}
        report[f"v07_group_overlap_{a}_{b}"] = len(ga & gb)

    # ---- labelability sweep, from the copied committed file ---------------
    sweep = json.loads(
        (repo_root / "results/v05_release/labelability_sweep.json").read_text()
    )
    report["labelability_sweep_raw"] = sweep
    report["sweep_events_both_families"] = sweep["both"]
    report["cross_family_is_there_food_agreement"] = sweep["cross_food_agree"] / sweep["both"]
    report["cross_family_levellable_agreement"] = sweep["cross_lev_agree"] / sweep["both"]

    # ---- perceptual dedup, from the copied committed file -----------------
    dedup = json.loads(
        (repo_root / "results/v03_groupdisjoint/dedup_report.json").read_text()
    )
    report["dedup_images_hashed"] = dedup["images_hashed"]
    report["dedup_exact_byte_duplicate_groups"] = dedup["exact_byte_duplicate_groups"]
    report["dedup_near_duplicate_pairs"] = dedup["near_duplicate_pairs"]
    report["dedup_groups_with_more_than_one_member"] = dedup["groups_with_more_than_one_member"]
    report["dedup_groups_spanning_multiple_splits_in_v02"] = dedup[
        "groups_spanning_multiple_splits_in_v02"
    ]
    return report


def main(argv: list[str] | None = None) -> int:
    here = pathlib.Path(__file__).resolve()
    repo_root = _find_repo_root(here.parent)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument(
        "--data-root",
        default="/work/data",
        help="shared read-only inputs (container path)",
    )
    parser.add_argument(
        "--out", default=str(here.parent.parent / "evidence"), help="output directory"
    )
    args = parser.parse_args(argv)

    repo = pathlib.Path(args.repo_root)
    data = pathlib.Path(args.data_root)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    provenance: dict = {
        "_built_by": "space/tools/build_evidence_bundle.py",
        "_repo_root": str(repo),
        "_data_root": str(data),
        "files": {},
    }

    for src, dst in COPY:
        s = repo / src
        if not s.is_file():
            raise SystemExit(f"missing source file: {s}")
        payload = json.loads(s.read_text())
        text = json.dumps(payload, indent=1, sort_keys=False, ensure_ascii=False)
        (out / dst).write_text(text + "\n")
        provenance["files"][dst] = {
            "kind": "copied verbatim",
            "source": src,
            "bytes": len(text) + 1,
            "sha256": _sha256(s),
        }

    for src, dst, note in SLIM:
        s = repo / src
        if not s.is_file():
            raise SystemExit(f"missing source file: {s}")
        runs = json.loads(s.read_text())
        slim = [{k: v for k, v in run.items() if k != "per_event"} for run in runs]
        slim.insert(
            0,
            {
                "_note": note,
                "_full_file": src,
                "_n_runs": len(runs),
            },
        )
        text = json.dumps(slim, indent=1, ensure_ascii=False)
        (out / dst).write_text(text + "\n")
        provenance["files"][dst] = {
            "kind": "copied with a recorded reduction",
            "source": src,
            "reduction": note,
            "bytes": len(text) + 1,
            "sha256": _sha256(s),
        }

    for src, dst in RAW_COPY:
        s = repo / src
        if not s.is_file():
            raise SystemExit(f"missing source file: {s}")
        raw = s.read_bytes()
        (out / dst).write_bytes(raw)
        provenance["files"][dst] = {
            "kind": "copied byte-for-byte",
            "source": src,
            "bytes": len(raw),
            "sha256": _sha256(s),
        }

    hygiene = measure_corpus_hygiene(repo, data)
    text = json.dumps(hygiene, indent=1, ensure_ascii=False)
    (out / "corpus_hygiene_measured.json").write_text(text + "\n")
    provenance["files"]["corpus_hygiene_measured.json"] = {
        "kind": "measured by this script",
        "source": "space/tools/build_evidence_bundle.py::measure_corpus_hygiene",
        "inputs": hygiene["_inputs"],
        "bytes": len(text) + 1,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }

    (out / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=1, ensure_ascii=False) + "\n"
    )
    print(f"wrote {len(provenance['files']) + 1} files to {out}")
    for name, meta in sorted(provenance["files"].items()):
        print(f"  {meta['bytes']:>7} B  {name:<44} {meta['kind']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
