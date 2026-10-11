"""Why does (or doesn't) seed averaging help? Diversity diagnostics.

Seed ensembles only help to the extent the seeds disagree in *useful* ways. The
five v0.7 checkpoints share the pretrained backbone, the manifest, the split and
the whole training schedule; only the seed differs. This script measures how
much genuine diversity that leaves, on the 108 held-out test events:

  * pairwise agreement of the CORAL rank decode (pre-abstention)
  * mean pairwise cosine similarity of the calibrated probability vectors
  * fraction of events where all five seeds decode to the same level
  * hard majority vote vs the probability average
  * abstention overlap between the ensemble and the singles

    docker exec -w /work/lanes/exp08_ensemble iddsi-train python -m exp08.seed_diversity
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from exp08.build_ensemble import SEEDS, decode_metrics, load_seed
from train.model import coral_decode


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "exp08" / "ensemble"
    seeds = {seed: load_seed(repo / "exp08" / "raw", seed, Path("/work/runs/v07")) for seed in SEEDS}
    labels = seeds[SEEDS[0]]["test_labels"]
    val_labels = seeds[SEEDS[0]]["val_labels"]

    decoded = {s: coral_decode(seeds[s]["test_probs"]) for s in SEEDS}
    confidence = {s: decoded[s][1] for s in SEEDS}
    predictions = {s: decoded[s][0] for s in SEEDS}

    pairwise = []
    cosine = []
    for a, b in itertools.combinations(SEEDS, 2):
        pairwise.append(float((predictions[a] == predictions[b]).mean()))
        pa = seeds[a]["test_probs"] / np.linalg.norm(seeds[a]["test_probs"], axis=1, keepdims=True)
        pb = seeds[b]["test_probs"] / np.linalg.norm(seeds[b]["test_probs"], axis=1, keepdims=True)
        cosine.append(float((pa * pb).sum(axis=1).mean()))
    stacked = np.stack([predictions[s] for s in SEEDS])
    unanimous = np.all(stacked == stacked[0], axis=0)

    # per-event spread of the decoded level across seeds
    spread = stacked.max(axis=0) - stacked.min(axis=0)

    # hard majority vote (ties broken toward the lower level, deterministic)
    votes = np.stack([predictions[s] for s in SEEDS])
    majority = np.array(
        [np.bincount(votes[:, i], minlength=5).argmax() for i in range(len(labels))]
    )
    # confidence for the majority vote = mean seed confidence on the voted level
    majority_conf = np.array(
        [
            float(np.mean([confidence[s][i] for s in SEEDS if predictions[s][i] == majority[i]]))
            for i in range(len(labels))
        ]
    )

    ensemble_test = np.mean([seeds[s]["test_probs"] for s in SEEDS], axis=0)
    ensemble_val = np.mean([seeds[s]["val_probs"] for s in SEEDS], axis=0)
    calibration = json.loads((out_dir / "mean_prob_calibration.json").read_text(encoding="utf-8"))
    threshold = float(calibration["threshold"])

    from eval.calibration import calibrate_abstention_threshold  # noqa: E402

    majority_calibration = calibrate_abstention_threshold(
        np.zeros((len(val_labels), 5)) + 0.0, val_labels, 0.05, 0.30
    )  # placeholder, replaced below

    # majority vote needs its own confidence vector to calibrate a threshold on
    fake_probs = np.zeros((len(val_labels), 5))
    # Instead: calibrate the majority-vote rule on validation using the same
    # decode, then evaluate on test.
    val_decoded = {s: coral_decode(seeds[s]["val_probs"]) for s in SEEDS}
    val_votes = np.stack([val_decoded[s][0] for s in SEEDS])
    val_majority = np.array([np.bincount(val_votes[:, i], minlength=5).argmax() for i in range(len(val_labels))])
    val_majority_conf = np.array(
        [
            float(np.mean([val_decoded[s][1][i] for s in SEEDS if val_decoded[s][0][i] == val_majority[i]]))
            for i in range(len(val_labels))
        ]
    )
    from eval.metrics import _point_metrics  # noqa: E402

    candidates = np.unique(np.concatenate(([0.0], val_majority_conf, [np.nextafter(val_majority_conf.max(), np.inf)])))
    best = None
    for candidate in candidates:
        accepted = val_majority_conf >= candidate
        coverage = float(accepted.mean())
        if coverage + 1e-12 < 0.7 or not accepted.any():
            continue
        under_fnr = float((val_majority[accepted] < val_labels[accepted]).mean())
        if under_fnr <= 0.05:
            if best is None or coverage > best[1]:
                best = (float(candidate), coverage)
    majority_threshold = best[0] if best else float(candidates[-1])
    accepted = majority_conf >= majority_threshold
    majority_metrics = {
        "threshold": majority_threshold,
        "coverage": float(accepted.mean()),
        "weighted_kappa_covered": float(
            __import__("sklearn.metrics", fromlist=["cohen_kappa_score"]).cohen_kappa_score(
                labels[accepted], majority[accepted], weights="quadratic"
            )
        ),
        "dangerous_underclassification_fnr_covered": float((majority[accepted] < labels[accepted]).mean()),
    }

    # abstention overlap
    ens_accept = confidence[SEEDS[0]] >= threshold  # same for all seeds (identical conf? no) -> recompute
    ens_pred, ens_conf = coral_decode(ensemble_test)
    ens_accept = ens_conf >= threshold
    single_accept = {s: confidence[s] >= seeds[s]["threshold"] for s in SEEDS}
    all_singles_accept = np.all(np.stack([single_accept[s] for s in SEEDS]), axis=0)
    any_single_accept = np.any(np.stack([single_accept[s] for s in SEEDS]), axis=0)

    report = {
        "denominators": {"n_test": int(len(labels)), "n_val": int(len(val_labels)), "n_seeds": len(SEEDS)},
        "pairwise_decoded_agreement_mean": float(np.mean(pairwise)),
        "pairwise_decoded_agreement_min_max": [float(np.min(pairwise)), float(np.max(pairwise))],
        "pairwise_probability_cosine_mean": float(np.mean(cosine)),
        "events_all_five_seeds_agree": int(unanimous.sum()),
        "events_all_five_seeds_agree_frac": float(unanimous.mean()),
        "decoded_level_spread_hist": np.bincount(spread, minlength=5).tolist(),
        "decoded_level_spread_mean": float(spread.mean()),
        "hard_majority_vote": majority_metrics,
        "abstention_overlap": {
            "ensemble_covers": int(ens_accept.sum()),
            "covered_by_all_five_singles": int((all_singles_accept & ens_accept).sum()),
            "ensemble_covers_but_no_single_does": int((ens_accept & ~any_single_accept).sum()),
            "all_singles_cover_but_ensemble_does_not": int((~ens_accept & all_singles_accept).sum()),
        },
    }
    (out_dir / "seed_diversity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
