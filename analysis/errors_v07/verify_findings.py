#!/usr/bin/env python
"""Assert every headline number in findings.md against the analysis JSON files.

Run: docker exec -u 1000:1000 iddsi-train python \
       /work/lanes/an02_errors/analysis/errors_v07/verify_findings.py
"""
from __future__ import annotations

import json
from pathlib import Path

A = Path("/work/lane_runs/an02_errors/analysis")
fails = []


def check(label, got, want, tol=0.0005):
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    print(f"{'PASS' if ok else 'FAIL'}  {label}: got={got} want={want}")
    if not ok:
        fails.append(label)


rank = json.loads((A / "seed_ranking.json").read_text())
check("best seed", rank["best_seed"], 43)
tbl = {r["seed"]: r for r in rank["table"]}
check("seed43 macro_f1", tbl[43]["macro_f1_with_abstentions"], 0.3949309489682161)
check("seed17 macro_f1", tbl[17]["macro_f1_with_abstentions"], 0.3594155844155844)
check("seed101 macro_f1", tbl[101]["macro_f1_with_abstentions"], 0.33986013986013985)
check("seed29 macro_f1", tbl[29]["macro_f1_with_abstentions"], 0.33169129720853857)
check("seed71 macro_f1", tbl[71]["macro_f1_with_abstentions"], 0.3193103448275862)
check("all seeds met calibration gate",
      all(r["calibration_target_met"] for r in rank["table"]), True)

repro = json.loads((A / "reproduction_check.json").read_text())
check("all 5 seeds reproduce published confusion matrix",
      all(v["matches_published_confusion_matrix"] for v in repro.values()), True)

cm = json.loads((A / "confusion_seed43.json").read_text())
m6 = cm["matrix_6x6_true_x_pred_with_abstain"]["counts"]
check("6x6 row totals", [sum(r) for r in m6], [10, 11, 7, 10, 70, 0])
check("6x6 col totals", [sum(c) for c in zip(*m6)], [13, 2, 0, 0, 77, 16])
check("L6->L7 cell", m6[3][4], 8)
check("L5->L3 cell", m6[2][0], 3)
check("L4->L3 cell", m6[1][0], 1)
check("L5->L7 cell", m6[2][4], 2)
check("L4->L7 cell", m6[1][4], 1)
check("L3->L7 cell", m6[0][4], 1)
check("L4 abstain", m6[1][5], 7)
check("L7 abstain", m6[4][5], 5)
check("total n", cm["n_test_events"], 108)

adj = json.loads((A / "adjacent_pairs_seed43.json").read_text())
pairs = {p["pair"]: p for p in adj["adjacent_pairs"]}
check("L6-L7 adjacent total", pairs["L6-L7"]["adjacent_total"], 8)
check("L6-L7 rate up per true L6", pairs["L6-L7"]["rate_up_per_true_L6"], 0.8)
check("L3-L4 adjacent total", pairs["L3-L4"]["adjacent_total"], 1)
check("L4-L5 adjacent total", pairs["L4-L5"]["adjacent_total"], 0)
check("L5-L6 adjacent total", pairs["L5-L6"]["adjacent_total"], 0)
check("adjacent slips", adj["adjacent_slip_errors"], 9)
check("wrong level decisions", adj["wrong_level_decisions"], 16)
check("decided", adj["decided_events"], 92)
check("gap2", adj["gap2_errors"], 5)
check("gap3plus", adj["gap3plus_errors"], 2)

b = json.loads((A / f"bias_seed{43}.json").read_text())
check("over", b["over_predicted"], 12)
check("under", b["under_predicted"], 4)
check("correct", b["correct"], 76)
check("abstained", b["abstained"], 16)
check("mean signed error", b["mean_signed_error"], 0.1304)
check("under rate all test", b["under_events_all_test_rate"], 0.037)
check("under fnr of decided", b["under_prediction_fnr_of_decided"], 0.0615)
check("under denominator", b["under_predictable_denominator"], 65)
check("hist +1", b["signed_error_histogram"]["1"], 8)
check("hist +2", b["signed_error_histogram"]["2"], 2)
check("hist +3", b["signed_error_histogram"]["3"], 1)
check("hist +4", b["signed_error_histogram"]["4"], 1)
check("hist -1", b["signed_error_histogram"]["-1"], 1)
check("hist -2", b["signed_error_histogram"]["-2"], 3)
check("L6 signed bias", b["per_level_signed_bias"]["L6"]["mean_signed_error_decided"], 1.0)
check("L5 signed bias", b["per_level_signed_bias"]["L5"]["mean_signed_error_decided"], -0.4)

sig = json.loads((A / "significance_seed43.json").read_text())
check("agreement 1/2 rate", sig["label_agreement"]["worst_rate"], 0.4091)
check("agreement 1/2 fisher p", sig["label_agreement"]["fisher_two_sided_p"], 0.0006)
check("agreement 1/2 n", sig["label_agreement"]["worst_n"], 22)
check("agreement 1/2 wrong", sig["label_agreement"]["worst_wrong_level"], 9)
check("weak_rule L6 rate", sig["harvest_visual_bucket_weak_rule"]["worst_rate"], 0.2759)
check("weak_rule L6 fisher p", sig["harvest_visual_bucket_weak_rule"]["fisher_two_sided_p"], 0.0332)
check("corpus fisher p", sig["source_corpus"]["fisher_two_sided_p"], 0.3607)
check("provider fisher p", sig["provider"]["fisher_two_sided_p"], 0.3525)
check("food family fisher p", sig["food_family_heuristic"]["fisher_two_sided_p"], 0.2265)
check("over/under binom p", sig["over_under_asymmetry"]["binom_sign_test_p"], 0.0768)

st = json.loads((A / "structural_seed43.json").read_text())
check("levels never emitted (seed43)",
      st["emitted_level_census"]["43"]["levels_never_emitted"], ["L5", "L6"])
for s in ("17", "29", "71", "101"):
    check(f"levels never emitted (seed{s})",
          "L6" in st["emitted_level_census"][s]["levels_never_emitted"], True)
check("emitted L3 count", st["emitted_level_census"]["43"]["emitted_level_counts"]["L3"], 13)
check("emitted L4 count", st["emitted_level_census"]["43"]["emitted_level_counts"]["L4"], 2)
check("emitted L7 count", st["emitted_level_census"]["43"]["emitted_level_counts"]["L7"], 77)
check("argmax never tops L6 (seed43)",
      st["emitted_level_census"]["43"]["argmax_levels_never_top"], ["L4", "L6"])
ab = st["abstention_analysis_seed43"]
check("abstain n", ab["abstained_n"], 16)
check("L4 abstain rate", ab["abstention_rate_by_true_level"]["L4"], 0.6364)
check("L3 abstain rate", ab["abstention_rate_by_true_level"]["L3"], 0.0)
check("argmax would-be under", ab["argmax_would_have_been_under"], 7)
h = st["harvest_l6_bucket_seed43"]
check("L6 bucket n", h["n"], 29)
check("L6 bucket actually L6", h["share_actually_labelled_L6"], 0.1379)
check("L6 bucket labelled L7", h["consensus_label_distribution"]["L7"], 18)
check("L6 bucket labelled L5", h["consensus_label_distribution"]["L5"], 6)
bc = st["bootstrap_ci_seed43"]
check("wrong_level_rate point", bc["wrong_level_rate"]["point"], 0.1481)
check("wrong_level_rate lo", bc["wrong_level_rate"]["lo"], 0.0833)
check("wrong_level_rate hi", bc["wrong_level_rate"]["hi"], 0.213)
check("under_rate_all lo", bc["under_rate_all"]["lo"], 0.0093)
check("under_rate_all hi", bc["under_rate_all"]["hi"], 0.0741)
check("L6_to_L7 lo", bc["L6_to_L7_rate"]["lo"], 0.0278)
check("L6_to_L7 hi", bc["L6_to_L7_rate"]["hi"], 0.1296)
hm = st["headline_metric_reproduction_seed43"]
check("macro f1 match", hm["match"], True)
check("kappa match", hm["kappa_match"], True)
check("recall L3", hm["per_level_recall_decided"]["L3"], 0.9)
check("recall L4", hm["per_level_recall_decided"]["L4"], 0.5)
check("recall L5", hm["per_level_recall_decided"]["L5"], 0.0)
check("recall L6", hm["per_level_recall_decided"]["L6"], 0.0)
check("recall L7", hm["per_level_recall_decided"]["L7"], 1.0)
check("precision L3", hm["per_level_precision_decided"]["L3"], 0.6923)
check("precision L4", hm["per_level_precision_decided"]["L4"], 1.0)
check("precision L7", hm["per_level_precision_decided"]["L7"], 0.8442)
check("distinct emitted", hm["distinct_levels_emitted_as_decisions"], ["L3", "L4", "L7"])

g = json.loads((A / "group_errors_seed43.json").read_text())
gg = {t["group"]: t for t in g["food_family_heuristic"]}
check("uninformative_title errors", gg["uninformative_title"]["wrong_level"], 7)
check("uninformative_title share of 16", round(gg["uninformative_title"]["wrong_level"] / 16, 4), 0.4375)
check("starch_dish errors", gg["starch_dish"]["wrong_level"], 2)
check("starch_dish n", gg["starch_dish"]["n"], 12)
check("meat_fish_egg errors", gg["meat_fish_egg"]["wrong_level"], 1)
check("candy_baked_sweet errors", gg["candy_baked_sweet"]["wrong_level"], 1)
cc = {t["group"]: t for t in g["source_corpus"]}
check("harvest_d1 rate", cc["harvest_d1"]["error_rate_wrong_level_of_n"], 0.1176)
check("harvest_d1b rate", cc["harvest_d1b"]["error_rate_wrong_level_of_n"], 0.2069)
check("realweak_pilot rate", cc["realweak_pilot"]["error_rate_wrong_level_of_n"], 0.1818)
pv = {t["group"]: t for t in g["provider"]}
check("open-images rate", pv["open-images"]["error_rate_wrong_level_of_n"], 0.1176)
check("openverse rate", pv["openverse"]["error_rate_wrong_level_of_n"], 0.2143)
check("nutrition5k rate", pv["nutrition5k"]["error_rate_wrong_level_of_n"], 0.1818)
check("wikimedia n", pv["wikimedia"]["n"], 1)
wr = {t["group"]: t for t in g["harvest_visual_bucket_weak_rule"]}
check("L6 bucket wrong", wr["visible_pieces_about_15mm_l6_candidate"]["wrong_level"], 8)
ag = {t["group"]: t for t in g["label_agreement"]}
check("1/2 n", ag["1/2"]["n"], 22)
check("1/2 wrong", ag["1/2"]["wrong_level"], 9)
check("2/2 n", ag["2/2"]["n"], 47)
check("1/1 n", ag["1/1"]["n"], 39)

cs = json.loads((A / "cross_seed_stability.json").read_text())
import collections
for s, v in cs.items():
    top = collections.Counter((t["true"], t["pred"]) for t in v["top_confusions"]).most_common(1)
    check(f"seed{s} top confusion is L6->L7", top[0][0], ("L6", "L7"))
    check(f"seed{s} over > under", v["over"] > v["under"], True)
mse = [v["mean_signed_error"] for v in cs.values()]
check("min cross-seed MSE", round(min(mse), 3), 0.112)
check("max cross-seed MSE", round(max(mse), 3), 0.130)

print()
print(f"{'ALL CHECKS PASSED' if not fails else 'FAILURES: ' + str(fails)}")
raise SystemExit(1 if fails else 0)
