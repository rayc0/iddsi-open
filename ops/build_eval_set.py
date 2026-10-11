#!/usr/bin/env python3
"""Build a SEALED physical EVALUATION set — the highest-value 9 hours.

Recommended by an independent review (gpt-6.1-sol) and it changes the shape of the
programme: stop trying to relabel the TRAINING corpus with machines, and instead
physically measure a small set that is used ONLY for evaluation, never for
training. Training on it would spend the only independent evidence we have.

Budget for ~9 h of one operator:
   1.0 h  pilot, 5 specimens (one per level) -- verify timing, equipment, evidence
   4.5 h  60 specimens: 12 base foods x 5 levels, wherever physically feasible
   1.5 h  ~12 CONTRASTS: visually similar preparations that are physically
          different. These probe whether the photograph carries the defining
          information at all.
   1.0 h  ~12 INDEPENDENT REPEATS: same food and level, prepared again. These
          expose preparation and measurement variability, which is the real
          noise floor of the whole dataset.
   1.0 h  cleanup, failures, missing evidence

Why 12 foods x 5 and not all 60: 40 well-documented specimens beat 84 hurried
nominal labels, and the purpose is to find out what the camera can predict before
building a bigger training corpus.

Food selection rule: STRUCTURAL DIVERSITY over nominal coverage. Pick foods whose
water content, particle structure and cohesion differ, so the set spans starches,
fruit, vegetables, protein and cohesive mixtures. Avoid collecting many variations
of essentially the same puree -- that is what the web corpus already over-represents.
"""
import json, pathlib, re, collections

REPO = pathlib.Path(__file__).resolve().parent.parent
CARDS = REPO / "docs/specimen_cards"
OUT = REPO / "capture"

NAME = re.compile(r"^(bf\d{2})_(.+)_L([3-7])$")

# Structural families the evaluation set must span. Derived from the food slug,
# not hand-assigned, so the selection is reproducible.
def family(slug):
    s = slug.lower()
    if any(k in s for k in ("rice", "congee", "noodle", "oat", "porridge", "taro",
                            "yam", "corn", "cereal", "bread", "bun", "dumpling",
                            "wonton", "pasta", "semolina")):
        return "starch"
    if any(k in s for k in ("banana", "mango", "papaya", "apple", "pear", "orange",
                            "persimmon", "melon", "grape", "peach", "plum", "fig")):
        return "fruit"
    if any(k in s for k in ("spinach", "carrot", "potato", "eggplant", "beetroot",
                            "pumpkin", "tofu", "zucchini", "okra", "cabbage",
                            "pea", "bean", "corn")):
        return "vegetable_or_legume"
    if any(k in s for k in ("chicken", "pork", "beef", "fish", "shrimp", "egg",
                            "meat", "steak", "brisket", "milk", "yogurt")):
        return "protein_or_dairy"
    if any(k in s for k in ("soup", "stew", "sauce", "soup", "congee", "risotto",
                            "custard", "sweet_and_sour")):
        return "cohesive_mixture"
    return "other"


foods = collections.defaultdict(dict)
for p in sorted(CARDS.glob("*.md")):
    m = NAME.match(p.stem)
    if m:
        foods[m.group(1)][int(m.group(3))] = {"slug": m.group(2),
                                               "family": family(m.group(2))}
print(f"base foods with cards: {len(foods)}")

fams = collections.defaultdict(list)
for bf, lv in foods.items():
    fams[lv[min(lv)]["family"]].append(bf)
print("\nfoods per structural family:")
for f, v in sorted(fams.items()):
    print(f"  {f:<22} {len(v):>3}  {', '.join(sorted(v)[:8])}")

# pick 12 foods spanning 5 families, preferring foods with all 5 levels available
full = {bf: lv for bf, lv in foods.items() if set(lv) >= {3, 4, 5, 6, 7}}
print(f"\nfoods with all 5 levels present: {len(full)}")

fam_of = {bf: lv[min(lv)]["family"] for bf, lv in full.items()}
by_fam = collections.defaultdict(list)
for bf, f in fam_of.items():
    by_fam[f].append(bf)

TARGET = {"starch": 3, "fruit": 2, "vegetable_or_legume": 3,
          "protein_or_dairy": 3, "cohesive_mixture": 1}
chosen = []
for f, n in TARGET.items():
    pool = sorted(by_fam.get(f, []))
    chosen += pool[:n]
chosen = sorted(set(chosen))[:12]
if len(chosen) < 12:
    for bf in sorted(full):
        if bf not in chosen:
            chosen.append(bf)
        if len(chosen) >= 12:
            break
chosen = sorted(chosen)
print(f"\nSELECTED 12 base foods for the sealed evaluation set:")
for bf in chosen:
    print(f"  {bf}  {fam_of[bf]:<22} {full[bf][3]['slug']}")

specimens = []
for bf in chosen:
    for lvl in (3, 4, 5, 6, 7):
        sid = f"{bf}_{full[bf][lvl]['slug']}_L{lvl}"
        if sid in {p.stem for p in CARDS.glob("*.md")}:
            specimens.append({"specimen_id": sid, "base_food_id": bf,
                              "level": lvl, "family": fam_of[bf],
                              "role": "EVALUATION_SEALED"})
print(f"\nspecimens in the sealed set: {len(specimens)}")
print("  by level:", dict(sorted(collections.Counter(s['level'] for s in specimens).items())))

# operator must be BLIND to model predictions on these
plan = {
    "name": "SEALED-EVAL-SET-v1",
    "created": "2026-10-10",
    "purpose": "Physical measurement used ONLY to evaluate. Never train on it.",
    "operator_blind_to_model_predictions": True,
    "label_from_observed_tests_not_intended_level": True,
    "retain_failures": True,
    "n_base_foods": len(chosen),
    "specimens": specimens,
    "contrasts": {
        "n_target": 12,
        "definition": "visually similar preparations that are physically different; "
                      "probes whether the photograph carries the defining information",
        "status": "TO BE SPECIFIED",
    },
    "repeats": {
        "n_target": 12,
        "definition": "same food and level, independently prepared again; exposes "
                      "preparation and measurement variability",
        "status": "TO BE SPECIFIED",
    },
    "budget_hours": {"pilot": 1.0, "main": 4.5, "contrasts": 1.5,
                     "repeats": 1.0, "cleanup": 1.0, "total": 9.0},
    "recording_rules": [
        "photograph the untouched serving BEFORE destructive tests",
        "record applicable test results, temperature, timing, particle measurements",
        "record short test video for pressure, tilt and flow",
        "keep recipe identifiers, labels and apparatus out of classifier images",
        "an intended L5 preparation that tests as L4 is an L4 result",
    ],
    "evaluation_plan": [
        "old-label student", "consensus-label student",
        "each labeller individually",
        "majority-class baseline", "food-identity baseline",
        "report exact accuracy, per-class recall, MAE, full confusion matrix, "
        "quadratic kappa, with paired uncertainty clustered by base food",
        "note: 5 training seeds measure training variability, they do NOT multiply "
        "the number of independently tested foods",
    ],
}
OUT.mkdir(exist_ok=True)
(OUT / "sealed_eval_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
print(f"\nwrote {OUT/'sealed_eval_plan.json'}")

# Pilot must span FIVE DIFFERENT FOODS, one per level. Taking all five levels of
# a single food would test the equipment once and tell us nothing about timing and
# handling across food structures, which is the whole point of a 1-hour pilot.
pilot = []
for lvl in (3, 4, 5, 6, 7):
    cand = [s for s in specimens if s["level"] == lvl]
    if cand:
        pilot.append(cand[lvl % len(cand)])          # different food per level
print(f"pilot (one per level, DIFFERENT foods, {len(pilot)} specimens):")
for s in pilot:
    print(f"  {s['specimen_id']:<44} {s['family']}")
(OUT / "sealed_eval_pilot.json").write_text(json.dumps(pilot, indent=2))
print("  (pilot deliberately spans different foods so timing/handling is measured "
      "across structures, not once on a single starch)")
