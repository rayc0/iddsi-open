"""Per-level visual-signature checklists and negative signatures for the VLM judge.

This module sharpens the judge prompt with two structured artefacts per IDDSI
level (3-7):

1. a **visual-signature checklist** - visible, prompt-oriented phrasings of
   the official IDDSI descriptor wording (smooth/lump-free, holds its own
   shape, 4 mm minced particles, 15 mm bite-sized pieces, ...), and
2. a **negative signature** - things that must NOT appear in the image.

The negative signatures encode the audited failure modes of the 0/24 pilot
(2026-09-01 Spark run, judged by ``Qwen/Qwen3-VL-2B-Instruct``): grain
texture ignored (whole rice grains rendered inside L3/L4 masses), garnish
added on top (sesame seeds, scallions, herbs), separated liquid pooling
around the mass, smartphones/devices rendered literally in or on the food,
and camera-facet nouns from the generation prompt (e.g. ``phone-camera
style``, ``noise``) rendered as physical objects.

Design notes:

- Checklist items are *visual* only.  No item claims an image can establish
  rheology, flow rate, fork pressure, or chewing effort - the judge prompt
  already states that photographs cannot prove a physical test.
- Wording is aligned with ``prompt_bank.LEVEL_DESCRIPTORS`` (the compact
  prompt-oriented paraphrase of the official IDDSI level descriptions) and
  with the descriptor-phrase coverage matrix in ``coverage.py``.
- Everything is pure data plus deterministic functions: no network, no model
  imports, no randomness.

``judge.py`` is intentionally NOT edited (fleet board decision).  It can be
wired up later by replacing its local ``_judge_prompt(spec)`` with::

    from .judge_prompts import build_judge_prompt   # noqa: E402

    ... text=build_judge_prompt(spec) ...

The returned prompt keeps the exact JSON response contract of the current
judge, so ``decision_from_response`` parses it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from .prompt_bank import LEVEL_DESCRIPTORS, PromptSpec


LEVELS: tuple[int, ...] = (3, 4, 5, 6, 7)

# Failure modes confirmed by the 0/24 pilot audit.  They apply to every
# level and are merged into each level's negative signature.
COMMON_NEGATIVE_SIGNATURES: tuple[str, ...] = (
    "no garnish of any kind on or around the food - no sesame seeds, scallion, "
    "spring onion, coriander, parsley, herbs, or crispy toppings sprinkled on top",
    "no smartphone, phone, handset, camera, or other device rendered inside, on, "
    "stuck into, or lying partially buried in the food",
    "no literal rendering of camera-facet nouns - the words describing the photo "
    "style (phone-camera style, snapshot, sensor noise, sharpening) must not "
    "appear as physical objects in the scene",
    "no visible text, letters, labels, logos, level numbers, or IDDSI marks",
    "no faces or people; no medical or clinical setting",
)

COMMON_NEGATIVE_SIGNATURES_ZH: tuple[str, ...] = (
    "食物上及周圍不得出現任何裝飾配料——芝麻、蔥花、芫荽、香草或脆脆配料",
    "不得出現智能手機、電話、相機或任何電子裝置出現在食物之內、之上或插入食物中",
    "不得把描述拍攝風格的詞語（手機拍照風格、雜訊）實體化為畫面中的物件",
    "不得出現文字、標籤、標誌、級別數字或 IDDSI 標記",
    "不得出現人臉或人物；不得出現醫療場景",
)


@dataclass(frozen=True)
class LevelSignature:
    """Per-level visual signature checklist plus negative signature."""

    level: int
    name: str
    name_zh: str
    #: Visible checklist items grounded in the official IDDSI descriptor
    #: wording (EN; used by the judge prompt).
    checklist: tuple[str, ...]
    #: Traditional-Chinese checklist items for the app UI.
    checklist_zh: tuple[str, ...]
    #: Level-specific things that must NOT appear (EN).
    negative_signatures: tuple[str, ...]
    #: Level-specific things that must NOT appear (繁中).
    negative_signatures_zh: tuple[str, ...]


LEVEL_SIGNATURES: dict[int, LevelSignature] = {
    3: LevelSignature(
        level=3,
        name="Liquidised",
        name_zh="液狀／中度稠",
        checklist=(
            "completely smooth, lump-free and homogeneous - a liquidised consistency with no visible pieces",
            "no chunks, fibres, skins, seeds, husks or crusts anywhere in the mass",
            "visibly thick and cohesive: it flows slowly and coats rather than splashes - not watery",
            "if a spoon is present it shows a visible coating of the liquidised food, not a clean run-off",
        ),
        checklist_zh=(
            "完全幼滑、無顆粒、質地均勻的液狀食物，看不見任何食物塊粒",
            "沒有塊狀物、纖維、外皮、種籽、殼或結皮",
            "明顯有一定稠度：流動緩慢、能掛在匙羹上，而非水狀濺開",
            "如有匙羹，匙面上留有一層食物塗層而非完全流走",
        ),
        negative_signatures=(
            "no whole grains - no intact rice grains, rice grain texture, quinoa, "
            "or other visible grain kernels in or on the liquidised mass",
            "no separated thin liquid pooling around or bleeding out from the mass",
            "no lumps, curdled clumps, or partially blended pieces",
        ),
        negative_signatures_zh=(
            "不得出現完整米粒、穀物顆粒或其他穀粒紋理混在液狀食物中",
            "不得出現食物周圍滲出或積聚的分離稀液",
            "不得出現結塊、凝固小塊或未完全攪拌的粒塊",
        ),
    ),
    4: LevelSignature(
        level=4,
        name="Pureed",
        name_zh="糊狀／極度稠",
        checklist=(
            "completely smooth, lump-free and homogeneous pureed consistency",
            "thick and cohesive: holds its own shape - it sits as a soft mound on the "
            "plate or spoon instead of spreading out flat",
            "not pourable and not drinkable-looking: when a tilted spoon is visible, the "
            "puree falls off as a single soft mass, not a pouring stream",
            "no crust, skin, fibres, seeds or bits on the surface",
        ),
        checklist_zh=(
            "完全幼滑、無顆粒、質地均勻的糊狀食物",
            "質地濃稠、能保持形狀：在碟或匙上呈柔軟小丘狀，不會攤平散開",
            "不可呈可傾倒或可飲用狀：如畫面有傾斜的匙羹，糊狀食物應整團掉落而非呈流線倒出",
            "表面沒有結皮、薄膜、纖維、種籽或碎屑",
        ),
        negative_signatures=(
            "no whole grains - no intact rice grains, grain-kernel texture, or seeds "
            "rendered in or on the puree",
            "no separated liquid pooling in a ring or puddle around the puree mound",
            "no visible lumps or unblended pieces in the puree",
        ),
        negative_signatures_zh=(
            "不得出現完整米粒、穀粒紋理或種籽混在糊狀食物中",
            "不得在糊狀食物周圍出現環狀或積聚的分離液體",
            "純糊中不得出現可見顆粒或未攪拌的塊粒",
        ),
    ),
    5: LevelSignature(
        level=5,
        name="Minced & Moist",
        name_zh="碎粒及濕軟",
        checklist=(
            "soft, moist minced food - individual minced particles are visible and no "
            "larger than about 4 mm",
            "the minced particles are moist and lightly bound together into a cohesive "
            "mass that can be scooped with a fork or spoon",
            "the food looks mashable - a fork visibly pressed into it would squash the "
            "particles, no biting or chewing needed to break them down",
            "no hard pieces, stringy fibres, skins, bones, husks or crusts among the particles",
        ),
        checklist_zh=(
            "濕軟的碎粒食物：可見的碎粒不大於約 4 毫米",
            "碎粒濕潤並輕微黏合，可用叉或匙整體舀起",
            "食物看來可用叉壓爛——碎粒無需咬嚼即可壓散",
            "碎粒中沒有硬塊、粗纖維、外皮、骨頭、殼或脆皮",
        ),
        negative_signatures=(
            "no whole grains or intact rice grains left among the minced particles; the "
            "mince must not look like ordinary loose rice",
            "no particles visibly larger than about 4 mm - no large chunks or unminced pieces",
            "no separated thin liquid pooling around the minced mass",
        ),
        negative_signatures_zh=(
            "碎粒中不得殘留完整穀粒或米粒；不得看似一般鬆散的白飯",
            "不得出現明顯大於約 4 毫米的顆粒、大塊或未切碎的食物",
            "碎粒食物周圍不得出現積聚的分離稀液",
        ),
    ),
    6: LevelSignature(
        level=6,
        name="Soft & Bite-Sized",
        name_zh="軟質及一口大小",
        checklist=(
            "soft, tender, moist pieces of food cut into adult bite-sized portions no "
            "larger than about 1.5 cm (15 mm)",
            "the pieces look easy to break apart - soft enough to be mashed with the "
            "pressure of a fork, spoon or chopstick",
            "moist throughout: pieces glisten or look sauce-coated, not dry, crumbly or tough",
            "no hard, chewy, crispy, fibrous, stringy components; no bones, seeds, skins or husks",
        ),
        checklist_zh=(
            "軟腍、濕潤的食物，切成成人一口大小，每件不大於約 1.5 厘米（15 毫米）",
            "食物件看來容易分開——用叉、匙或筷子按壓即可壓爛",
            "全件濕潤：有光澤或裹有汁液，不乾身、不鬆散、不韌",
            "沒有硬身、有嚼勁、脆口、粗纖維或韌的部分；沒有骨頭、種籽、外皮或殼",
        ),
        negative_signatures=(
            "no whole intact grains served as the texture base; rice, if present, must look "
            "soft, moist and clumped rather than a plate of separate firm grains",
            "no pieces visibly larger than about 15 mm - no large steaks or whole dumplings",
            "no hard, crispy or tough items such as fried toppings, crackers, nuts or raw "
            "crunchy vegetables",
        ),
        negative_signatures_zh=(
            "不得以完整穀粒作為質感基礎；如有米飯，須看來軟腍濕潤黏成糰，而非一碟分明硬身的白飯",
            "不得出現明顯大於約 15 毫米的食物件——沒有大塊肉排或整隻點心",
            "不得出現炸物、脆片、餅乾、果仁或生脆蔬菜等硬脆韌口食物",
        ),
    ),
    7: LevelSignature(
        level=7,
        name="Easy to Chew / Regular",
        name_zh="容易咀嚼／正常質地",
        checklist=(
            "an ordinary, appetising everyday meal with natural food shapes and ordinary "
            "everyday textures",
            "for the easy-to-chew variant: the food looks soft, tender and moist, with no "
            "deliberately hard, tough, chewy, fibrous, stringy, sharp or dry components",
            "ordinary plate presentation - a normal meal, not visibly pureed or minced and "
            "not carrying any medical or texture-modified styling",
        ),
        checklist_zh=(
            "一份正常、吸引的日常餐食，保留天然食物形狀及日常質地",
            "容易咀嚼版本：食物看來軟腍濕潤，沒有刻意造成的硬、韌、有嚼勁、粗糙、尖銳或乾身的部分",
            "一般擺盤——看來是一份正常餐食，而非明顯打成糊或切碎，亦無醫療或質地改良的呈現",
        ),
        negative_signatures=(
            "no garnish on top that would suggest an unaudited texture - sesame seeds, "
            "crushed nuts or crispy flakes are the clearest easy-to-chew violations",
            "for the easy-to-chew variant: no hard, tough, chewy, crispy or dry components",
            "no bones, shells or sharp fragments that the visible pieces could hide",
        ),
        negative_signatures_zh=(
            "不得加上暗示未經審核質地的裝飾——芝麻、碎果仁或脆片是「容易咀嚼」最明顯的違規",
            "容易咀嚼版本：不得出現硬、韌、有嚼勁、脆或乾身的部分",
            "不得有可能藏於食物件中的骨頭、殼或尖銳碎片",
        ),
    ),
}

# The L7 Regular variant has no texture restriction (official IDDSI), so its
# checklist is deliberately reduced to presentation hygiene.
L7_REGULAR_CHECKLIST: tuple[str, ...] = (
    "an ordinary, unmodified everyday meal with natural food shapes and normal "
    "everyday textures - Level 7 Regular has no particle-size or texture restriction",
    "ordinary plate presentation with no texture-modified or medical styling",
    "do not infer chewing ability, suitability, or safety from appearance alone",
)
L7_REGULAR_CHECKLIST_ZH: tuple[str, ...] = (
    "一份未經修改的日常餐食，保留天然食物形狀及正常質地——第 7 級正常質地沒有顆粒大小或質地限制",
    "一般擺盤，沒有質地改良或醫療化的呈現",
    "不得單憑外觀推斷咀嚼能力、適合程度或安全性",
)


def judge_checklist(level: int) -> LevelSignature:
    """Return the per-level visual-signature checklist and negative signature."""
    try:
        return LEVEL_SIGNATURES[level]
    except KeyError:
        raise ValueError("level must be one of 3, 4, 5, 6, or 7") from None


def negative_signature(level: int, *, zh: bool = False) -> tuple[str, ...]:
    """Return the merged negative signature (common + level-specific).

    ``zh=True`` returns the Traditional-Chinese wording instead of English.
    """
    signature = judge_checklist(level)
    common = COMMON_NEGATIVE_SIGNATURES_ZH if zh else COMMON_NEGATIVE_SIGNATURES
    specific = signature.negative_signatures_zh if zh else signature.negative_signatures
    return common + specific


def checklist_items(level: int, *, zh: bool = False) -> tuple[str, ...]:
    """Return the visual-signature checklist items for a level.

    For level 7 the caller picks the variant explicitly via
    :func:`build_judge_prompt`; this function always returns the combined
    Easy-to-Chew / Regular wording.
    """
    signature = judge_checklist(level)
    return signature.checklist_zh if zh else signature.checklist


def _checklist_block(spec: PromptSpec) -> str:
    if spec.level == 7 and spec.level_variant == "Regular":
        items = L7_REGULAR_CHECKLIST
    else:
        items = judge_checklist(spec.level).checklist
    lines = "\n".join(f"- {item}." for item in items)
    return f"Per-level visual-signature checklist for IDDSI Level {spec.level} ({spec.level_variant}):\n{lines}"


def _negative_block(level: int) -> str:
    lines = "\n".join(f"- {item}." for item in negative_signature(level))
    return (
        f"Negative signatures - any single one appearing in the image is a rejection "
        f"reason for IDDSI Level {level}:\n{lines}"
    )


def build_judge_prompt(spec: PromptSpec) -> str:
    """Build the sharpened deterministic judge prompt for one candidate.

    Drop-in replacement for ``judge._judge_prompt``: same role line, same
    non-visual limitation, and the exact same JSON response contract, plus a
    per-level visual-signature checklist and the audited negative
    signatures.  Pure function of ``spec`` - no randomness, no network.
    """
    descriptor = LEVEL_DESCRIPTORS[spec.level]
    limit = descriptor.non_visual_limit
    limit = limit[0].lower() + limit[1:] if limit else limit
    cue = spec.cue or "none expected"
    return f"""You are auditing a synthetic food photograph for visual consistency only.
Target: IDDSI Level {spec.level} {spec.level_variant}.
Prompt-oriented visual descriptor: {spec.visual_guidance}.
Dish: {spec.dish.description_en}. Expected staged utensil cue: {cue}.

{_checklist_block(spec)}

{_negative_block(spec.level)}

Important: an image cannot prove rheology, flow, hardness, adhesiveness, cohesiveness, or a physical IDDSI test. Do not claim that it does. Judge only visible compatibility, image quality, prohibited text/faces, and whether a visible cue contradicts the target. The checklist and negative signatures are visible-appearance guidance only ({limit}).

Return exactly one JSON object with these keys:
{{"descriptor_match_score": 0.0, "visual_quality_score": 0.0, "cue_consistency_score": null, "observed_level": null, "contains_face": false, "contains_text": false, "reasons": ["brief reason"]}}
Scores are numbers from 0 to 1. cue_consistency_score is null when no cue is expected. observed_level is 3, 4, 5, 6, 7, or null. contains_face and contains_text must be JSON booleans. No markdown."""
