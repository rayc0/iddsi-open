"""Prompt construction for visually varied, explicitly synthetic food images.

The descriptor text is a compact prompt-oriented paraphrase of the official
IDDSI L3--L7 descriptions.  It is visual guidance, never a physical-test label.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class LevelDescriptor:
    level: int
    name: str
    prompt_guidance: str
    non_visual_limit: str


@dataclass(frozen=True)
class Dish:
    slug: str
    name_zh: str
    description_en: str
    cuisine_tags: tuple[str, ...]


@dataclass(frozen=True)
class PromptSpec:
    level: int
    level_variant: str
    visual_guidance: str
    dish: Dish
    prompt: str
    short_prompt: str
    negative_prompt: str
    lighting_category: str
    lighting_description: str
    plate_category: str
    plate_description: str
    phone_profile: str
    angle: str
    cue: str | None


LEVEL_DESCRIPTORS: dict[int, LevelDescriptor] = {
    3: LevelDescriptor(
        3,
        "Liquidised",
        "completely smooth and lump-free, an even cohesive liquidised consistency that visibly flows slowly; no chunks, fibres, skins, seeds, crusts, or separated thin liquid",
        "A photograph cannot establish flow rate or oral-processing requirements.",
    ),
    4: LevelDescriptor(
        4,
        "Pureed",
        "completely smooth and lump-free, thick and cohesive, holding a soft mound on a spoon or plate without being pourable, with no crust, skin, fibres, seeds, garnish, or separated liquid",
        "A photograph cannot establish spoon-tilt, fork-drip, stickiness, or cohesiveness.",
    ),
    5: LevelDescriptor(
        5,
        "Minced & Moist",
        "soft moist minced food with small visible particles no larger than about 4 mm, cohesive enough to hold together, no hard pieces, stringy fibres, skins, bones, crusts, or separated thin liquid",
        "A photograph cannot establish particle softness, tongue pressure, or safe particle size.",
    ),
    6: LevelDescriptor(
        6,
        "Soft & Bite-Sized",
        "soft tender moist pieces cut into adult bite-sized portions no larger than about 15 mm, pieces visibly easy to break apart, no hard, tough, chewy, fibrous, crispy, crumbly, seeded, bony, or stringy components",
        "A photograph cannot establish fork-pressure softness, deformation, or rebound.",
    ),
    7: LevelDescriptor(
        7,
        "Easy to Chew / Regular",
        "an ordinary appetising meal presented in an easy-to-chew soft and tender form; natural food shapes may remain, with no deliberately hard, tough, chewy, fibrous, stringy, sharp, or dry components in the easy-to-chew variant",
        "A photograph cannot establish chewing effort; Level 7 Regular itself has no texture restriction.",
    ),
}


def _dish(slug: str, zh: str, en: str, *tags: str) -> Dish:
    return Dish(slug, zh, en, tuple(tags))


# More than sixty Hong Kong / Chinese soft-meal foundations.  Every item can be
# transformed into each declared prompt level; the prompt describes that
# transformation rather than asserting the original recipe already meets it.
CHINESE_CANTONESE_DISHES: tuple[Dish, ...] = (
    _dish("plain_congee", "白粥", "plain rice congee", "chinese", "cantonese"),
    _dish("fish_congee", "魚片粥", "Cantonese fish congee", "chinese", "cantonese"),
    _dish("chicken_congee", "雞粥", "chicken rice congee", "chinese", "cantonese"),
    _dish("lean_pork_congee", "瘦肉粥", "lean pork congee", "chinese", "cantonese"),
    _dish("century_egg_congee", "皮蛋瘦肉粥", "century egg and lean pork congee", "chinese", "cantonese"),
    _dish("beancurd_sheet_congee", "腐竹白果粥", "bean-curd sheet and ginkgo congee", "chinese", "cantonese"),
    _dish("pumpkin_congee", "南瓜粥", "pumpkin rice congee", "chinese", "cantonese"),
    _dish("millet_congee", "小米粥", "millet congee", "chinese"),
    _dish("sweet_potato_congee", "番薯粥", "sweet potato congee", "chinese"),
    _dish("taro_congee", "芋頭粥", "taro rice congee", "chinese", "cantonese"),
    _dish("steamed_egg", "蒸水蛋", "silky Cantonese steamed egg custard", "chinese", "cantonese"),
    _dish("egg_tofu_custard", "蛋白豆腐蒸", "steamed egg-white and tofu custard", "chinese", "cantonese"),
    _dish("minced_pork_egg", "肉碎蒸蛋", "steamed egg with finely chopped pork", "chinese", "cantonese"),
    _dish("tofu", "嫩豆腐", "soft silken tofu", "chinese"),
    _dish("braised_tofu", "紅燒豆腐", "soft braised tofu", "chinese"),
    _dish("tofu_fish", "豆腐魚蓉", "tofu with fish paste", "chinese", "cantonese"),
    _dish("fish_puree", "魚蓉", "steamed boneless fish mash", "chinese", "cantonese"),
    _dish("fish_mousse", "魚滑", "Cantonese fish mousse", "chinese", "cantonese"),
    _dish("steamed_fish", "清蒸魚肉", "boneless steamed fish flesh", "chinese", "cantonese"),
    _dish("fish_tofu", "魚腐", "soft fish tofu", "chinese", "cantonese"),
    _dish("steamed_pork_patty", "蒸肉餅", "Cantonese steamed pork patty", "chinese", "cantonese"),
    _dish("mushroom_pork_patty", "冬菇蒸肉餅", "steamed pork patty with mushroom flavour", "chinese", "cantonese"),
    _dish("minced_chicken", "雞肉蓉", "moist finely chopped chicken", "chinese"),
    _dish("minced_beef", "牛肉蓉", "moist finely chopped beef", "chinese", "cantonese"),
    _dish("soft_lion_head", "軟燴獅子頭", "soft braised pork meatball", "chinese"),
    _dish("pumpkin_puree", "南瓜蓉", "steamed pumpkin mash", "chinese"),
    _dish("winter_melon", "冬瓜", "soft braised winter melon", "chinese", "cantonese"),
    _dish("wax_gourd_puree", "冬瓜蓉", "winter melon mash", "chinese", "cantonese"),
    _dish("carrot_puree", "甘筍蓉", "carrot mash", "chinese", "cantonese"),
    _dish("sweet_potato_mash", "番薯蓉", "sweet potato mash", "chinese"),
    _dish("taro_mash", "芋泥", "smooth taro mash", "chinese", "cantonese"),
    _dish("yam_mash", "淮山蓉", "Chinese yam mash", "chinese", "cantonese"),
    _dish("potato_mash_cn", "薯蓉", "Chinese-style potato mash", "chinese", "cantonese"),
    _dish("lotus_root_puree", "蓮藕蓉", "lotus-root mash", "chinese", "cantonese"),
    _dish("chestnut_puree", "栗子蓉", "chestnut mash", "chinese"),
    _dish("pea_puree", "青豆蓉", "green pea mash", "chinese"),
    _dish("spinach_puree", "菠菜蓉", "spinach mash", "chinese"),
    _dish("choy_sum_puree", "菜心蓉", "choy sum mash", "chinese", "cantonese"),
    _dish("broccoli_puree", "西蘭花蓉", "broccoli mash", "chinese", "cantonese"),
    _dish("cauliflower_puree", "椰菜花蓉", "cauliflower mash", "chinese", "cantonese"),
    _dish("eggplant_braise", "燴茄子", "very soft braised eggplant", "chinese"),
    _dish("luffa_braise", "燴絲瓜", "soft braised luffa", "chinese", "cantonese"),
    _dish("marrow_braise", "節瓜炆煮", "soft braised fuzzy melon", "chinese", "cantonese"),
    _dish("tomato_egg", "番茄炒蛋", "soft tomato and egg", "chinese"),
    _dish("egg_drop_custard", "蛋花羹", "thick egg-drop soup", "chinese"),
    _dish("winter_melon_soup", "冬瓜羹", "thick winter melon soup", "chinese", "cantonese"),
    _dish("corn_soup", "粟米羹", "thick sweetcorn soup", "chinese", "cantonese"),
    _dish("fish_maw_style_soup", "仿魚肚羹", "smooth thick Cantonese-style savoury soup", "chinese", "cantonese"),
    _dish("west_lake_beef_soup", "西湖牛肉羹", "West Lake beef thick soup", "chinese", "cantonese"),
    _dish("hot_sour_soup", "酸辣羹", "hot-and-sour thick soup", "chinese", "cantonese"),
    _dish("pumpkin_soup_cn", "南瓜羹", "Chinese pumpkin soup", "chinese"),
    _dish("thickened_chicken_soup", "增稠雞湯", "texture-modified thickened chicken soup", "chinese", "cantonese"),
    _dish("thickened_fish_soup", "增稠魚湯", "texture-modified thickened fish soup", "chinese", "cantonese"),
    _dish("thickened_herbal_soup", "增稠老火湯", "thickened Cantonese slow-cooked soup", "chinese", "cantonese"),
    _dish("rice_paste", "米糊", "smooth rice paste", "chinese"),
    _dish("sesame_paste", "芝麻糊", "black sesame dessert paste", "chinese", "cantonese"),
    _dish("walnut_paste", "合桃糊", "walnut dessert paste", "chinese", "cantonese"),
    _dish("almond_tea", "杏仁糊", "thick almond dessert soup", "chinese", "cantonese"),
    _dish("peanut_paste", "花生糊", "peanut dessert paste", "chinese", "cantonese"),
    _dish("cashew_paste", "腰果糊", "cashew dessert paste", "chinese", "cantonese"),
    _dish("red_bean_paste", "紅豆沙", "smooth red bean dessert soup", "chinese", "cantonese"),
    _dish("mung_bean_paste", "綠豆沙", "smooth mung bean dessert soup", "chinese", "cantonese"),
    _dish("egg_custard_cn", "燉蛋", "Chinese steamed egg dessert custard", "chinese", "cantonese"),
    _dish("milk_custard", "燉奶", "Cantonese double-skin-style milk custard", "chinese", "cantonese"),
    _dish("tofu_pudding", "豆腐花", "silky tofu pudding", "chinese", "cantonese"),
    _dish("snow_fungus_papaya", "雪耳燉木瓜", "snow fungus and papaya sweet soup", "chinese", "cantonese"),
    _dish("pear_puree", "燉雪梨蓉", "smooth stewed pear", "chinese", "cantonese"),
    _dish("papaya_puree", "木瓜蓉", "smooth ripe papaya", "chinese", "cantonese"),
    _dish("banana_mash", "香蕉蓉", "banana mash", "chinese"),
    _dish("mango_puree", "芒果蓉", "smooth ripe mango", "chinese", "cantonese"),
    _dish("soft_noodles", "軟爛麵", "very soft shortened noodles", "chinese"),
    _dish("rice_vermicelli_soft", "軟米粉", "very soft shortened rice vermicelli", "chinese", "cantonese"),
    _dish("rice_roll_soft", "軟腸粉", "soft plain rice noodle roll", "chinese", "cantonese"),
    _dish("turnip_cake_soft", "軟蘿蔔糕", "soft steamed turnip cake", "chinese", "cantonese"),
    _dish("rice_custard_chicken", "雞蓉燴飯", "moist rice with finely chopped chicken", "chinese", "cantonese"),
    _dish("minced_fish_rice", "魚蓉燴飯", "moist rice with fish mash", "chinese", "cantonese"),
    _dish("soft_dumpling_filling", "軟餃子餡", "moist dumpling filling without wrapper", "chinese"),
)


WESTERN_DISHES: tuple[Dish, ...] = (
    _dish("mashed_potato", "薯蓉", "buttery mashed potato", "western"),
    _dish("pumpkin_soup", "南瓜濃湯", "cream of pumpkin soup", "western"),
    _dish("tomato_soup", "番茄濃湯", "smooth tomato soup", "western"),
    _dish("mushroom_soup", "蘑菇濃湯", "smooth cream of mushroom soup", "western"),
    _dish("pea_soup", "青豆濃湯", "smooth green pea soup", "western"),
    _dish("chicken_mousse", "雞肉慕斯", "savoury chicken mousse", "western"),
    _dish("salmon_mousse", "三文魚慕斯", "boneless salmon mousse", "western"),
    _dish("fish_pie", "魚批", "soft crust-free fish pie filling", "western"),
    _dish("cottage_pie", "農舍批", "soft cottage pie filling", "western"),
    _dish("shepherds_pie", "牧羊人批", "soft shepherd's pie filling", "western"),
    _dish("scrambled_egg", "炒滑蛋", "soft moist scrambled egg", "western"),
    _dish("cheese_omelette", "芝士奄列", "soft cheese omelette", "western"),
    _dish("macaroni_cheese", "芝士通粉", "very soft macaroni cheese", "western"),
    _dish("risotto", "意大利燴飯", "soft creamy risotto", "western"),
    _dish("polenta", "玉米糊", "soft creamy polenta", "western"),
    _dish("oatmeal", "燕麥糊", "smooth oatmeal porridge", "western"),
    _dish("bread_pudding", "麵包布甸", "soft crust-free bread pudding", "western"),
    _dish("vanilla_custard", "雲呢拿吉士", "smooth vanilla custard", "western"),
    _dish("yogurt", "乳酪", "smooth plain yogurt", "western"),
    _dish("stewed_apple", "燉蘋果蓉", "smooth stewed apple", "western"),
    _dish("soft_meatloaf", "軟肉餅", "soft moist meatloaf", "western"),
    _dish("braised_chicken", "燴雞肉", "very tender moist braised chicken", "western"),
)

ALL_DISHES: tuple[Dish, ...] = CHINESE_CANTONESE_DISHES + WESTERN_DISHES

LIGHTING = (
    ("daylight", "soft window daylight with natural shadows"),
    ("overhead", "ordinary warm overhead kitchen lighting"),
    ("mixed", "mixed window and kitchen light with realistic white-balance variation"),
    ("low_light", "slightly dim domestic dining light with mild phone-camera noise"),
    ("controlled", "even neutral food-photography light without glamour styling"),
)

PLATES = (
    ("light_plain", "a plain white ceramic plate"),
    ("dark_plain", "a matte dark ceramic plate"),
    ("patterned", "a restrained blue-and-white patterned plate"),
    ("reflective", "a stainless-steel institutional meal tray"),
    ("bowl", "a simple ceramic bowl"),
)

PHONE_PROFILES = (
    "clean modern mobile snapshot with natural computational sharpening",
    "mid-range mobile snapshot with slight oversharpening",
    "older mobile snapshot with modest sensor noise and limited dynamic range",
    "modern mobile snapshot with realistic auto white balance",
)

ANGLES = (
    "approximately 45 degrees above the plate",
    "slightly higher than 45 degrees, full dish visible",
    "slightly lower than 45 degrees, full dish visible",
    "near-45-degree handheld framing with a small natural rotation",
)

CUES: dict[int, tuple[str, ...]] = {
    3: ("a spoon held tilted above the bowl in a frozen still moment", "a standard fork held above the food showing a frozen fork-drip cue"),
    4: ("a spoon held tilted above the plate in a frozen still moment", "a standard fork touching the puree in a frozen fork-drip cue"),
    5: ("a spoon held tilted beside the moist mince in a frozen still moment", "a standard fork visibly pressed into one representative portion"),
    6: ("a standard fork visibly pressed into one representative bite-sized piece",),
    7: ("a standard fork visibly pressed into one representative soft piece",),
}

NEGATIVE_PROMPT = (
    "text, words, letters, captions, labels, logos, watermarks, packaging, menu, signage, "
    "faces, head, person, patient, child, medical setting, hospital wristband, medication, "
    "unsafe swallowing claim, IDDSI logo, rating badge, level number, infographic, diagram, "
    "phone, smartphone, mobile phone, handset, screen, device, camera, hand holding a phone, "
    "whole rice grains, loose grains, visible grains, intact vegetables, fried egg, egg yolk, "
    "herb garnish, parsley, spring onion, crispy topping, bread, noodles, "
    "utensil-only shot with no food visible, empty plate, empty bowl, cutlery alone, "
    "extra utensils, malformed fork, deformed spoon, duplicate food, floating objects, "
    "plastic-looking food, impossible geometry, illustration, cartoon, CGI render, excessive bokeh"
)


def dishes(cuisine: str = "all") -> tuple[Dish, ...]:
    if cuisine == "all":
        return ALL_DISHES
    if cuisine in {"chinese", "cantonese"}:
        return tuple(item for item in CHINESE_CANTONESE_DISHES if cuisine in item.cuisine_tags)
    if cuisine == "western":
        return WESTERN_DISHES
    raise ValueError(f"unsupported cuisine pool: {cuisine!r}")


def short_caption(dish: Dish, plate_description: str) -> str:
    """Derive a compact caption of the dish and plating essentials.

    FLUX routes ``prompt`` to a CLIP text encoder with a hard 77-token limit,
    so the full descriptor is silently truncated there.  This deterministic
    caption keeps only the dish identity and plating surface for the CLIP
    pathway; the full descriptor travels via ``prompt_2`` (T5) instead.
    """
    return (
        f"Documentary food photograph of {dish.description_en} served on {plate_description}, "
        "casual handheld phone-camera style, entire dish in frame."
    )


def build_prompt(
    level: int,
    rng: random.Random,
    *,
    cuisine: str = "all",
    cue_probability: float = 0.35,
    dish: Dish | None = None,
) -> PromptSpec:
    """Sample one deterministic prompt specification from a supplied RNG."""
    if level not in LEVEL_DESCRIPTORS:
        raise ValueError("level must be one of 3, 4, 5, 6, or 7")
    if not 0.0 <= cue_probability <= 1.0:
        raise ValueError("cue_probability must be between 0 and 1")
    selected_dish = dish or rng.choice(dishes(cuisine))
    lighting_category, lighting_description = rng.choice(LIGHTING)
    plate_category, plate_description = rng.choice(PLATES)
    phone_profile = rng.choice(PHONE_PROFILES)
    angle = rng.choice(ANGLES)
    cue = rng.choice(CUES[level]) if rng.random() < cue_probability else None
    descriptor = LEVEL_DESCRIPTORS[level]
    level_variant = descriptor.name
    visual_guidance = descriptor.prompt_guidance
    if level == 7:
        level_variant = rng.choice(("Easy to Chew", "Regular"))
        if level_variant == "Easy to Chew":
            visual_guidance = (
                "an ordinary appetising meal in a deliberately soft and tender easy-to-chew form, with natural food shapes allowed "
                "and no visibly hard, tough, chewy, fibrous, stringy, sharp, or dry components"
            )
        else:
            visual_guidance = (
                "an ordinary unmodified meal with natural food shapes and no particle-size restriction; avoid implying that appearance "
                "alone establishes chewing ability or suitability"
            )

    cue_sentence = (
        f" Include {cue}; it is only a visible staged cue in one photograph, not proof of a test result."
        if cue
        else " No hand or person is visible."
    )
    prompt = (
        "Documentary food photograph, casual handheld phone-camera style, for synthetic machine-learning pipeline development. "
        f"Show {selected_dish.description_en} ({selected_dish.name_zh}) deliberately prepared to look visually compatible with "
        f"IDDSI Level {level} {level_variant}: {visual_guidance}. "
        f"Serve it on {plate_description}, photographed {angle}, under {lighting_description}; image quality like a {phone_profile} (no device visible in the scene). "
        "Realistic food-service portion, entire dish in frame, ordinary imperfect smartphone capture, accurate food colour and texture."
        f"{cue_sentence} Do not show any written text, label, level marker, logo, face, or clinical context."
    )
    return PromptSpec(
        level=level,
        level_variant=level_variant,
        visual_guidance=visual_guidance,
        dish=selected_dish,
        prompt=prompt,
        short_prompt=short_caption(selected_dish, plate_description),
        negative_prompt=NEGATIVE_PROMPT,
        lighting_category=lighting_category,
        lighting_description=lighting_description,
        plate_category=plate_category,
        plate_description=plate_description,
        phone_profile=phone_profile,
        angle=angle,
        cue=cue,
    )


def crossed_dishes(levels: Sequence[int] = (3, 4, 5, 6, 7)) -> Iterable[tuple[int, Dish]]:
    """Yield the full descriptor-by-dish cross without materialising prompts."""
    for level in levels:
        if level not in LEVEL_DESCRIPTORS:
            raise ValueError(f"unsupported level: {level}")
        for item in ALL_DISHES:
            yield level, item


_PUNCTUATION = re.compile(r"[^0-9a-z一-鿿]+")


def normalize_prompt(text: str) -> str:
    """Fold a prompt to a canonical form for near-duplicate detection.

    Lowercases, strips punctuation (keeping CJK characters), and collapses
    whitespace, so two prompts differing only in case, spacing, or
    punctuation compare equal.
    """
    lowered = text.lower()
    collapsed = " ".join(lowered.split())
    folded = _PUNCTUATION.sub(" ", collapsed)
    return " ".join(folded.split())


@dataclass(frozen=True)
class DedupeStats:
    """Counts behind an exact + normalised prompt dedupe pass."""

    total: int
    removed_exact: int
    removed_normalized: int
    kept: int

    @property
    def removed_total(self) -> int:
        return self.total - self.kept


def dedupe_prompts(prompts: Sequence[str]) -> tuple[list[str], DedupeStats]:
    """Return prompts with exact and normalised near-duplicates removed.

    Order of first occurrence is preserved.  ``removed_exact`` counts
    duplicates found by exact string equality; ``removed_normalized``
    counts additional near-duplicates that only matched after
    :func:`normalize_prompt`.
    """
    unique: list[str] = []
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    removed_exact = removed_normalized = 0
    for prompt in prompts:
        if prompt in seen_exact:
            removed_exact += 1
            continue
        key = normalize_prompt(prompt)
        if key in seen_normalized:
            removed_normalized += 1
            continue
        seen_exact.add(prompt)
        seen_normalized.add(key)
        unique.append(prompt)
    return unique, DedupeStats(
        total=len(prompts),
        removed_exact=removed_exact,
        removed_normalized=removed_normalized,
        kept=len(unique),
    )
