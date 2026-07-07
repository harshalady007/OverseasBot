"""Extract structured pricing attributes from item descriptions and
compare an input item's attributes against a dataset item's attributes.

All extraction works on cleaned text (see cleaner.clean_text).
"""
from __future__ import annotations

import re
from typing import Any

from cleaner import clean_text, normalise_unit

# ---------------------------------------------------------------------------
# vocabularies: pattern -> canonical value (first match wins within a group)
# ---------------------------------------------------------------------------

MATERIALS = [
    (r"stainless\s*steel|ss\s?30[46]|ss\s?316", "stainless steel"),
    (r"galvanized\s*steel|galvanized\s*iron|\bgi\b", "galvanized steel"),
    (r"mild\s*steel|\bms\b(?=\s|$)", "mild steel"),
    (r"corten", "corten steel"),
    (r"\bsteel\b|\biron\b", "steel"),
    (r"aluminium", "aluminium"),
    (r"\bwpc\b", "wpc"),
    (r"bamboo", "bamboo"),
    (r"iroko|teak|pine|oak|hardwood|timber|\bwood(?:en)?\b", "timber"),
    (r"concrete|\brcc\b|cement", "concrete"),
    (r"\bglass\b", "glass"),
    (r"gypsum|plasterboard", "gypsum"),
    (r"\bpvc\b|\bupvc\b", "pvc"),
    (r"\bhdpe\b", "hdpe"),
    (r"copper", "copper"),
    (r"brass", "brass"),
    (r"granite|marble|\bstone\b", "stone"),
    (r"plastic|polymer|polyethylene", "plastic"),
]

FINISHES = [
    (r"powder\s*coated", "powder coated"),
    (r"brushed", "brushed"),
    (r"polished|mirror\s*finish", "polished"),
    (r"painted|paint\s*finish", "painted"),
    (r"galvanized", "galvanized"),
    (r"anodi[sz]ed", "anodized"),
    (r"laminated", "laminated"),
    (r"wood\s*grain", "wood grain"),
    (r"textured", "textured"),
    (r"matte|matt\b", "matte"),
    (r"gloss", "gloss"),
    (r"satin", "satin"),
]

GRADES = [
    (r"ss\s?304|grade\s*304", "ss304"),
    (r"ss\s?316|grade\s*316", "ss316"),
    (r"\bm(15|20|25|30|35|40)\b", None),  # concrete grade, keep matched text
    (r"fire\s*rated", "fire rated"),
    (r"acoustic", "acoustic"),
    (r"water\s*proof|waterproof", "waterproof"),
    (r"marine\s*grade", "marine grade"),
    (r"heavy\s*duty", "heavy duty"),
    (r"grade\s*[a-c1-9]\w*", None),
]

SCOPES = [
    (r"supply\s*(?:,|and|&)\s*install(?:ation)?|supply\s*&\s*fix", "supply and install"),
    (r"supply\s*only|supply\s*of\b", "supply only"),
    (r"install(?:ation)?\s*only|fix(?:ing)?\s*only", "install only"),
    (r"labou?r\s*only", "labour only"),
    (r"testing\s*(?:and|&)\s*commissioning", "testing and commissioning"),
]

CATEGORIES = [
    (r"handrail|balustrade|railing|guardrail", "metalwork - railing"),
    (r"\bbench(?:es)?\b|litter\s*bin|trash\s*bin|recycling|\bbin\b|bollard|"
     r"bike\s*rack|bicycle|cycle\s*stand|planter|pergola|gazebo|shade\s*structure|"
     r"sun\s*lounger|daybed|picnic|street\s*furniture|outdoor\s*furniture|ashtray",
     "street furniture"),
    (r"\btable\b|\bchair\b|sofa|cabinet|wardrobe|joinery|furniture", "furniture/joinery"),
    (r"\bcable\b|wiring|switch|socket|light(?:ing)?|\bled\b|electrical|\bdb\b",
     "electrical"),
    (r"\bpipe\b|plumbing|drainage|sanitary|water\s*supply", "plumbing"),
    (r"\bduct\b|hvac|air\s*condition|ventilation|chiller|\bac\b(?=\s|$)", "hvac"),
    (r"floor(?:ing)?|tile|tiling|carpet|vinyl|screed", "flooring"),
    (r"ceiling|bulkhead", "ceiling"),
    (r"paint(?:ing)?\b|emulsion|epoxy\s*coating", "painting"),
    (r"door|window|glazing|curtain\s*wall", "doors/windows"),
    (r"excavation|backfill|foundation|masonry|blockwork|plaster|civil", "civil"),
    (r"signage|sign\s*board", "signage"),
]

LOCATIONS = [
    (r"outdoor|external|park|garden|landscape|playground", "outdoor"),
    (r"indoor|internal", "indoor"),
    (r"fa[cç]ade|elevation|cladding", "facade"),
    (r"\broof\b|terrace", "roof"),
    (r"bathroom|toilet|washroom", "bathroom"),
    (r"kitchen|pantry", "kitchen"),
    (r"plant\s*room|substation", "plant room"),
]

# item-type keywords used for keyword overlap scoring (nouns that identify
# what the item IS, which matters more than adjectives)
ITEM_TYPE_WORDS = [
    "handrail", "balustrade", "railing", "bench", "bin", "bollard", "rack",
    "table", "chair", "lounger", "daybed", "pergola", "gazebo", "shade",
    "planter", "umbrella", "shelter", "sign", "gate", "fence", "door",
    "window", "ladder", "grating", "plate", "pipe", "cable", "duct",
    "cushion", "ashtray", "canopy", "stand", "swing", "seesaw", "slide",
]

_UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "inch": 25.4, '"': 25.4}


def _to_mm(value: float, unit: str) -> float:
    return value * _UNIT_TO_MM.get(unit, 1.0)


def _first_match(text: str, vocab: list[tuple[str, str | None]]) -> str | None:
    for pattern, canonical in vocab:
        m = re.search(pattern, text)
        if m:
            return canonical if canonical is not None else m.group(0)
    return None


def _all_matches(text: str, vocab: list[tuple[str, str | None]]) -> list[str]:
    found: list[str] = []
    for pattern, canonical in vocab:
        m = re.search(pattern, text)
        if m:
            val = canonical if canonical is not None else m.group(0)
            if val not in found:
                found.append(val)
    return found


def _extract_dimensions(text: str) -> dict[str, Any]:
    """Pull dimensions in mm out of text: LxWxH blocks, dia, thickness,
    lengths, areas/volumes."""
    dims: dict[str, Any] = {}

    # "l 1800 x w 530 x h 530mm" or "1200x600mm" or "2280x2048x770mm"
    block = re.search(
        r"(?:l\s*)?(\d{2,5}(?:\.\d+)?)\s*(?:mm|cm|m)?\s*x\s*(?:w\s*)?"
        r"(\d{2,5}(?:\.\d+)?)\s*(?:mm|cm|m)?(?:\s*x\s*(?:h\s*)?"
        r"(\d{2,5}(?:\.\d+)?))?\s*(mm|cm|m)?\b",
        text,
    )
    if block:
        unit = block.group(4) or "mm"
        vals = [
            _to_mm(float(v), unit)
            for v in block.groups()[:3]
            if v is not None
        ]
        dims["dimensions_mm"] = vals

    dia = re.search(r"(\d+(?:\.\d+)?)\s*(mm|cm|m|inch|in)?\s*dia\b", text) or re.search(
        r"\bdia\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|inch|in)?", text
    )
    if dia:
        dims["diameter_mm"] = _to_mm(float(dia.group(1)), dia.group(2) or "mm")

    thick = re.search(r"(\d+(?:\.\d+)?)\s*(mm|cm|m)?\s*thick\b", text)
    if thick:
        dims["thickness_mm"] = _to_mm(float(thick.group(1)), thick.group(2) or "mm")

    length = re.search(r"(\d+(?:\.\d+)?)\s*(mm|cm|m)\s*(?:long|length)\b", text)
    if length:
        dims["length_mm"] = _to_mm(float(length.group(1)), length.group(2))

    area = re.search(r"(\d+(?:\.\d+)?)\s*m2\b", text)
    if area:
        dims["area_m2"] = float(area.group(1))
    volume = re.search(r"(\d+(?:\.\d+)?)\s*m3\b", text)
    if volume:
        dims["volume_m3"] = float(volume.group(1))
    return dims


def extract_attributes(description: str, unit: str | None = None) -> dict[str, Any]:
    """Extract a structured attribute dictionary from a description.

    Absent attributes are simply not present in the returned dict.
    """
    text = clean_text(description)
    attrs: dict[str, Any] = {}

    material = _first_match(text, MATERIALS)
    if material:
        attrs["material"] = material
    finishes = _all_matches(text, FINISHES)
    if finishes:
        attrs["finish"] = finishes
    grade = _first_match(text, GRADES)
    if grade:
        attrs["grade"] = grade
    scope = _first_match(text, SCOPES)
    if scope:
        attrs["scope"] = scope
    category = _first_match(text, CATEGORIES)
    if category:
        attrs["category"] = category
    location = _first_match(text, LOCATIONS)
    if location:
        attrs["location"] = location

    attrs.update(_extract_dimensions(text))

    item_types = [w for w in ITEM_TYPE_WORDS if re.search(rf"\b{w}s?\b", text)]
    if item_types:
        attrs["item_type"] = item_types

    unit_norm = normalise_unit(unit) if unit else ""
    if not unit_norm:
        # units sometimes live inside the description ("per m2", "/rm")
        m = re.search(r"\bper\s+(\w+)|/(\w+)\b", text)
        if m:
            unit_norm = normalise_unit(m.group(1) or m.group(2))
    if unit_norm:
        attrs["unit"] = unit_norm

    brand = re.search(r"\b(?:brand|make|manufacturer)[:\s]+([a-z0-9][\w\- ]{2,25})", text)
    if brand:
        attrs["brand"] = brand.group(1).strip()

    return attrs


# ---------------------------------------------------------------------------
# attribute comparison
# ---------------------------------------------------------------------------

# weight of each attribute in the attribute score (normalized over the
# attributes actually present in the input)
ATTRIBUTE_WEIGHTS = {
    "item_type": 0.28,
    "material": 0.22,
    "dimensions": 0.14,
    "finish": 0.08,
    "scope": 0.08,
    "unit": 0.08,
    "category": 0.07,
    "grade": 0.05,
}

_DIM_TOLERANCE = 0.25  # relative tolerance for "similar" dimensions


def _dims_similar(a: float, b: float) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) / max(a, b) <= _DIM_TOLERANCE


def _compare_dimensions(inp: dict, item: dict) -> tuple[float, list[str], list[str]]:
    """Compare all numeric dimension attributes; returns (score 0..1,
    matched notes, mismatched notes)."""
    matched, mismatched = [], []
    scores = []
    for key, label in (
        ("diameter_mm", "diameter"),
        ("thickness_mm", "thickness"),
        ("length_mm", "length"),
        ("area_m2", "area"),
        ("volume_m3", "volume"),
    ):
        if key in inp and key in item:
            if _dims_similar(float(inp[key]), float(item[key])):
                matched.append(f"{label} ~{inp[key]:g}")
                scores.append(1.0)
            else:
                mismatched.append(f"{label} {inp[key]:g} vs {item[key]:g}")
                scores.append(0.0)
    if "dimensions_mm" in inp and "dimensions_mm" in item:
        a, b = inp["dimensions_mm"], item["dimensions_mm"]
        pairs = list(zip(sorted(a, reverse=True), sorted(b, reverse=True)))
        if pairs:
            hit = sum(1 for x, y in pairs if _dims_similar(x, y)) / len(pairs)
            scores.append(hit)
            note = f"overall size {'x'.join(f'{v:g}' for v in a)}mm"
            (matched if hit >= 0.5 else mismatched).append(
                note if hit >= 0.5 else
                f"size {'x'.join(f'{v:g}' for v in a)} vs {'x'.join(f'{v:g}' for v in b)} mm"
            )
    score = sum(scores) / len(scores) if scores else -1.0  # -1 = not comparable
    return score, matched, mismatched


def compare_attributes(input_attrs: dict, item_attrs: dict) -> dict[str, Any]:
    """Compare input attributes with one dataset item's attributes.

    Returns dict with: score (0..1), matched, mismatched, missing —
    where missing = attributes the input specifies but the item does not
    mention (unknown, not necessarily different).
    """
    matched: list[str] = []
    mismatched: list[str] = []
    missing: list[str] = []
    total_weight = 0.0
    gained = 0.0

    def judge(name: str, weight: float, inp_val, item_val, hit: bool | None):
        nonlocal total_weight, gained
        total_weight += weight
        if item_val is None:
            missing.append(f"{name} ({inp_val}) not stated in match")
            gained += weight * 0.4  # unknown: mildly penalised, not zeroed
        elif hit:
            matched.append(f"{name}: {inp_val}")
            gained += weight
        else:
            mismatched.append(f"{name}: {inp_val} vs {item_val}")

    if "item_type" in input_attrs:
        inp_types = set(input_attrs["item_type"])
        item_types = set(item_attrs.get("item_type", []))
        if not item_attrs.get("item_type"):
            judge("item type", ATTRIBUTE_WEIGHTS["item_type"],
                  ", ".join(inp_types), None, None)
        else:
            overlap = inp_types & item_types
            judge("item type", ATTRIBUTE_WEIGHTS["item_type"],
                  ", ".join(sorted(inp_types)),
                  ", ".join(sorted(item_types)), bool(overlap))

    if "material" in input_attrs:
        im, dm = input_attrs["material"], item_attrs.get("material")
        related = dm is not None and (
            im == dm or ("steel" in im and "steel" in dm)
        )
        judge("material", ATTRIBUTE_WEIGHTS["material"], im, dm,
              im == dm or (related and im == dm))
        # partially related materials (both steels) get half credit
        if dm is not None and im != dm and related:
            gained += ATTRIBUTE_WEIGHTS["material"] * 0.5
            mismatched[-1] += " (related)"

    dim_keys = ("dimensions_mm", "diameter_mm", "thickness_mm", "length_mm",
                "area_m2", "volume_m3")
    if any(k in input_attrs for k in dim_keys):
        w = ATTRIBUTE_WEIGHTS["dimensions"]
        score, dm_matched, dm_mismatched = _compare_dimensions(input_attrs, item_attrs)
        total_weight += w
        if score < 0:
            missing.append("dimensions not comparable (match has no sizes)")
            gained += w * 0.4
        else:
            gained += w * score
            matched.extend(dm_matched)
            mismatched.extend(dm_mismatched)

    if "finish" in input_attrs:
        inp_f = set(input_attrs["finish"])
        item_f = set(item_attrs.get("finish", []))
        if not item_f:
            judge("finish", ATTRIBUTE_WEIGHTS["finish"],
                  ", ".join(sorted(inp_f)), None, None)
        else:
            judge("finish", ATTRIBUTE_WEIGHTS["finish"],
                  ", ".join(sorted(inp_f)), ", ".join(sorted(item_f)),
                  bool(inp_f & item_f))

    for name, key in (("scope", "scope"), ("unit", "unit"),
                      ("category", "category"), ("grade", "grade")):
        if key in input_attrs:
            judge(name, ATTRIBUTE_WEIGHTS[key], input_attrs[key],
                  item_attrs.get(key),
                  input_attrs[key] == item_attrs.get(key))

    score = gained / total_weight if total_weight > 0 else 0.5
    return {
        "score": round(min(max(score, 0.0), 1.0), 4),
        "matched": matched,
        "mismatched": mismatched,
        "missing": missing,
    }
