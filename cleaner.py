"""Clean the raw dataset and build a normalized searchable text field."""
from __future__ import annotations

import re

import pandas as pd


class CleaningError(Exception):
    """Raised when cleaning leaves no usable rows."""


_WS_RE = re.compile(r"\s+")
# characters that carry no pricing meaning; keep alphanumerics, dims and units
_JUNK_RE = re.compile(r"[^\w\s.,x×*/&%+\-()°²³\"']")

# unit spelling variants -> canonical unit
UNIT_SYNONYMS = {
    "no": "nos", "nos": "nos", "no's": "nos", "no.": "nos", "number": "nos",
    "pc": "nos", "pcs": "nos", "piece": "nos", "pieces": "nos",
    "each": "nos", "ea": "nos", "item": "nos", "unit": "nos",
    "set": "set", "sets": "set", "pair": "pair", "pairs": "pair",
    "m": "m", "mtr": "m", "meter": "m", "metre": "m", "rm": "m",
    "lm": "m", "rmt": "m", "lin.m": "m",
    "m2": "m2", "sqm": "m2", "sq.m": "m2", "m²": "m2", "sq m": "m2",
    "m3": "m3", "cum": "m3", "cu.m": "m3", "m³": "m3",
    "kg": "kg", "kgs": "kg", "ton": "ton", "tons": "ton", "tonne": "ton",
    "day": "day", "days": "day", "hour": "hour", "hr": "hour", "hrs": "hour",
    "ls": "lump sum", "lumpsum": "lump sum", "lump sum": "lump sum",
    "l": "litre", "ltr": "litre",
}

# common construction/spec spelling fixes applied inside text
_TERM_FIXES = [
    (re.compile(r"\bgalvani[sz]?ed\b|\bgalvanzied\b|\bgalv\.?\b", re.I), "galvanized"),
    (re.compile(r"\balumin(?:i?um|um)\b", re.I), "aluminium"),
    (re.compile(r"\bs\.?s\.?\s*(304|316)\b", re.I), r"ss\1"),
    (re.compile(r"\bstainless\s*steel\b", re.I), "stainless steel"),
    (re.compile(r"\bdia\.?\b|\bdiameter\b", re.I), "dia"),
    (re.compile(r"\bthk\.?\b|\bthickness\b", re.I), "thick"),
    (re.compile(r"\bsq\.?\s*m\b|\bsqm\b|m²", re.I), "m2"),
    (re.compile(r"\bcu\.?\s*m\b|\bcum\b|m³", re.I), "m3"),
    (re.compile(r"\bpowder[\s-]*coat(?:ed|ing)?\b", re.I), "powder coated"),
    (re.compile(r"\bw\.?p\.?c\.?\b", re.I), "wpc"),
]


def normalise_unit(unit: object) -> str:
    """Canonicalize a unit string; empty string when missing/unknown blank."""
    if unit is None or (isinstance(unit, float) and pd.isna(unit)):
        return ""
    text = str(unit).strip().lower().rstrip(".")
    if not text or text == "nan":
        return ""
    return UNIT_SYNONYMS.get(text, text)


def clean_text(text: object) -> str:
    """Normalize one free-text value: casing, symbols, spacing, terms.

    Dimensions, grades, units and materials are preserved.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    if s.strip().lower() in ("nan", "none", "-", "n/a", "na"):
        return ""
    s = s.replace("\r", " ").replace("\n", " ").replace("|", " ")
    s = s.lower()
    s = _JUNK_RE.sub(" ", s)
    for pattern, repl in _TERM_FIXES:
        s = pattern.sub(repl, s)
    # unify dimension separators: 1200*600 / 1200×600 -> 1200x600
    s = re.sub(r"(\d)\s*[×*]\s*(\d)", r"\1x\2", s)
    s = re.sub(r"(\d)\s*x\s*(\d)", r"\1x\2", s)
    # glue numbers to their units: "50 mm" -> "50mm"
    s = re.sub(r"(\d)\s+(mm|cm|m2|m3|kg|inch|in)\b", r"\1\2", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and add cleaned/search columns.

    Adds: display_description, clean_description, search_text, unit_norm.
    Keeps only rows with a usable description and a valid positive rate.
    """
    df = raw.copy()

    text_parts = ["item_name", "description", "specification"]
    context_parts = ["category", "section", "location", "remarks"]

    def build_display(row: pd.Series) -> str:
        parts = []
        for col in text_parts:
            val = row.get(col)
            if val is not None and not pd.isna(val):
                sval = str(val).strip()
                if sval and sval.lower() != "nan":
                    parts.append(sval.replace("\n", " "))
        return " — ".join(parts)

    df["display_description"] = df.apply(build_display, axis=1)
    df["clean_description"] = df["display_description"].map(clean_text)

    def build_search(row: pd.Series) -> str:
        parts = [row["clean_description"]]
        for col in context_parts:
            cleaned = clean_text(row.get(col))
            if cleaned:
                parts.append(cleaned)
        return " ".join(p for p in parts if p)

    df["search_text"] = df.apply(build_search, axis=1)
    df["unit_norm"] = df["unit"].map(normalise_unit)

    # drop rows that cannot support pricing
    df = df[df["clean_description"].str.len() >= 3]
    df = df[df["rate"].notna() & (df["rate"] > 0)]
    # negative/zero quantities are unusable numbers -> treat as missing
    df.loc[df["quantity"].notna() & (df["quantity"] <= 0), "quantity"] = pd.NA
    df = df.drop_duplicates(subset=["clean_description", "unit_norm", "rate"])
    df = df.reset_index(drop=True)

    if df.empty:
        raise CleaningError(
            "After cleaning, no rows remain with both a usable description "
            "and a valid rate. Check the Excel file contents."
        )
    return df
