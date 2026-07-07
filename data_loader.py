"""Load the quotation Excel file and map its columns to standard fields.

The loader does not assume exact column names. Each internal field has a
list of likely header names; the first matching column in the sheet wins.
The sheet with the most detected fields is used.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when the Excel dataset cannot be loaded or understood."""


# internal field -> likely column header names (lowercase, stripped)
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "item_name": ["item name", "item", "product", "product name", "item/service"],
    "description": [
        "description", "item description", "service description", "desc",
        "work description", "particulars", "details", "scope of work",
    ],
    "specification": ["specification", "spec", "specs", "technical specification"],
    "unit": ["unit", "uom", "unit of measure", "unit of measurement"],
    "quantity": ["quantity", "qty", "qty.", "no of units"],
    "rate": [
        "item price", "unit price", "rate", "unit rate", "price",
        "rate (usd)", "unit cost",
    ],
    "amount": ["amount", "total", "total price", "total amount", "line total"],
    "currency": ["currency", "curr"],
    "category": ["category", "trade", "package", "discipline"],
    "section": ["section", "bill", "bill section", "group"],
    "location": ["location", "area", "zone", "building"],
    "remarks": ["remarks", "notes", "note", "comment", "comments", "price check"],
    "source": ["source file", "source", "file", "source path", "source document"],
    "source_evidence": ["source evidence", "evidence", "source location"],
}


def _normalise_header(name: object) -> str:
    return str(name).strip().lower()


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map internal field names to actual column names found in df."""
    headers = {_normalise_header(c): c for c in df.columns}
    mapping: dict[str, str] = {}
    for field, candidates in COLUMN_CANDIDATES.items():
        for cand in candidates:
            if cand in headers and headers[cand] not in mapping.values():
                mapping[field] = headers[cand]
                break
    return mapping


def _pick_best_sheet(xl: pd.ExcelFile) -> tuple[str, pd.DataFrame, dict[str, str]]:
    """Choose the sheet whose headers match the most internal fields.

    Must at least contain a description-like column and a price-like column.
    """
    best: tuple[str, pd.DataFrame, dict[str, str]] | None = None
    best_score = -1
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet)
        except Exception:
            continue
        mapping = detect_columns(df)
        has_text = "description" in mapping or "item_name" in mapping
        has_price = "rate" in mapping or "amount" in mapping
        score = len(mapping) + (5 if has_text and has_price else 0)
        if has_text and has_price and score > best_score:
            best = (sheet, df, mapping)
            best_score = score
    if best is None:
        raise DataLoadError(
            "No sheet contains both a description-like column and a "
            "price/amount-like column. Sheets found: "
            + ", ".join(xl.sheet_names)
        )
    return best


def load_dataset(excel_path: str) -> tuple[pd.DataFrame, dict[str, str], str]:
    """Load the Excel file and return (raw dataframe with standard columns,
    column mapping, sheet name used).

    The returned dataframe has one column per detected internal field,
    named by the internal field. Rate is derived from amount/quantity when
    a rate column is missing.
    """
    path = Path(excel_path)
    if not path.exists():
        raise DataLoadError(
            f"Excel file not found: {excel_path}. Set the PRICING_BOT_EXCEL "
            "environment variable or place the file at data/quotation_items.xlsx."
        )
    try:
        xl = pd.ExcelFile(path)
    except Exception as exc:
        raise DataLoadError(f"Could not read Excel file {excel_path}: {exc}") from exc
    if not xl.sheet_names:
        raise DataLoadError(f"No sheets found in {excel_path}.")

    sheet, df, mapping = _pick_best_sheet(xl)
    logger.info("Using sheet %r with column mapping %s", sheet, mapping)

    out = pd.DataFrame(index=df.index)
    for field, col in mapping.items():
        out[field] = df[col]
    for field in COLUMN_CANDIDATES:
        if field not in out.columns:
            out[field] = pd.NA

    # numeric coercion at the boundary; bad values become NaN
    for num_field in ("quantity", "rate", "amount"):
        out[num_field] = pd.to_numeric(out[num_field], errors="coerce")

    # derive rate = amount / quantity when rate missing and quantity valid
    need_rate = out["rate"].isna() & out["amount"].notna() & out["quantity"].notna()
    valid_qty = out["quantity"] > 0
    derive = need_rate & valid_qty
    if derive.any():
        out.loc[derive, "rate"] = out.loc[derive, "amount"] / out.loc[derive, "quantity"]
    # derive amount when possible (display only)
    need_amount = out["amount"].isna() & out["rate"].notna() & out["quantity"].notna()
    out.loc[need_amount, "amount"] = (
        out.loc[need_amount, "rate"] * out.loc[need_amount, "quantity"]
    )

    if out.empty:
        raise DataLoadError(f"Sheet {sheet!r} in {excel_path} has no data rows.")
    return out, mapping, sheet
