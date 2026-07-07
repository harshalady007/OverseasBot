"""Ties the pipeline together: load -> clean -> search -> predict.

Main entry point:

    engine = PricingEngine()          # loads and indexes the Excel dataset
    result = engine.predict_price("Supply and install ...", top_k=5)
"""
from __future__ import annotations

import logging
from typing import Any

import config
from cleaner import clean_dataset
from data_loader import load_dataset
from deepseek_pricing import predict_price_with_deepseek
from similarity_search import SearchError, SimilaritySearch

logger = logging.getLogger(__name__)


class PricingEngineError(Exception):
    """User-facing error with a readable message."""


class PricingEngine:
    def __init__(self, excel_path: str | None = None):
        self.excel_path = excel_path or config.get_excel_path()
        raw, self.column_mapping, self.sheet_name = load_dataset(self.excel_path)
        self.raw_row_count = len(raw)
        self.df = clean_dataset(raw)
        self.search = SimilaritySearch(self.df)
        logger.info(
            "Loaded %d raw rows, %d usable rows from sheet %r",
            self.raw_row_count, len(self.df), self.sheet_name,
        )

    def dataset_summary(self) -> dict[str, Any]:
        return {
            "excel_path": self.excel_path,
            "sheet": self.sheet_name,
            "column_mapping": self.column_mapping,
            "raw_rows": self.raw_row_count,
            "usable_rows": len(self.df),
            "currencies": sorted(
                self.df["currency"].dropna().astype(str).unique().tolist()
            ),
        }

    def predict_price(self, input_description: str,
                      top_k: int = config.DEFAULT_TOP_K) -> dict[str, Any]:
        if not input_description or not input_description.strip():
            raise PricingEngineError("Please enter an item or service description.")
        if len(input_description.strip()) < 3:
            raise PricingEngineError(
                "The description is too short to search. Add more detail "
                "(item type, material, size, finish, scope)."
            )
        top_k = max(1, min(int(top_k), 20))

        try:
            search_result = self.search.search(input_description, top_k=top_k)
        except SearchError as exc:
            raise PricingEngineError(str(exc)) from exc

        matches = search_result["matches"]
        input_attrs = search_result["input_attributes"]

        warnings: list[str] = []
        if not matches:
            raise PricingEngineError(
                "No historical matches found for this description."
            )
        best = max(m["similarity_score"] for m in matches)
        if best < config.MIN_SIMILARITY_WARNING:
            warnings.append(
                f"Best similarity score is only {best:.2f}. The dataset may "
                "not contain items like this; treat the prediction as a "
                "rough guide."
            )

        prediction = predict_price_with_deepseek(
            input_description, input_attrs, matches
        )
        warnings.extend(prediction.get("warnings", []))

        return {
            "input_description": input_description.strip(),
            "input_attributes": input_attrs,
            "predicted_unit_price": prediction["predicted_unit_price"],
            "currency": prediction["currency"],
            "unit": prediction["unit"],
            "confidence": prediction["confidence"],
            "reasoning": prediction["reasoning"],
            "price_basis": prediction["price_basis"],
            "adjustments": prediction["adjustments"],
            "fallback_used": prediction["fallback_used"],
            "fallback_reason": prediction["fallback_reason"],
            "warnings": [w for w in dict.fromkeys(warnings) if w],
            "matches": matches,
        }
