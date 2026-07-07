"""Hybrid similarity search: TF-IDF cosine similarity over cleaned text,
adjusted by structured attribute comparison."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from attribute_extractor import compare_attributes, extract_attributes
from cleaner import clean_text


class SearchError(Exception):
    """Raised when the search cannot run (empty input/index)."""


class SimilaritySearch:
    """Builds a TF-IDF index over the cleaned dataset once, then answers
    queries with hybrid text + attribute scores."""

    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise SearchError("Cannot build search index: dataset is empty.")
        self.df = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w[\w.]*\b",  # keep tokens like 50mm, ss304
        )
        self.matrix = self.vectorizer.fit_transform(self.df["search_text"])
        # pre-extract attributes for every dataset row (unit passed separately)
        self.item_attrs: list[dict] = [
            extract_attributes(row.search_text, row.unit_norm or None)
            for row in self.df.itertuples()
        ]

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> dict[str, Any]:
        """Return input attributes and the ranked top matches."""
        cleaned = clean_text(query)
        if not cleaned:
            raise SearchError("Input description is empty.")
        input_attrs = extract_attributes(query)

        query_vec = self.vectorizer.transform([cleaned])
        text_scores = cosine_similarity(query_vec, self.matrix)[0]

        # score every row on attributes only for the text top-50 (cheap enough
        # for small datasets, avoids attribute work on obvious non-matches)
        candidate_count = min(len(self.df), max(top_k * 10, 50))
        candidates = text_scores.argsort()[::-1][:candidate_count]

        results = []
        for idx in candidates:
            comparison = compare_attributes(input_attrs, self.item_attrs[idx])
            final = (
                config.TEXT_SCORE_WEIGHT * float(text_scores[idx])
                + config.ATTRIBUTE_SCORE_WEIGHT * comparison["score"]
            )
            results.append((idx, float(text_scores[idx]), comparison, final))

        results.sort(key=lambda r: r[3], reverse=True)
        top = results[:top_k]

        matches = []
        for rank, (idx, text_score, comparison, final) in enumerate(top, start=1):
            row = self.df.iloc[idx]
            matches.append({
                "rank": rank,
                "description": row["display_description"],
                "clean_description": row["clean_description"],
                "unit": row["unit_norm"] or (str(row["unit"]) if pd.notna(row["unit"]) else ""),
                "quantity": None if pd.isna(row["quantity"]) else float(row["quantity"]),
                "rate": float(row["rate"]),
                "amount": None if pd.isna(row["amount"]) else float(row["amount"]),
                "currency": str(row["currency"]) if pd.notna(row["currency"]) else "",
                "category": _first_present(row, ("category", "section")),
                "source": str(row["source"]) if pd.notna(row["source"]) else "",
                "similarity_score": round(final, 4),
                "text_similarity": round(text_score, 4),
                "attribute_score": comparison["score"],
                "matched_attributes": comparison["matched"],
                "mismatched_attributes": comparison["mismatched"],
                "missing_attributes": comparison["missing"],
                "explanation": _explain(text_score, comparison),
            })

        return {"input_attributes": input_attrs, "matches": matches}


def _first_present(row: pd.Series, cols: tuple[str, ...]) -> str:
    for col in cols:
        val = row.get(col)
        if val is not None and pd.notna(val) and str(val).strip():
            return str(val)
    return ""


def _explain(text_score: float, comparison: dict) -> str:
    parts = []
    if comparison["matched"]:
        parts.append("matches on " + "; ".join(comparison["matched"]))
    if comparison["mismatched"]:
        parts.append("differs on " + "; ".join(comparison["mismatched"]))
    if comparison["missing"]:
        parts.append("unknown: " + "; ".join(comparison["missing"]))
    if not parts:
        parts.append("text similarity only, no comparable attributes")
    return f"Text similarity {text_score:.2f}; " + ". ".join(parts)
