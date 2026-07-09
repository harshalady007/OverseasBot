"""Hybrid similarity search: TF-IDF cosine similarity over cleaned text,
adjusted by structured attribute comparison.

The TF-IDF implementation is pure Python (unigrams + bigrams, sublinear
tf, smoothed idf, l2-normalised vectors — the same formulation
scikit-learn uses). The dataset is small (hundreds of rows), so this
needs no heavy numeric dependencies and also runs inside a serverless
function.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import config
from attribute_extractor import compare_attributes, extract_attributes
from cleaner import clean_text

_TOKEN_RE = re.compile(r"\b\w[\w.]*\b")  # keeps tokens like 50mm, ss304


class SearchError(Exception):
    """Raised when the search cannot run (empty input/index)."""


def _tokenize(text: str) -> list[str]:
    unigrams = _TOKEN_RE.findall(text)
    bigrams = [f"{a} {b}" for a, b in zip(unigrams, unigrams[1:])]
    return unigrams + bigrams


class _Tfidf:
    """Minimal TF-IDF vectorizer + cosine similarity over sparse dicts."""

    def __init__(self, docs: list[str]):
        doc_tokens = [_tokenize(d) for d in docs]
        n_docs = len(docs)
        doc_freq: Counter[str] = Counter()
        for tokens in doc_tokens:
            doc_freq.update(set(tokens))
        # smoothed idf, as in sklearn: ln((1+n)/(1+df)) + 1
        self.idf = {
            term: math.log((1 + n_docs) / (1 + df)) + 1.0
            for term, df in doc_freq.items()
        }
        self.doc_vectors = [self._vectorize(tokens) for tokens in doc_tokens]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        vec: dict[str, float] = {}
        for term, count in Counter(tokens).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            vec[term] = (1.0 + math.log(count)) * idf  # sublinear tf
        norm = math.sqrt(sum(w * w for w in vec.values()))
        if norm > 0:
            vec = {t: w / norm for t, w in vec.items()}
        return vec

    def similarities(self, query: str) -> list[float]:
        q = self._vectorize(_tokenize(query))
        scores = []
        for doc in self.doc_vectors:
            small, large = (q, doc) if len(q) <= len(doc) else (doc, q)
            scores.append(sum(w * large.get(t, 0.0) for t, w in small.items()))
        return scores


class SimilaritySearch:
    """Builds a TF-IDF index over cleaned dataset records once, then
    answers queries with hybrid text + attribute scores.

    ``rows`` are plain dicts as produced by cleaner.dataset_records():
    display_description, clean_description, search_text, unit, unit_norm,
    quantity, rate, amount, currency, category, section, source.
    """

    def __init__(self, rows: list[dict]):
        if not rows:
            raise SearchError("Cannot build search index: dataset is empty.")
        self.rows = rows
        self.tfidf = _Tfidf([r["search_text"] for r in rows])
        self.item_attrs: list[dict] = [
            extract_attributes(r["search_text"], r.get("unit_norm") or None)
            for r in rows
        ]

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> dict[str, Any]:
        """Return input attributes and the ranked top matches."""
        cleaned = clean_text(query)
        if not cleaned:
            raise SearchError("Input description is empty.")
        input_attrs = extract_attributes(query)

        text_scores = self.tfidf.similarities(cleaned)

        # attribute-score only the text top candidates (cheap for small data,
        # avoids attribute work on obvious non-matches)
        candidate_count = min(len(self.rows), max(top_k * 10, 50))
        candidates = sorted(
            range(len(self.rows)), key=lambda i: text_scores[i], reverse=True
        )[:candidate_count]

        results = []
        for idx in candidates:
            comparison = compare_attributes(input_attrs, self.item_attrs[idx])
            final = (
                config.TEXT_SCORE_WEIGHT * text_scores[idx]
                + config.ATTRIBUTE_SCORE_WEIGHT * comparison["score"]
            )
            results.append((idx, text_scores[idx], comparison, final))

        results.sort(key=lambda r: r[3], reverse=True)
        top = results[:top_k]

        matches = []
        for rank, (idx, text_score, comparison, final) in enumerate(top, start=1):
            row = self.rows[idx]
            matches.append({
                "rank": rank,
                "description": row["display_description"],
                "clean_description": row["clean_description"],
                "unit": row.get("unit_norm") or row.get("unit") or "",
                "quantity": row.get("quantity"),
                "rate": float(row["rate"]),
                "amount": row.get("amount"),
                "currency": row.get("currency") or "",
                "category": row.get("category") or row.get("section") or "",
                "source": row.get("source") or "",
                "similarity_score": round(final, 4),
                "text_similarity": round(text_score, 4),
                "attribute_score": comparison["score"],
                "matched_attributes": comparison["matched"],
                "mismatched_attributes": comparison["mismatched"],
                "missing_attributes": comparison["missing"],
                "explanation": _explain(text_score, comparison),
            })

        return {"input_attributes": input_attrs, "matches": matches}


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
