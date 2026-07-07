"""Predict a unit price with the DeepSeek API, acting as a quotation
estimator. Falls back to a statistical estimate from the historical
matches when the API is unavailable or fails.

The API key is read from the DEEPSEEK_API_KEY environment variable and is
never logged or included in returned data.
"""
from __future__ import annotations

import json
import statistics
from typing import Any

import requests

import config

SYSTEM_PROMPT = (
    "You are an experienced quotation estimator for construction, fit-out "
    "and outdoor/street furniture works. You are given a new item to price "
    "and a set of similar historical quotation items with their unit rates. "
    "Compare the new item to each historical item on material, size, "
    "thickness, diameter, finish, grade, scope of work, unit and category. "
    "Decide whether the new item should be priced similar to, higher than, "
    "or lower than the historical rates, and by how much. Higher for better "
    "material, larger size, thicker sections, better finish, higher grade, "
    "installation included or more complex work; lower for the opposite. "
    "Respond ONLY with a JSON object with exactly these keys: "
    "predicted_unit_price (number), currency (string), unit (string), "
    "confidence (\"High\", \"Medium\" or \"Low\"), reasoning (string), "
    "price_basis (string naming which historical items mattered most), "
    "adjustments (array of strings), warnings (array of strings)."
)


class DeepSeekError(Exception):
    """Raised internally on API problems; callers get a fallback result."""


def _build_user_prompt(input_description: str, input_attrs: dict,
                       matches: list[dict], weak: bool) -> str:
    lines = [
        "NEW ITEM TO PRICE:",
        input_description.strip(),
        "",
        "EXTRACTED ATTRIBUTES OF NEW ITEM:",
        json.dumps(input_attrs, ensure_ascii=False),
        "",
        "TOP SIMILAR HISTORICAL ITEMS:",
    ]
    for m in matches:
        lines.append(json.dumps({
            "rank": m["rank"],
            "description": m["description"][:500],
            "unit": m["unit"] or "unknown",
            "quantity": m["quantity"],
            "unit_rate": m["rate"],
            "currency": m["currency"] or "unknown",
            "category": m["category"],
            "similarity_score": m["similarity_score"],
            "matched_attributes": m["matched_attributes"],
            "mismatched_attributes": m["mismatched_attributes"],
            "missing_attributes": m["missing_attributes"],
        }, ensure_ascii=False))
    if weak:
        lines += ["", "WARNING: all matches have low similarity to the new "
                      "item. Be cautious and lower your confidence."]
    lines += ["", "Estimate the unit price for the NEW ITEM. Apply explicit "
                  "adjustments relative to the historical rates and explain "
                  "them. Return only the JSON object."]
    return "\n".join(lines)


def _call_deepseek(api_key: str, user_prompt: str) -> dict:
    resp = requests.post(
        config.DEEPSEEK_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        },
        timeout=config.DEEPSEEK_TIMEOUT_SECONDS,
    )
    if resp.status_code != 200:
        # do not echo the response body wholesale; it can be large
        raise DeepSeekError(
            f"DeepSeek API returned HTTP {resp.status_code}: "
            f"{resp.text[:300]}"
        )
    try:
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, ValueError) as exc:
        raise DeepSeekError(f"Invalid JSON in DeepSeek response: {exc}") from exc
    if not isinstance(parsed, dict) or "predicted_unit_price" not in parsed:
        raise DeepSeekError("DeepSeek JSON missing predicted_unit_price.")
    try:
        parsed["predicted_unit_price"] = float(parsed["predicted_unit_price"])
    except (TypeError, ValueError) as exc:
        raise DeepSeekError("predicted_unit_price is not numeric.") from exc
    return parsed


def _normalise_result(parsed: dict) -> dict:
    """Force the response into the documented schema with safe defaults."""
    conf = str(parsed.get("confidence", "Medium")).title()
    if conf not in ("High", "Medium", "Low"):
        conf = "Medium"
    return {
        "predicted_unit_price": round(float(parsed["predicted_unit_price"]), 2),
        "currency": str(parsed.get("currency", "unknown")) or "unknown",
        "unit": str(parsed.get("unit", "unknown")) or "unknown",
        "confidence": conf,
        "reasoning": str(parsed.get("reasoning", "")),
        "price_basis": str(parsed.get("price_basis", "")),
        "adjustments": [str(a) for a in parsed.get("adjustments", []) or []],
        "warnings": [str(w) for w in parsed.get("warnings", []) or []],
    }


def fallback_estimate(matches: list[dict], reason: str) -> dict:
    """Similarity-weighted average of the top matches, cross-checked with
    the median. Used when DeepSeek is unavailable."""
    priced = [m for m in matches if m.get("rate")]
    if not priced:
        return {
            "predicted_unit_price": 0.0,
            "currency": "unknown",
            "unit": "unknown",
            "confidence": "Low",
            "reasoning": "No priced historical matches available.",
            "price_basis": "none",
            "adjustments": [],
            "warnings": [reason, "No fallback estimate possible."],
        }
    weights = [max(m["similarity_score"], 0.01) for m in priced]
    weighted = sum(m["rate"] * w for m, w in zip(priced, weights)) / sum(weights)
    median = statistics.median(m["rate"] for m in priced)
    estimate = (weighted + median) / 2
    best = max(m["similarity_score"] for m in priced)
    confidence = "Medium" if best >= 0.6 else "Low"
    currencies = {m["currency"] for m in priced if m["currency"]}
    units = {m["unit"] for m in priced if m["unit"]}
    return {
        "predicted_unit_price": round(estimate, 2),
        "currency": currencies.pop() if len(currencies) == 1 else "unknown",
        "unit": units.pop() if len(units) == 1 else "unknown",
        "confidence": confidence,
        "reasoning": (
            "Statistical fallback: similarity-weighted average "
            f"({weighted:.2f}) blended with the median ({median:.2f}) of the "
            f"top {len(priced)} historical rates. No estimator judgement was "
            "applied to attribute differences."
        ),
        "price_basis": "; ".join(
            f"#{m['rank']} {m['description'][:80]} @ {m['rate']:g}" for m in priced[:3]
        ),
        "adjustments": [],
        "warnings": [reason],
    }


def predict_price_with_deepseek(input_description: str, input_attrs: dict,
                                matches: list[dict]) -> dict[str, Any]:
    """Main entry: returns the schema dict plus 'fallback_used' (bool) and
    'fallback_reason' (str or None)."""
    weak = not matches or max(
        (m["similarity_score"] for m in matches), default=0
    ) < config.MIN_SIMILARITY_WARNING

    api_key = config.get_deepseek_api_key()
    if not api_key:
        result = fallback_estimate(
            matches,
            "DEEPSEEK_API_KEY is not set. Set it in your environment to "
            "enable estimator-quality predictions "
            "(e.g. set DEEPSEEK_API_KEY=your_key on Windows).",
        )
        result["fallback_used"] = True
        result["fallback_reason"] = "missing API key"
        return result

    prompt = _build_user_prompt(input_description, input_attrs, matches, weak)
    try:
        parsed = _call_deepseek(api_key, prompt)
        result = _normalise_result(parsed)
        if weak and "Low" not in result["confidence"]:
            result["warnings"].append(
                "Match quality is weak; treat this prediction with caution."
            )
        result["fallback_used"] = False
        result["fallback_reason"] = None
        return result
    except (DeepSeekError, requests.RequestException) as exc:
        result = fallback_estimate(matches, f"DeepSeek API failed: {exc}")
        result["fallback_used"] = True
        result["fallback_reason"] = str(exc)
        return result
