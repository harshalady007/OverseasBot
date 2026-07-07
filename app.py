"""Streamlit web interface for the pricing bot.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from cleaner import CleaningError
from data_loader import DataLoadError
from pricing_engine import PricingEngine, PricingEngineError

st.set_page_config(page_title="Quotation Pricing Bot", page_icon="💰", layout="wide")


@st.cache_resource(show_spinner="Loading and indexing the quotation dataset...")
def get_engine() -> PricingEngine:
    return PricingEngine()


st.title("💰 Quotation Pricing Bot")
st.caption(
    "Enter a new item or service description. The bot finds the most "
    "similar historical quotation items, compares materials, sizes, "
    "finishes, scope and units, and predicts a unit price."
)

try:
    engine = get_engine()
except (DataLoadError, CleaningError) as exc:
    st.error(f"Could not load the quotation dataset: {exc}")
    st.stop()
except Exception as exc:  # unexpected, still avoid a raw traceback
    st.error(f"Unexpected error while loading the dataset: {exc}")
    st.stop()

summary = engine.dataset_summary()
with st.sidebar:
    st.header("Dataset")
    st.write(f"**File:** `{summary['excel_path']}`")
    st.write(f"**Sheet:** {summary['sheet']}")
    st.write(f"**Usable rows:** {summary['usable_rows']} of {summary['raw_rows']}")
    if summary["currencies"]:
        st.write(f"**Currencies:** {', '.join(summary['currencies'])}")
    if not config.get_deepseek_api_key():
        st.warning(
            "DEEPSEEK_API_KEY is not set. Predictions will use the "
            "statistical fallback (weighted average / median of matches). "
            "Set the environment variable and restart to enable "
            "estimator-quality predictions."
        )
    with st.expander("Detected columns"):
        st.json(summary["column_mapping"])

description = st.text_area(
    "Item / service description",
    placeholder="e.g. Supply and install 50mm diameter stainless steel "
                "handrail with brushed finish",
    height=100,
)
col_a, col_b = st.columns([1, 3])
with col_a:
    top_k = st.slider("Top matches", min_value=1, max_value=10,
                      value=config.DEFAULT_TOP_K)
run = st.button("Predict price", type="primary")

if run:
    try:
        with st.spinner("Searching historical items and predicting price..."):
            result = engine.predict_price(description, top_k=top_k)
    except PricingEngineError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error during prediction: {exc}")
        st.stop()

    for warning in result["warnings"]:
        st.warning(warning)

    m1, m2, m3 = st.columns(3)
    price_text = f"{result['predicted_unit_price']:,.2f}"
    if result["currency"] not in ("", "unknown"):
        price_text = f"{result['currency']} {price_text}"
    unit = result["unit"] if result["unit"] not in ("", "unknown") else "unit"
    m1.metric("Predicted unit price", f"{price_text} / {unit}")
    m2.metric("Confidence", result["confidence"])
    m3.metric("Source", "Statistical fallback" if result["fallback_used"]
              else "DeepSeek estimator")

    st.subheader("Explanation")
    st.write(result["reasoning"])
    if result["price_basis"]:
        st.write(f"**Price basis:** {result['price_basis']}")
    if result["adjustments"]:
        st.write("**Adjustments applied:**")
        for adj in result["adjustments"]:
            st.write(f"- {adj}")

    with st.expander("Attributes extracted from your input"):
        st.json(result["input_attributes"])

    st.subheader(f"Top {len(result['matches'])} similar historical items")
    table = pd.DataFrame([{
        "Rank": m["rank"],
        "Similarity": m["similarity_score"],
        "Text sim.": m["text_similarity"],
        "Attr. score": m["attribute_score"],
        "Description": m["description"],
        "Unit": m["unit"],
        "Qty": m["quantity"],
        "Rate": m["rate"],
        "Amount": m["amount"],
        "Category/Section": m["category"],
        "Matched": "; ".join(m["matched_attributes"]),
        "Mismatched": "; ".join(m["mismatched_attributes"]),
    } for m in result["matches"]])
    st.dataframe(table, width="stretch", hide_index=True)

    st.subheader("Input vs match comparison")
    for m in result["matches"]:
        with st.expander(
            f"#{m['rank']} (score {m['similarity_score']:.2f}) — "
            f"{m['description'][:110]}"
        ):
            left, right = st.columns(2)
            with left:
                st.markdown("**Matched attributes**")
                st.write("\n".join(f"- {a}" for a in m["matched_attributes"])
                         or "none")
                st.markdown("**Mismatched attributes**")
                st.write("\n".join(f"- {a}" for a in m["mismatched_attributes"])
                         or "none")
                st.markdown("**Not stated in match**")
                st.write("\n".join(f"- {a}" for a in m["missing_attributes"])
                         or "none")
            with right:
                st.markdown("**Historical record**")
                st.write(f"Unit: {m['unit'] or '—'}")
                st.write(f"Quantity: {m['quantity'] if m['quantity'] is not None else '—'}")
                st.write(f"Rate: {m['rate']:g} {m['currency']}")
                if m["amount"] is not None:
                    st.write(f"Amount: {m['amount']:g} {m['currency']}")
                if m["source"]:
                    st.write(f"Source: {m['source']}")
            st.caption(m["explanation"])
