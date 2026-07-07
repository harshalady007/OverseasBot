"""Tests for attribute extraction and hybrid similarity search."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attribute_extractor import compare_attributes, extract_attributes  # noqa: E402
from cleaner import clean_dataset, clean_text, normalise_unit  # noqa: E402
from similarity_search import SearchError, SimilaritySearch  # noqa: E402


def make_dataset() -> pd.DataFrame:
    """Small synthetic dataset covering the weak-match traps from the spec."""
    rows = [
        # good match: ss handrail, supply & install, per metre
        ("Supply and install stainless steel handrail 50mm dia, grade SS316",
         "m", 10, 120.0),
        # aluminium handrail (material mismatch)
        ("Supply and install aluminium handrail 50mm dia powder coated",
         "m", 10, 80.0),
        # supply-only handrail (scope mismatch)
        ("Supply only stainless steel handrail 40mm dia", "m", 5, 60.0),
        # stainless steel plate (item type mismatch)
        ("Stainless steel plate 3mm thick brushed finish", "m2", 20, 45.0),
        # generic steel item
        ("Mild steel angle 50x50x5mm galvanized", "kg", 100, 2.5),
        # unrelated item sharing only 'finish'
        ("Painting to internal walls, matte finish emulsion", "m2", 200, 4.0),
        # street furniture
        ("Outdoor bench galvanized steel powder coated with iroko wood, "
         "L 1800 x W 530 x H 530 mm", "nos", 4, 380.0),
    ]
    return pd.DataFrame([{
        "item_name": None, "description": d, "specification": None,
        "unit": u, "quantity": q, "rate": r, "amount": q * r,
        "currency": "USD", "category": None, "section": None,
        "location": None, "remarks": None, "source": "test",
        "source_evidence": None,
    } for d, u, q, r in rows])


@pytest.fixture(scope="module")
def search():
    return SimilaritySearch(clean_dataset(make_dataset()))


def test_clean_text_preserves_specs():
    out = clean_text("Supply & Install 50 mm DIA S.S.304 Handrail\nBrushed!!")
    assert "50mm" in out
    assert "ss304" in out
    assert "handrail" in out
    assert "\n" not in out


def test_unit_normalisation():
    assert normalise_unit("No's") == "nos"
    assert normalise_unit("PCS") == "nos"
    assert normalise_unit("Sqm") == "m2"
    assert normalise_unit(None) == ""


def test_extract_attributes_handrail():
    attrs = extract_attributes(
        "Supply and install 50mm diameter stainless steel handrail "
        "with brushed finish", unit="rm",
    )
    assert attrs["material"] == "stainless steel"
    assert attrs["diameter_mm"] == 50.0
    assert "brushed" in attrs["finish"]
    assert attrs["scope"] == "supply and install"
    assert "handrail" in attrs["item_type"]
    assert attrs["unit"] == "m"


def test_extract_dimensions_lxwxh():
    attrs = extract_attributes("Bench size L 1800 x W 530 x H 530 mm")
    assert attrs["dimensions_mm"] == [1800.0, 530.0, 530.0]


def test_compare_attributes_penalises_material_mismatch():
    inp = extract_attributes("stainless steel handrail 50mm dia brushed")
    same = extract_attributes("stainless steel handrail 50mm dia")
    diff = extract_attributes("aluminium handrail 50mm dia")
    assert compare_attributes(inp, same)["score"] > compare_attributes(inp, diff)["score"]
    assert any("material" in m for m in compare_attributes(inp, diff)["mismatched"])


def test_search_prefers_true_match(search):
    result = search.search(
        "Supply and install 50mm diameter stainless steel handrail "
        "with brushed finish", top_k=5,
    )
    top = result["matches"][0]
    assert "stainless steel handrail" in top["clean_description"]
    assert "supply and install" in top["clean_description"]
    # the unrelated painting row must not outrank real handrail items
    ranked = [m["clean_description"] for m in result["matches"]]
    painting_rank = next(
        (i for i, d in enumerate(ranked) if "painting" in d), len(ranked)
    )
    handrail_ranks = [i for i, d in enumerate(ranked) if "handrail" in d]
    assert handrail_ranks and painting_rank > max(handrail_ranks)


def test_search_scores_and_explanations(search):
    result = search.search("stainless steel handrail", top_k=3)
    for m in result["matches"]:
        assert 0.0 <= m["similarity_score"] <= 1.0
        assert m["explanation"]
        assert m["rate"] > 0


def test_search_empty_query_raises(search):
    with pytest.raises(SearchError):
        search.search("   ")
