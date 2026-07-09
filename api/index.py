"""FastAPI version of the pricing bot for Vercel serverless deployment.

Serves a single-page HTML interface at / and a JSON API at /api/predict.
Reads the prebuilt api/dataset.json (see scripts/build_index.py) so the
function stays small — no pandas or Excel parsing at runtime.

Run locally with:  uvicorn api.index:app --reload
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import config  # noqa: E402
from deepseek_pricing import predict_price_with_deepseek  # noqa: E402
from similarity_search import SearchError, SimilaritySearch  # noqa: E402

app = FastAPI(title="Quotation Pricing Bot")

_DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
_DATASET_GZ_PATH = Path(__file__).resolve().parent / "dataset.json.gz"


def _read_dataset() -> dict:
    if _DATASET_PATH.exists():
        return json.loads(_DATASET_PATH.read_text(encoding="utf-8"))
    if _DATASET_GZ_PATH.exists():
        return json.loads(gzip.decompress(_DATASET_GZ_PATH.read_bytes()))
    raise FileNotFoundError


_load_error: str | None = None
_search: SimilaritySearch | None = None
_meta: dict = {}
try:
    _data = _read_dataset()
    _meta = {
        "sheet": _data.get("sheet"),
        "raw_rows": _data.get("raw_rows"),
        "usable_rows": len(_data.get("rows", [])),
    }
    _search = SimilaritySearch(_data["rows"])
except FileNotFoundError:
    _load_error = ("dataset.json not found. Run scripts/build_index.py "
                   "and redeploy.")
except (ValueError, KeyError, SearchError) as exc:
    _load_error = f"Could not load dataset.json: {exc}"


class PredictRequest(BaseModel):
    description: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=10)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok" if _search else "dataset_error",
        "dataset": _meta,
        "deepseek_key_configured": bool(config.get_deepseek_api_key()),
        "error": _load_error,
    }


@app.post("/api/predict")
def predict(req: PredictRequest) -> JSONResponse:
    if _search is None:
        raise HTTPException(status_code=503, detail=_load_error)
    description = req.description.strip()
    if len(description) < 3:
        raise HTTPException(
            status_code=400,
            detail="The description is too short to search. Add more detail "
                   "(item type, material, size, finish, scope).",
        )
    try:
        search_result = _search.search(description, top_k=req.top_k)
    except SearchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    matches = search_result["matches"]
    input_attrs = search_result["input_attributes"]
    if not matches:
        raise HTTPException(
            status_code=404,
            detail="No historical matches found for this description.",
        )

    warnings: list[str] = []
    best = max(m["similarity_score"] for m in matches)
    if best < config.MIN_SIMILARITY_WARNING:
        warnings.append(
            f"Best similarity score is only {best:.2f}. The dataset may not "
            "contain items like this; treat the prediction as a rough guide."
        )

    prediction = predict_price_with_deepseek(description, input_attrs, matches)
    warnings.extend(prediction.get("warnings", []))

    return JSONResponse({
        "input_description": description,
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
    })


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quotation Pricing Bot</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         margin: 0 auto; max-width: 1080px; padding: 24px; line-height: 1.5; }
  h1 { font-size: 1.5rem; margin-bottom: 4px; }
  .muted { opacity: .72; font-size: .9rem; }
  textarea { width: 100%; min-height: 90px; font: inherit; padding: 10px;
             border-radius: 8px; border: 1px solid #8886; }
  .row { display: flex; gap: 16px; align-items: center; margin: 12px 0;
         flex-wrap: wrap; }
  button { font: inherit; font-weight: 600; padding: 10px 22px; border: 0;
           border-radius: 8px; background: #2563eb; color: #fff; cursor: pointer; }
  button:disabled { opacity: .5; cursor: wait; }
  .cards { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
  .card { border: 1px solid #8884; border-radius: 10px; padding: 12px 18px;
          min-width: 180px; }
  .card .v { font-size: 1.3rem; font-weight: 700; }
  .warn { background: #f59e0b22; border: 1px solid #f59e0b66;
          border-radius: 8px; padding: 8px 12px; margin: 8px 0; }
  .error { background: #dc262622; border: 1px solid #dc262666;
           border-radius: 8px; padding: 8px 12px; margin: 8px 0; }
  .tablewrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: .85rem; }
  th, td { border: 1px solid #8884; padding: 6px 8px; text-align: left;
           vertical-align: top; }
  th { background: #8881; }
  details { margin: 8px 0; border: 1px solid #8884; border-radius: 8px;
            padding: 8px 12px; }
  code { background: #8882; padding: 1px 5px; border-radius: 4px; }
  #reasoning { white-space: pre-wrap; }
</style>
</head>
<body>
<h1>&#128176; Quotation Pricing Bot</h1>
<p class="muted">Enter a new item or service description. The bot finds the
most similar historical quotation items, compares material, size, finish,
scope and unit, and predicts a unit price.</p>

<textarea id="desc" placeholder="e.g. Outdoor bench in galvanized steel, powder coated, with iroko wood slats, L1800 x W530 x H530mm"></textarea>
<div class="row">
  <label>Top matches
    <input id="topk" type="number" min="1" max="10" value="5" style="width:60px">
  </label>
  <button id="go">Predict price</button>
  <span id="status" class="muted"></span>
</div>
<div id="out"></div>

<script>
const el = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

async function run() {
  const desc = el("desc").value.trim();
  const out = el("out");
  out.innerHTML = "";
  if (!desc) { out.innerHTML = '<div class="error">Please enter a description.</div>'; return; }
  el("go").disabled = true;
  el("status").textContent = "Searching and predicting…";
  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({description: desc, top_k: +el("topk").value || 5}),
    });
    const data = await resp.json();
    if (!resp.ok) {
      out.innerHTML = '<div class="error">' + esc(data.detail || "Request failed") + "</div>";
      return;
    }
    render(data, out);
  } catch (err) {
    out.innerHTML = '<div class="error">Network error: ' + esc(err.message) + "</div>";
  } finally {
    el("go").disabled = false;
    el("status").textContent = "";
  }
}

function render(d, out) {
  let html = "";
  for (const w of d.warnings) html += '<div class="warn">&#9888;&#65039; ' + esc(w) + "</div>";
  const price = (d.currency && d.currency !== "unknown" ? d.currency + " " : "")
    + Number(d.predicted_unit_price).toLocaleString(undefined, {minimumFractionDigits: 2});
  const unit = d.unit && d.unit !== "unknown" ? d.unit : "unit";
  html += '<div class="cards">'
    + card("Predicted unit price", price + " / " + esc(unit))
    + card("Confidence", esc(d.confidence))
    + card("Source", d.fallback_used ? "Statistical fallback" : "DeepSeek estimator")
    + "</div>";
  html += "<h3>Explanation</h3><p id='reasoning'>" + esc(d.reasoning) + "</p>";
  if (d.price_basis) html += "<p><b>Price basis:</b> " + esc(d.price_basis) + "</p>";
  if (d.adjustments.length) html += "<p><b>Adjustments:</b></p><ul>"
    + d.adjustments.map(a => "<li>" + esc(a) + "</li>").join("") + "</ul>";
  html += "<details><summary>Attributes extracted from your input</summary><pre>"
    + esc(JSON.stringify(d.input_attributes, null, 2)) + "</pre></details>";
  html += "<h3>Top " + d.matches.length + " similar historical items</h3>"
    + '<div class="tablewrap"><table><tr><th>Rank</th><th>Similarity</th>'
    + "<th>Description</th><th>Unit</th><th>Qty</th><th>Rate</th><th>Amount</th>"
    + "<th>Category</th><th>Matched</th><th>Mismatched</th></tr>";
  for (const m of d.matches) {
    html += "<tr><td>" + m.rank + "</td>"
      + "<td>" + m.similarity_score.toFixed(3) + "<br><span class='muted'>text "
      + m.text_similarity.toFixed(2) + " / attr " + m.attribute_score.toFixed(2) + "</span></td>"
      + "<td>" + esc(m.description) + "</td>"
      + "<td>" + esc(m.unit) + "</td>"
      + "<td>" + (m.quantity ?? "—") + "</td>"
      + "<td>" + m.rate + (m.currency ? " " + esc(m.currency) : "") + "</td>"
      + "<td>" + (m.amount ?? "—") + "</td>"
      + "<td>" + esc(m.category) + "</td>"
      + "<td>" + esc(m.matched_attributes.join("; ")) + "</td>"
      + "<td>" + esc(m.mismatched_attributes.join("; ")) + "</td></tr>";
  }
  html += "</table></div>";
  for (const m of d.matches) {
    html += "<details><summary>#" + m.rank + " (score "
      + m.similarity_score.toFixed(2) + ") — " + esc(m.description.slice(0, 110))
      + "</summary>"
      + "<p><b>Matched:</b> " + (esc(m.matched_attributes.join("; ")) || "none") + "</p>"
      + "<p><b>Mismatched:</b> " + (esc(m.mismatched_attributes.join("; ")) || "none") + "</p>"
      + "<p><b>Not stated in match:</b> " + (esc(m.missing_attributes.join("; ")) || "none") + "</p>"
      + "<p class='muted'>" + esc(m.explanation) + "</p>"
      + (m.source ? "<p class='muted'>Source: " + esc(m.source) + "</p>" : "")
      + "</details>";
  }
  out.innerHTML = html;
}

const card = (label, value) =>
  '<div class="card"><div class="muted">' + label + '</div><div class="v">' + value + "</div></div>";

el("go").addEventListener("click", run);
el("desc").addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return _PAGE
