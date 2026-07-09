# Quotation Pricing Bot

A local Python assistant that predicts a unit price for a new item or
service description by searching your historical quotation Excel dataset,
comparing practical pricing attributes (material, size, diameter,
thickness, finish, grade, scope, unit, category) like a real estimator,
and asking the DeepSeek API for a reasoned price prediction.

It does **not** match by text alone: every candidate is scored with a
hybrid of TF-IDF text similarity **and** structured attribute comparison,
so a "stainless steel handrail, supply and install" input prefers real
handrail items over an aluminium handrail, a supply-only item, or an
unrelated row that only shares the word "finish".

## Project structure

```text
OverseasBot/
├── app.py                  # Streamlit web interface (local use)
├── config.py               # paths, API settings, score weights
├── data_loader.py          # Excel loading + fuzzy column detection
├── cleaner.py              # row cleaning, text normalization, search_text
├── attribute_extractor.py  # material/size/finish/scope/... extraction + comparison
├── similarity_search.py    # pure-Python TF-IDF + cosine + attribute scoring
├── deepseek_pricing.py     # DeepSeek API call + statistical fallback
├── pricing_engine.py       # ties everything together (predict_price)
├── api/
│   ├── index.py            # FastAPI app for Vercel (HTML UI + /api/predict)
│   └── dataset.json        # prebuilt cleaned dataset for serverless use
├── scripts/
│   └── build_index.py      # regenerates api/dataset.json from the Excel
├── data/
│   └── quotation_items.xlsx  # bundled quotation dataset
├── vercel.json             # Vercel routing + function config
├── requirements.txt        # light API deps (what Vercel installs)
├── requirements-local.txt  # full local deps (Streamlit, pandas, tests)
├── README.md
└── tests/
    └── test_similarity.py
```

## Requirements

- Python 3.10 or newer (developed on 3.11)
- `requirements.txt` — API/serverless deps (fastapi, requests)
- `requirements-local.txt` — everything for local use (adds pandas,
  openpyxl, streamlit, uvicorn, pytest)

## Setup (Windows, cmd)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-local.txt
set DEEPSEEK_API_KEY=your_api_key_here
streamlit run app.py
```

## Setup (Windows, PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-local.txt
$env:DEEPSEEK_API_KEY="your_api_key_here"
streamlit run app.py
```

## Setup (macOS / Linux)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-local.txt
export DEEPSEEK_API_KEY=your_api_key_here
streamlit run app.py
```

## DeepSeek API key

The key is read from the `DEEPSEEK_API_KEY` environment variable only —
it is never hardcoded, printed or logged. **Without the key the app still
works**: it falls back to a statistical estimate (similarity-weighted
average blended with the median of the top matches) and clearly labels
the result as a fallback with Low/Medium confidence.

Optional overrides: `DEEPSEEK_MODEL` (default `deepseek-chat`),
`DEEPSEEK_API_URL`, `DEEPSEEK_TIMEOUT` (seconds).

## Excel file path

The dataset path is resolved in this order:

1. `PRICING_BOT_EXCEL` environment variable, e.g.

   ```bash
   set PRICING_BOT_EXCEL=C:\Harshal\Quotation Dataset\Turner & Townsend International Limited\BS-QT-22-20_output\quotation_items.xlsx
   ```

2. `data/quotation_items.xlsx` inside the project (bundled copy)
3. The original Windows path above, as a last resort

The loader does not assume exact column names. It scans every sheet,
maps likely headers (Description, Item Name, Specification, Unit, Qty,
Item Price / Unit Price / Rate, Amount/Total, Currency, Category,
Section, Location, Remarks/Notes, Source File...) to standard internal
fields, and picks the sheet with the best coverage. If a rate column is
missing but amount and quantity exist, rate is computed as
`amount / quantity` (skipping invalid quantities).

## Running the app locally

Streamlit (richest UI):

```bash
streamlit run app.py
```

FastAPI version (same one that runs on Vercel):

```bash
uvicorn api.index:app --reload
```

## Deploying to Vercel

Streamlit needs a long-running server, so the Vercel deployment uses the
FastAPI app in `api/index.py` instead. It serves an HTML interface at `/`
and a JSON API at `POST /api/predict`
(`{"description": "...", "top_k": 5}`), plus `GET /api/health`.

- The serverless function reads the prebuilt `api/dataset.json` instead
  of the Excel file. After changing the Excel dataset, regenerate it and
  redeploy: `python scripts/build_index.py`
- Root `requirements.txt` is deliberately light (fastapi, requests) — it
  is what Vercel installs. Local dev deps live in `requirements-local.txt`.
- Set `DEEPSEEK_API_KEY` in the Vercel dashboard (Project → Settings →
  Environment Variables) and redeploy to enable DeepSeek predictions;
  without it the deployment uses the statistical fallback.
- Note: the deployed URL is public by default and exposes your
  historical rates through the matches it returns. Enable Vercel
  Deployment Protection if that matters.

Then open the URL Streamlit prints (usually http://localhost:8501).
Enter a description, pick the number of top matches, and click
**Predict price**. You get:

- predicted unit price, currency, unit and confidence
- the estimator's reasoning, price basis and adjustments
- warnings when match quality is weak or the fallback was used
- a table of the top historical matches (rank, similarity, description,
  unit, qty, rate, amount, category, matched/mismatched attributes)
- a per-match comparison of your input vs the historical record

## Testing sample descriptions

Example input:

```text
Supply and install 50mm diameter stainless steel handrail with brushed finish
```

Example output (fallback mode, shape identical with DeepSeek):

```json
{
  "predicted_unit_price": 214.05,
  "currency": "USD",
  "unit": "unknown",
  "confidence": "Low",
  "reasoning": "Statistical fallback: similarity-weighted average blended with the median of the top 5 historical rates...",
  "price_basis": "#1 Bike rack ... @ 75; #2 ...",
  "adjustments": [],
  "warnings": ["DEEPSEEK_API_KEY is not set. ..."]
}
```

You can also use the engine from Python directly:

```python
from pricing_engine import PricingEngine
engine = PricingEngine()
result = engine.predict_price("Outdoor bench galvanized steel with iroko wood, L1800xW530xH530mm", top_k=5)
print(result["predicted_unit_price"], result["confidence"])
```

Run the automated tests:

```bash
python -m pytest tests/ -v
```

## Error handling

The app shows readable messages (instead of crashing) for: missing or
unreadable Excel file, no sheets, no usable description/price columns,
empty dataset after cleaning, empty or too-short input, no matches,
missing `DEEPSEEK_API_KEY`, DeepSeek API/network failures, and invalid
JSON from DeepSeek (all API failures drop to the statistical fallback).
Duplicate rows, invalid numbers, and zero/negative quantities are cleaned
out during loading.

## Known limitations and assumptions

- The bundled dataset is mostly **outdoor/street furniture** priced in
  USD per piece/set. Inputs from other trades (e.g. MEP, civil) will get
  weak matches and a warning; the prediction is then a rough guide only.
- Dataset "Item Price" is treated as the **unit rate**; the Summary
  sheet is ignored (it is an audit sheet, not row data).
- Attribute extraction is rule/regex based. It covers common
  construction materials, finishes, grades, scopes and units, but exotic
  phrasing may not be recognized; unrecognized attributes simply don't
  contribute to the attribute score.
- Dimensions are compared with a ±25% tolerance; "similar size" is a
  heuristic, not an engineering equivalence.
- Historical rates are used as-is: no inflation/date adjustment and no
  currency conversion.
- The DeepSeek prediction quality depends on the matches supplied; when
  all matches are weak the bot says so and lowers confidence.
