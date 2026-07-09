"""Export the cleaned quotation dataset to api/dataset.json.

The Vercel serverless function reads this prebuilt JSON instead of the
Excel file, so it does not need pandas/openpyxl at runtime. Re-run this
script whenever the Excel dataset changes, then redeploy:

    python scripts/build_index.py
"""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from cleaner import clean_dataset, dataset_records  # noqa: E402
from data_loader import load_dataset  # noqa: E402


def main() -> None:
    excel_path = config.get_excel_path()
    raw, mapping, sheet = load_dataset(excel_path)
    records = dataset_records(clean_dataset(raw))
    out_path = Path(__file__).resolve().parent.parent / "api" / "dataset.json"
    out_path.parent.mkdir(exist_ok=True)
    payload = json.dumps({"sheet": sheet, "column_mapping": mapping,
                          "raw_rows": len(raw), "rows": records},
                         ensure_ascii=False)
    out_path.write_text(payload, encoding="utf-8")
    # gzipped copy: small enough to ship inside a serverless bundle/payload
    gz_path = out_path.with_suffix(".json.gz")
    gz_path.write_bytes(gzip.compress(payload.encode("utf-8"), 9))
    print(f"Wrote {len(records)} usable rows (of {len(raw)} raw, "
          f"sheet {sheet!r}) to {out_path} and {gz_path}")


if __name__ == "__main__":
    main()
