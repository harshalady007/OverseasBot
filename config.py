"""Central configuration for the pricing bot.

Everything can be overridden with environment variables so no code
changes are needed when moving between machines.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Excel dataset path resolution order:
#   1. PRICING_BOT_EXCEL environment variable
#   2. data/quotation_items.xlsx bundled next to this file
#   3. the original Windows path the dataset came from
_DEFAULT_WINDOWS_PATH = (
    r"C:\Harshal\Quotation Dataset\Turner & Townsend International Limited"
    r"\BS-QT-22-20_output\quotation_items.xlsx"
)


def get_excel_path() -> str:
    env_path = os.environ.get("PRICING_BOT_EXCEL")
    if env_path:
        return env_path
    bundled = PROJECT_ROOT / "data" / "quotation_items.xlsx"
    if bundled.exists():
        return str(bundled)
    return _DEFAULT_WINDOWS_PATH


# DeepSeek API (OpenAI-compatible chat completions endpoint)
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_API_URL = os.environ.get(
    "DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions"
)
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TIMEOUT_SECONDS = int(os.environ.get("DEEPSEEK_TIMEOUT", "90"))


def get_deepseek_api_key() -> str | None:
    """Return the API key or None. Never print or log the key itself."""
    key = os.environ.get(DEEPSEEK_API_KEY_ENV, "").strip()
    return key or None


# Similarity search tuning
TEXT_SCORE_WEIGHT = 0.55
ATTRIBUTE_SCORE_WEIGHT = 0.45
MIN_SIMILARITY_WARNING = 0.35  # below this, warn that matches are weak
DEFAULT_TOP_K = 5
