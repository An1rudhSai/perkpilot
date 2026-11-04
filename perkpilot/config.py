"""Configuration, read from environment variables."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TERMS_DIR = Path(__file__).parent / "terms"
DATA_DIR = Path(os.environ.get("PERKPILOT_DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.environ.get("PERKPILOT_DB", DATA_DIR / "perkpilot.sqlite"))

# The synthetic transaction set ends on this date, so "today" is pinned to it by default.
AS_OF = date.fromisoformat(os.environ.get("PERKPILOT_AS_OF", "2026-11-04"))

# card id -> (display name, clause prefix)
CARDS: dict[str, tuple[str, str]] = {
    "amex_gold": ("Amex Gold", "AG"),
    "amex_platinum": ("Amex Platinum", "AP"),
    "sapphire_reserve": ("Sapphire Reserve", "SR"),
    "freedom_unlimited": ("Freedom Unlimited", "FU"),
}

# The no-fee card used as the baseline in annual-fee calculations.
BASELINE_CARD = "freedom_unlimited"

# Cents per point. An assumption, not something the terms state; override with PERKPILOT_CPP_<CARD>.
POINT_VALUE_CENTS: dict[str, float] = {
    card: float(os.environ.get(f"PERKPILOT_CPP_{card.upper()}", "1.0")) for card in CARDS
}


def llm_settings() -> dict[str, str | None]:
    """LLM mode is 'offline' (scripted, no network) unless an API key is set or the mode is forced."""
    key = os.environ.get("PERKPILOT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    mode = os.environ.get("PERKPILOT_LLM_MODE") or ("openai" if key else "offline")
    return {
        "mode": mode,
        "api_key": key,
        "base_url": os.environ.get("PERKPILOT_LLM_BASE_URL", "https://api.openai.com/v1"),
        "model": os.environ.get("PERKPILOT_LLM_MODEL", "gpt-4o-mini"),
        "embed_model": os.environ.get("PERKPILOT_EMBED_MODEL"),
    }
