"""Token prices, USD per million tokens (input, output).

Claude API list prices as of 2026-09. Amazon Bedrock prices differ by region and
are set by AWS: check the Bedrock pricing page and override with
STUDY_RAG_PRICE_IN / STUDY_RAG_PRICE_OUT if you deploy there.
"""

from __future__ import annotations

import os

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def cost_usd(model: str, usage: dict[str, int] | None) -> float | None:
    """Cost of one call, or None when usage or price is unknown (never guess)."""
    if not usage:
        return None
    if "STUDY_RAG_PRICE_IN" in os.environ:
        price_in = float(os.environ["STUDY_RAG_PRICE_IN"])
        price_out = float(os.environ.get("STUDY_RAG_PRICE_OUT", "0"))
    else:
        prices = PRICES_PER_MTOK.get(model.removeprefix("anthropic."))
        if prices is None:
            return None
        price_in, price_out = prices
    total = (
        usage.get("input_tokens", 0) * price_in + usage.get("output_tokens", 0) * price_out
    ) / 1_000_000
    return round(total, 6)
