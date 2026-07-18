"""Approximate per-model USD pricing for cost estimation.

Prices are USD per 1,000 tokens and are manually maintained -- treat
estimated costs as directional, not billing-accurate.
"""

# (prompt_price_per_1k, completion_price_per_1k)
_PRICING_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.010),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
}

_DEFAULT_PRICING = (0.0, 0.0)


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_price, completion_price = _PRICING_PER_1K.get(model, _DEFAULT_PRICING)
    return (prompt_tokens / 1000) * prompt_price + (completion_tokens / 1000) * completion_price
