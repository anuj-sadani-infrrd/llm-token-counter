"""Cost estimation from token counts."""

from llm_tokens.pricing import get_input_price_per_million


def estimate_cost(tokens: int, model: str) -> float:
    """
    Estimate USD cost for a given token count (input/prompt tokens).
    Uses input pricing only; completion cost would need separate calculation.
    """
    price_per_million = get_input_price_per_million(model)
    if price_per_million <= 0:
        return 0.0
    return (tokens / 1_000_000) * price_per_million
