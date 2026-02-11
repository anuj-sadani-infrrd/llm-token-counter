"""Provider/model pricing for cost estimation."""

# USD per 1M tokens (input) - fallback when tokencost lacks the model
DEFAULT_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4.1": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 1.10, "output": 4.40},
    "gemini-1.5-flash": {"input": 0.08, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


def get_input_price_per_million(model: str) -> float:
    """Get input price per 1M tokens for a model. Returns 0 if unknown."""
    key = model.lower().split("/")[-1]
    if key in DEFAULT_PRICING:
        return DEFAULT_PRICING[key]["input"]

    # Try tokencost for 400+ models
    try:
        from tokencost.costs import calculate_cost_by_tokens

        cost = calculate_cost_by_tokens(1_000_000, model, "input")
        return float(cost)
    except (KeyError, ImportError):
        pass

    return 0.0
