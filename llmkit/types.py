"""Plain data objects passed around the library."""

from dataclasses import dataclass


@dataclass
class ProviderResult:
    """What a provider returns for one successful call."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    """What LLMClient.generate() returns to the caller."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    cached: bool = False
    attempts: int = 1
