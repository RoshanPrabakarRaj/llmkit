"""LLMClient: wraps a provider with retries, a fallback model, a cache and cost.

Read this file top to bottom: each method is small and commented with the WHY.
Run the tests with:   uv run pytest -v
"""

import time
from collections.abc import Callable

from llmkit.errors import PermanentError, TransientError  # noqa: F401
from llmkit.providers.base import Provider
from llmkit.types import LLMResponse, ProviderResult


class LLMClient:
    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        fallback_model: str | None = None,
        max_retries: int = 3,
        base_delay_s: float = 1.0,
        price_per_m_input: float = 0.0,
        price_per_m_output: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Store the settings.

        max_retries = retries AFTER the first try, so 3 means up to 4 attempts.
        `sleep` is injectable so tests can run instantly without really waiting.
        Prices are dollars per 1 million tokens.
        """
        self.provider = provider
        self.model = model
        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.price_per_m_input = price_per_m_input
        self.price_per_m_output = price_per_m_output
        self.sleep = sleep
        self._cache: dict = {}

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        """Dollars = tokens / 1,000,000 * price per million, for input and output."""
        total = (((input_tokens / 1_000_000) * self.price_per_m_input) + ((output_tokens / 1000000) * self.price_per_m_output))
        return total

    def _call_with_retries(
        self, prompt: str, model: str, temperature: float | None
    ) -> tuple[ProviderResult, int]:
        """Call the provider, retrying temporary failures with exponential backoff.

        Returns (result, number_of_attempts_used).
        """
        last_error: TransientError | None = None

        # range(max_retries + 1) = the first try plus max_retries retries.
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                # Wait before each RETRY (never before the first try).
                # attempt 1 -> base * 1, attempt 2 -> base * 2, attempt 3 -> base * 4 ...
                # Doubling the wait gives a busy server time to recover.
                self.sleep(self.base_delay_s * 2 ** (attempt - 1))
            try:
                result = self.provider.generate(
                    prompt, model=model, temperature=temperature
                )
                return result, attempt + 1
            except TransientError as error:
                # Temporary problem: remember it and go around the loop again.
                # PermanentError is deliberately NOT caught here, so it escapes
                # immediately. Retrying a bad API key would never help.
                last_error = error

        # Every attempt failed. Surface the last error to the caller.
        assert last_error is not None
        raise last_error

    def generate(self, prompt: str, *, temperature: float | None = None) -> LLMResponse:
        """Answer a prompt, using the cache, retries and the fallback model."""
        # 1. Cache. The key must include everything that changes the answer:
        #    the model, the prompt and the temperature.
        key = (self.model, prompt, temperature)
        if key in self._cache:
            cached = self._cache[key]
            # A hit costs nothing and takes no time, and no API call is made.
            return LLMResponse(
                text=cached.text,
                model=cached.model,
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                latency_s=0.0,
                cost_usd=0.0,
                cached=True,
                attempts=0,
            )

        start = time.perf_counter()
        model = self.model

        # 2. Primary model, with retries.
        try:
            result, attempts = self._call_with_retries(prompt, model, temperature)
        except TransientError:
            # 3. The primary model kept failing. Use the backup if there is one.
            #    Only TransientError gets here: a PermanentError still propagates.
            if self.fallback_model is None:
                raise
            model = self.fallback_model
            result, attempts = self._call_with_retries(prompt, model, temperature)

        # 4. Build the response. `model` records which model REALLY answered.
        response = LLMResponse(
            text=result.text,
            model=model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_s=time.perf_counter() - start,
            cost_usd=self._cost(result.input_tokens, result.output_tokens),
            cached=False,
            attempts=attempts,
        )
        self._cache[key] = response
        return response
