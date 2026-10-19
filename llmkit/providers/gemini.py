"""Gemini provider. TODO (Day 2): fill this in after the client tests pass.

Hints:
- Reuse what you wrote in experiments.py to make the call.
- Build a ProviderResult from the response text. Token counts are in
  response.usage_metadata (prompt_token_count, candidates_token_count).
- Catch google.genai.errors.ServerError (5xx) and raise TransientError.
- Catch other google.genai.errors.APIError (4xx) and raise PermanentError.
"""

from llmkit.types import ProviderResult


class GeminiProvider:
    def __init__(self, api_key: str) -> None:
        raise NotImplementedError("TODO: create a google.genai.Client")

    def generate(
        self, prompt: str, *, model: str, temperature: float | None = None
    ) -> ProviderResult:
        raise NotImplementedError("TODO: call the API and map errors")
