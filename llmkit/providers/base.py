"""The interface every provider must follow.

A provider is any object with a `generate` method like this one. It makes ONE
call and returns a ProviderResult. It does not retry; the client does that.

It must raise TransientError for temporary failures (for example a 503) and
PermanentError for failures that retrying cannot fix (for example a bad key).
"""

from typing import Protocol

from llmkit.types import ProviderResult


class Provider(Protocol):
    def generate(
        self, prompt: str, *, model: str, temperature: float | None = None
    ) -> ProviderResult: ...
