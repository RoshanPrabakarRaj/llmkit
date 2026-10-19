"""Tests for LLMClient. Written first: your job is to make them pass.

Run:  uv run pytest -v
"""

import pytest

from llmkit import LLMClient, PermanentError, ProviderResult, TransientError


class FakeProvider:
    """A stand-in provider that follows a script, so tests need no network.

    `script` is a list of outcomes, used in order. Each item is either a
    ProviderResult (success) or an Exception instance (it gets raised).
    The last item repeats if the script runs out.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []  # records (model, prompt, temperature) for each call

    def generate(self, prompt, *, model, temperature=None):
        self.calls.append((model, prompt, temperature))
        index = min(len(self.calls), len(self.script)) - 1
        outcome = self.script[index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def ok(text="hello", in_tokens=1000, out_tokens=500):
    return ProviderResult(text=text, input_tokens=in_tokens, output_tokens=out_tokens)


def make_client(provider, **kwargs):
    sleeps = []
    client = LLMClient(
        provider,
        "primary",
        sleep=sleeps.append,  # records waits instead of really sleeping
        base_delay_s=1.0,
        **kwargs,
    )
    return client, sleeps


# TODO 1 -------------------------------------------------------------
def test_returns_response_with_cost():
    provider = FakeProvider([ok()])
    client, _ = make_client(
        provider, price_per_m_input=0.30, price_per_m_output=2.50
    )
    response = client.generate("hi")
    assert response.text == "hello"
    assert response.model == "primary"
    assert response.input_tokens == 1000
    assert response.output_tokens == 500
    assert response.attempts == 1
    assert response.cached is False
    assert response.latency_s >= 0
    # 1000/1e6*0.30 + 500/1e6*2.50 = 0.0003 + 0.00125
    assert response.cost_usd == pytest.approx(0.00155)


# TODO 3 -------------------------------------------------------------
def test_retries_then_succeeds():
    provider = FakeProvider([TransientError("503"), TransientError("503"), ok()])
    client, sleeps = make_client(provider, max_retries=3)
    response = client.generate("hi")
    assert response.text == "hello"
    assert response.attempts == 3
    assert len(provider.calls) == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff: 1s then 2s


def test_gives_up_after_max_retries():
    provider = FakeProvider([TransientError("503")])  # fails forever
    client, sleeps = make_client(provider, max_retries=2)
    with pytest.raises(TransientError):
        client.generate("hi")
    assert len(provider.calls) == 3  # 1 try + 2 retries
    assert sleeps == [1.0, 2.0]


def test_permanent_error_is_not_retried():
    provider = FakeProvider([PermanentError("bad key")])
    client, sleeps = make_client(provider, max_retries=3)
    with pytest.raises(PermanentError):
        client.generate("hi")
    assert len(provider.calls) == 1
    assert sleeps == []


# TODO 4 -------------------------------------------------------------
def test_uses_fallback_model_when_primary_keeps_failing():
    # Primary fails on all 2 attempts (1 try + 1 retry), then fallback works.
    provider = FakeProvider(
        [TransientError("503"), TransientError("503"), ok("from fallback")]
    )
    client, _ = make_client(provider, max_retries=1, fallback_model="backup")
    response = client.generate("hi")
    assert response.text == "from fallback"
    assert response.model == "backup"
    assert [call[0] for call in provider.calls] == ["primary", "primary", "backup"]


def test_fallback_failure_raises():
    provider = FakeProvider([TransientError("503")])  # everything fails
    client, _ = make_client(provider, max_retries=0, fallback_model="backup")
    with pytest.raises(TransientError):
        client.generate("hi")
    assert [call[0] for call in provider.calls] == ["primary", "backup"]


# TODO 5 -------------------------------------------------------------
def test_cache_hit_skips_the_provider():
    provider = FakeProvider([ok("first"), ok("second")])
    client, _ = make_client(provider)
    first = client.generate("same prompt")
    second = client.generate("same prompt")
    assert len(provider.calls) == 1
    assert first.cached is False
    assert second.cached is True
    assert second.text == "first"
    assert second.cost_usd == 0.0


def test_cache_key_includes_prompt_and_temperature():
    provider = FakeProvider([ok("a"), ok("b"), ok("c")])
    client, _ = make_client(provider)
    client.generate("one")
    client.generate("two")
    client.generate("one", temperature=0.0)
    assert len(provider.calls) == 3
