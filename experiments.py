"""Week 1 experiments: how prompts and temperature change a model's output."""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash-lite"


def ask(prompt: str, temperature: float | None = None) -> str:
    config = None
    if temperature is not None:
        config = types.GenerateContentConfig(temperature=temperature)
    # Retry with exponential backoff: wait 2s, 4s, 8s, 16s between attempts.
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=prompt, config=config
            )
            return response.text.strip()
        except errors.ServerError as e:  # 5xx: the server's fault, worth retrying
            if attempt == 4:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  (server error {e.code}, retrying in {wait}s)")
            time.sleep(wait)


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# Experiment 1: same prompt, three runs (default temperature).
section("1. Same prompt, 3 runs")
for i in range(3):
    print(f"Run {i + 1}: {ask('Explain what a token is in one sentence.')}\n")

# Experiment 2: same question, different audiences.
section("2. Same question, different audiences")
for audience in ["a 10-year-old", "a software engineer"]:
    print(f"For {audience}: {ask(f'Explain what a token is to {audience}, in two sentences.')}\n")

# Experiment 3: temperature 0.0 vs 1.0 on a creative prompt.
section("3. Temperature 0.0 vs 1.0")
slogan = "Write a one-line slogan for a coffee shop."
for temp in [0.0, 1.0]:
    print(f"--- temperature={temp}")
    for i in range(3):
        print(f"  {i + 1}. {ask(slogan, temperature=temp)}")
