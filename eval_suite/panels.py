"""LLM-judge panels used by the evaluation suite.

Two panels, both routed through OpenRouter:

- **attack** (5, small, family-diverse). Cheap enough to call in inference
  loops at training time (e.g. during rewriter RL). Used for metric (1):
  absolute score under 5 judges + leave-one-out generalisation probe.
- **gold** (5, strong, disjoint). Used for metric (2): "true" quality
  estimate. Significantly more expensive per call.

Both panels are parameterised by criterion so clarity / informativeness
can be scored by the same code. Right now only "clarity" is in scope.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeConfig:
    slug: str                 # short name used everywhere (labels, DB)
    model_id: str             # OpenRouter model id
    panel: str                # "attack" | "gold"
    prompt_in_per_1m_usd: float
    prompt_out_per_1m_usd: float


# Prices are OpenRouter list prices (Mar-Apr 2026). If the real cost comes
# back different on the smoke test we update these once, not per-call.
# Slugs below have been verified live against OpenRouter's model list
# (Apr 2026). Some models the operator originally named are no longer
# hosted; substitutions noted inline.
ATTACK_PANEL = [
    JudgeConfig("llama-3.1-8b",    "meta-llama/llama-3.1-8b-instruct",  "attack", 0.02, 0.05),
    JudgeConfig("qwen-2.5-7b",     "qwen/qwen-2.5-7b-instruct",          "attack", 0.04, 0.10),
    # Ministral 8B 2512 replaces retired mistral-7b-instruct (current Mistral-family small-judge).
    JudgeConfig("ministral-8b",    "mistralai/ministral-8b-2512",        "attack", 0.15, 0.15),
    # Gemma 3 4B replaces retired gemma-2-9b-it (current Gemma-family small-judge on OpenRouter).
    JudgeConfig("gemma-3-4b",      "google/gemma-3-4b-it",               "attack", 0.04, 0.08),
    # Microsoft Phi series no longer on OpenRouter, and NVIDIA Nemotron returns empty content
    # under response_format=json_object. Grok-3-mini is the cleanest 5th-family small judge.
    JudgeConfig("grok-3-mini",      "x-ai/grok-3-mini",                   "attack", 0.30, 0.50),
]

GOLD_PANEL = [
    # Per operator (2026-04-17): swap sonnet-4.6 -> haiku-4.5 and gemini-2.5-pro ->
    # gemini-2.5-flash to fit the $100 budget. Gold evaluation is also restricted
    # to a stratified subsample of human-written paragraphs (+ their rewrites).
    JudgeConfig("claude-haiku-4.5",  "anthropic/claude-haiku-4.5",              "gold", 0.80,  4.00),
    JudgeConfig("gemini-2.5-flash",  "google/gemini-2.5-flash",                 "gold", 0.15,  0.60),
    JudgeConfig("gpt-5-mini",        "openai/gpt-5-mini",                       "gold", 0.25,  2.00),
    JudgeConfig("deepseek-v3.2",     "deepseek/deepseek-v3.2",                  "gold", 0.27,  1.10),
    JudgeConfig("llama-4-maverick",  "meta-llama/llama-4-maverick",             "gold", 0.15,  0.60),
]


ALL_JUDGES = ATTACK_PANEL + GOLD_PANEL


def by_slug(slug: str) -> JudgeConfig:
    for j in ALL_JUDGES:
        if j.slug == slug:
            return j
    raise KeyError(slug)


def panel(name: str) -> list[JudgeConfig]:
    if name == "attack":
        return list(ATTACK_PANEL)
    if name == "gold":
        return list(GOLD_PANEL)
    if name == "all":
        return list(ALL_JUDGES)
    raise ValueError(name)


def estimate_cost_usd(judges: list[JudgeConfig], n_calls: int,
                       avg_input_tokens: int = 700, avg_output_tokens: int = 60) -> float:
    total = 0.0
    for j in judges:
        per_call = (avg_input_tokens / 1e6) * j.prompt_in_per_1m_usd \
                 + (avg_output_tokens / 1e6) * j.prompt_out_per_1m_usd
        total += per_call * n_calls
    return total
