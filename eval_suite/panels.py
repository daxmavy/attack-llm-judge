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
ATTACK_PANEL = [
    JudgeConfig("llama-3.1-8b",    "meta-llama/llama-3.1-8b-instruct", "attack",  0.05, 0.08),
    JudgeConfig("qwen-2.5-7b",     "qwen/qwen-2.5-7b-instruct",        "attack",  0.04, 0.10),
    JudgeConfig("mistral-7b",      "mistralai/mistral-7b-instruct",    "attack",  0.055, 0.055),
    JudgeConfig("gemma-2-9b",      "google/gemma-2-9b-it",             "attack",  0.06, 0.06),
    JudgeConfig("phi-3.5-mini",    "microsoft/phi-3.5-mini-128k-instruct", "attack", 0.04, 0.04),
]

GOLD_PANEL = [
    JudgeConfig("claude-sonnet-4.6", "anthropic/claude-sonnet-4.5",    "gold", 3.00, 15.00),
    JudgeConfig("gemini-2.5-pro",    "google/gemini-2.5-pro",          "gold", 1.25, 10.00),
    JudgeConfig("gpt-5-mini",        "openai/gpt-5-mini",              "gold", 0.25, 2.00),
    JudgeConfig("deepseek-v3",       "deepseek/deepseek-chat",         "gold", 0.27, 1.10),
    JudgeConfig("llama-3.1-405b",    "meta-llama/llama-3.1-405b-instruct", "gold", 0.80, 0.80),
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
