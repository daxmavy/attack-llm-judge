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
# Minimum-run design (MODELS.md, 2026-04-17): 2 in-panel judges used for both
# training signal (GRPO reward = mean of the two) and evaluation, plus 1
# held-out judge never seen during training. Same identities regardless of
# whether served via OpenRouter or locally via HF transformers.
ATTACK_PANEL = [
    JudgeConfig("judge_qwen7b",  "qwen/qwen-2.5-7b-instruct",          "attack", 0.04, 0.10),
    JudgeConfig("judge_llama8b", "meta-llama/llama-3.1-8b-instruct",   "attack", 0.02, 0.05),
]

HELD_OUT_PANEL = [
    JudgeConfig("judge_gemma9b", "google/gemma-2-9b-it",               "held_out", 0.06, 0.06),
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


ALL_JUDGES = ATTACK_PANEL + HELD_OUT_PANEL + GOLD_PANEL


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
    if name == "held_out":
        return list(HELD_OUT_PANEL)
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
