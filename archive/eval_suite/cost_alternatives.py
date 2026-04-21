"""Quick sanity-check script: which gold-panel substitutions fit the $100 limit?

Operator's stated gold panel (option ii):
- anthropic/claude-sonnet-4.6 @ $3 / $15 per 1M
- google/gemini-2.5-pro     @ $1.25 / $10 per 1M
- openai/gpt-5-mini          @ $0.25 / $2 per 1M
- deepseek/deepseek-v3       @ $0.27 / $1.10 per 1M
- meta-llama/llama-3.1-405b  @ $0.80 / $0.80 per 1M

Full corpus scoring scope (clarity only, 1 criterion):
- 4,503 writer originals + 4,503 × 10 method rewrites ≈ 49,533 paragraphs
  (plus 4,503 model + 1,002 edited originals if we benchmark them,
   but those can be a separate decision).

Print a table of candidate 5-judge gold panels with their projected
full-corpus cost, so we can pick the strongest one that fits $100.
"""
from __future__ import annotations

from dataclasses import dataclass


AVG_IN = 700
AVG_OUT = 60


@dataclass(frozen=True)
class Judge:
    slug: str
    in_per_1m: float
    out_per_1m: float

    def per_call(self) -> float:
        return (AVG_IN / 1e6) * self.in_per_1m + (AVG_OUT / 1e6) * self.out_per_1m


# Gold-panel candidates (April 2026 OpenRouter list prices; will refine from smoke-test
# numbers before committing spend).
CANDIDATES = {
    "sonnet-4.6":       Judge("anthropic/claude-sonnet-4.5",           3.00, 15.00),
    "opus-4.7":         Judge("anthropic/claude-opus-4.5",            15.00, 75.00),
    "haiku-4.5":        Judge("anthropic/claude-haiku-4.5",            0.80,  4.00),
    "gemini-2.5-pro":   Judge("google/gemini-2.5-pro",                 1.25, 10.00),
    "gemini-2.5-flash": Judge("google/gemini-2.5-flash",               0.15,  0.60),
    "gpt-5-mini":       Judge("openai/gpt-5-mini",                     0.25,  2.00),
    "deepseek-v3":      Judge("deepseek/deepseek-chat",                0.27,  1.10),
    "deepseek-r1":      Judge("deepseek/deepseek-r1",                  0.55,  2.19),
    "llama-3.1-405b":   Judge("meta-llama/llama-3.1-405b-instruct",    0.80,  0.80),
    "qwen-2.5-72b":     Judge("qwen/qwen-2.5-72b-instruct",            0.13,  0.40),
}


def cost_for_panel(panel_slugs: list[str], n_paragraphs: int) -> float:
    return sum(CANDIDATES[s].per_call() * n_paragraphs for s in panel_slugs)


def main():
    n = 4503 + 4503 * 10  # writers + 10 methods of rewrites
    print(f"Scope: {n} paragraphs × 1 criterion (clarity) × each panel.\n")
    PANELS = {
        "operator_original (ii)": ["sonnet-4.6", "gemini-2.5-pro", "gpt-5-mini", "deepseek-v3", "llama-3.1-405b"],
        "A_drop_sonnet":   ["haiku-4.5", "gemini-2.5-pro", "gpt-5-mini", "deepseek-v3", "llama-3.1-405b"],
        "B_drop_sonnet_and_pro": ["haiku-4.5", "gemini-2.5-flash", "gpt-5-mini", "deepseek-v3", "llama-3.1-405b"],
        "C_heavy_reasoning": ["sonnet-4.6", "gemini-2.5-pro", "deepseek-r1", "gpt-5-mini", "llama-3.1-405b"],
        "D_balanced_budget": ["sonnet-4.6", "gemini-2.5-flash", "gpt-5-mini", "deepseek-v3", "llama-3.1-405b"],
        "E_open_only":      ["deepseek-v3", "deepseek-r1", "llama-3.1-405b", "qwen-2.5-72b", "gemini-2.5-flash"],
    }
    rows = []
    for name, slugs in PANELS.items():
        total = cost_for_panel(slugs, n)
        per_pp = total / n * 5  # cost per paragraph across 5-judge panel
        rows.append((name, total, per_pp, slugs))
    rows.sort(key=lambda r: r[1])
    for name, total, per_pp, slugs in rows:
        print(f"{name:30s}  total=${total:7.2f}  per-paragraph-5-judges=${per_pp:.5f}")
        print("  members:", ", ".join(slugs))

    # What if we subsample by 5x (pick 1/5 of rewrites per method)?
    n_sub = 4503 + int(4503 * 10 * 0.2)
    print(f"\nSubsample (20% of rewrites) = {n_sub} paragraphs:")
    for name, slugs in PANELS.items():
        total = cost_for_panel(slugs, n_sub)
        print(f"  {name:30s}  total=${total:7.2f}")


if __name__ == "__main__":
    main()
