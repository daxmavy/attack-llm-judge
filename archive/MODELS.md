# Models in use

Status snapshot of the judge panel and rewriter base for this project. The
judge panel is split into **training-signal** judges (used to compute rewards
during GRPO/SFT/BoN) and **held-out** judges (never seen during training;
only used for transfer evaluation).

## Rewriter (policy being trained)

| Role | Model | Size | Notes |
|---|---|---|---|
| Rewriter base | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Instruct variant. Fits full fine-tuning in bf16 + AdamW fp32 on a single A100-80GB. |

## Current minimum GRPO run (2026-04-17)

Single fold, disjoint train (200) / eval (50) prompts drawn from the
top-decile writer paragraphs in `data/paragraphs.db`. Criterion: clarity.

### Training-signal judges (online, reward = mean of the two)

| Slug | Model | Size | Family | Gated |
|---|---|---|---|---|
| `judge_qwen7b` | `Qwen/Qwen2.5-7B-Instruct` | 7B | Alibaba / Qwen | no |
| `judge_llama8b` | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Meta | yes — accepted |

Both are in the plan's attack panel (`background_docs/methods_5_10_design.md`).
Chosen to hit cross-family agreement: one Qwen, one Llama. Scored via HF
transformers bf16 (SDPA attention); ensemble reward = `(score_qwen + score_llama) / 2`.

### Held-out judge (eval only — never in the reward loop)

| Slug | Model | Size | Family | Gated |
|---|---|---|---|---|
| `judge_gemma9b` | `google/gemma-2-9b-it` | 9B | Google / Gemma | yes — accepted |

Only loaded after training completes; used to score the 50 eval paragraphs
both pre- and post-training so we can compute the transfer gap
(`post - pre`) on an unseen evaluator.

### Gold panel (5, strong, disjoint from attack panel)

Serves as the "true quality" estimator. Scored via OpenRouter API (not
locally — frontier models don't fit available VRAM and aren't weight-
released anyway). Run on the stratified subsample of human-written
paragraphs (+ their rewrites) for cost reasons. Disjoint from the attack
and held-out panels (no Qwen-2.5-7B, no Llama-3.1-8B, no Gemma-2-9B
overlap).

| Slug | Model | OpenRouter | In / Out $/1M | Notes |
|---|---|---|---|---|
| `claude-haiku-4.5` | Anthropic Claude Haiku 4.5 | `anthropic/claude-haiku-4.5` | 0.80 / 4.00 | Anthropic family (downgraded from Sonnet 4.6 for \$100 budget). |
| `gemini-2.5-flash` | Google Gemini 2.5 Flash | `google/gemini-2.5-flash` | 0.15 / 0.60 | Google family (downgraded from Pro for budget). |
| `gpt-5-mini` | OpenAI GPT-5 mini | `openai/gpt-5-mini` | 0.25 / 2.00 | Reasoning model — judge client sets `reasoning.effort=low`, bumps `max_tokens=500`. |
| `deepseek-v3.2` | DeepSeek V3.2 | `deepseek/deepseek-v3.2` | 0.27 / 1.10 | Chinese-lab family. Replaces the deprecated `deepseek-chat` alias. |
| `llama-4-maverick` | Meta Llama-4-Maverick | `meta-llama/llama-4-maverick` | 0.15 / 0.60 | Meta frontier. Llama-3.1-405B (originally spec'd) is not hosted on OpenRouter; Llama-4-Maverick is the closest available. |

Configured in `eval_suite/panels.py::GOLD_PANEL`. Projected cost on the
`main_20pct` sample (903 writers + 10 rewriters × 903 ≈ 9,900 paragraphs)
is ~\$20 for clarity only.

**Caveat:** the gold panel is family-diverse but not reasoning-heavy
(only GPT-5-mini is a reasoning model; Opus 4.7 and DeepSeek-R1 would
have been stronger but cost more than the budget allows on the full
subsample). Documented in the write-up.

## Storage

All checkpoints cached at `/workspace/hf_cache/` on this pod. `/workspace`
is persistent (RunPod volume); `~/.cache` is ephemeral root fs — do not
rely on it for model weights.

