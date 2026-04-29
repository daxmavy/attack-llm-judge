# Out-of-sample judge panel — specs and benchmark sources

Reference table for the 6 OOS judges used to score rewrites in the main results. Open-weight models are listed with their published benchmark scores; for the OpenRouter-served `qwen/qwen3.5-flash-02-23` and `xiaomi/mimo-v2-flash`, the open-weight base model's numbers are cited (per the model providers' confirmation that these endpoints serve those open weights).

## Mislabel correction

Earlier versions of `analysis/analysis.ipynb` and the heatmap CSV labeled `judge_mimo_v2_flash` as **"MiniMax M2"**. This is wrong — `xiaomi/mimo-v2-flash` is **Xiaomi's MiMo-V2-Flash** (309B-total/15B-active MoE, MIT-licensed), not MiniMax. MiniMax M2 (`minimax/minimax-m2.7`) is a different model. The label has been corrected.

Sources confirming the model identity:

- [Xiaomi: MiMo-V2-Flash – API Pricing & Providers (OpenRouter)](https://openrouter.ai/xiaomi/mimo-v2-flash)
- [XiaomiMiMo/MiMo-V2-Flash · Hugging Face](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)
- [MiMo-V2-Flash – Intelligence Analysis (Artificial Analysis)](https://artificialanalysis.ai/models/mimo-v2-flash)
- [OpenRouter on X – announcement of XiaomiMiMo team release](https://x.com/OpenRouterAI/status/2000956004675281094)

## Identity of `qwen/qwen3.5-flash-02-23`

Per Qwen's documentation, the OpenRouter endpoint `qwen/qwen3.5-flash-02-23` is the hosted version of the open-weight **`Qwen/Qwen3.5-35B-A3B`** (35B-total/3B-active MoE, Apache 2.0). The endpoint adds production features (1M-token context via YaRN, built-in tools) on top of the same base weights.

- [Qwen/Qwen3.5-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Qwen3.5-Flash-02-23 – API Pricing (OpenRouter)](https://openrouter.ai/qwen/qwen3.5-flash-02-23)
- ["Qwen3.5-Flash is the hosted version corresponding to Qwen3.5-35B-A3B" (Unsloth docs)](https://unsloth.ai/docs/models/qwen3.5)

---

## Panel specs

| Judge label (in main text)              | Slug                          | Model identifier                                       | Source       | Total params | Active params | Released         | License     |
|-----------------------------------------|-------------------------------|--------------------------------------------------------|--------------|--------------|---------------|------------------|-------------|
| Mistral-7B v0.3                         | `judge_mistral7b`             | `mistralai/Mistral-7B-Instruct-v0.3`                  | open-weight  | 7.2 B        | dense (n/a)   | 22 May 2024      | Apache 2.0  |
| Phi-3.5-mini                            | `judge_phi35mini`             | `microsoft/Phi-3.5-mini-instruct`                      | open-weight  | 3.8 B        | dense (n/a)   | 20 Aug 2024      | MIT         |
| Command-R 7B                            | `judge_cmdr7b`                | `CohereLabs/c4ai-command-r7b-12-2024`                  | open-weight  | 7 B          | dense (n/a)   | 13 Dec 2024      | CC-BY-NC 4.0 |
| Qwen3.5 Flash (= Qwen3.5-35B-A3B)       | `judge_qwen3_5_flash_02_23`   | OR: `qwen/qwen3.5-flash-02-23` (hosted Qwen3.5-35B-A3B) | API/open-weight | 35 B (MoE) | 3 B           | 25 Feb 2026      | Apache 2.0 (open weights) |
| Gemini 2.5 Flash-Lite                   | `judge_gemini_2_5_flash_lite` | OR: `google/gemini-2.5-flash-lite`                    | API (proprietary) | undisclosed | undisclosed   | 22 Jul 2025      | proprietary |
| Xiaomi MiMo-V2-Flash                    | `judge_mimo_v2_flash`         | OR: `xiaomi/mimo-v2-flash` (hosted MiMo-V2-Flash)     | API/open-weight | 309 B (MoE) | 15 B          | 16 Dec 2025      | MIT (open weights) |

### Sources for the spec rows

- **Mistral-7B v0.3** size, license, release: [mistralai/Mistral-7B-Instruct-v0.3 · Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- **Phi-3.5-mini** size, license, release: [microsoft/Phi-3.5-mini-instruct · Hugging Face](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- **Command-R 7B** size, license, release: [CohereLabs/c4ai-command-r7b-12-2024 · Hugging Face](https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024) and [Cohere blog: Introducing Command R7B](https://cohere.com/blog/command-r7b)
- **Qwen3.5-35B-A3B** size, license: [Qwen/Qwen3.5-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- **Gemini 2.5 Flash-Lite** release: [Google blog – Gemini 2.5 Flash-Lite generally available 22 Jul 2025](https://blog.google/products/gemini/gemini-2-5-flash-lite/) (n.b. proprietary; param count not disclosed)
- **MiMo-V2-Flash** size, license, release: [XiaomiMiMo/MiMo-V2-Flash · Hugging Face](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)

---

## Benchmark scores (consistently reported)

| Judge label                       | AAII (v4.0) | MMLU (5-shot) | MMLU-Pro (0-shot CoT) | benchmark sources |
|-----------------------------------|-------------|---------------|------------------------|-------------------|
| Mistral-7B v0.3                   | 7           | 60.1\* (Mistral 7B Instruct)   | 23.1                 | [AAII](https://artificialanalysis.ai/models/mistral-7b-instruct), [MMLU 60.1](https://mistral.ai/news/announcing-mistral-7b), [MMLU-Pro 23.1](https://llm-explorer.com/model/mistralai%2FMistral-7B-Instruct-v0.3,5UPzbP1x0T9qBVpCLTF61T) |
| Phi-3.5-mini                      | 10\*\*      | 69.0          | 47.4                   | AAII for Phi-3-mini family ([page](https://artificialanalysis.ai/models/phi-3-mini)), MMLU/MMLU-Pro from [HF model card](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) |
| Command-R 7B                      | not on AAII | not officially published (HF Open LLM Leaderboard v2 average normalized: 28.5) | not officially published | [HF model card](https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024); Cohere did not publish 5-shot MMLU or MMLU-Pro for this checkpoint |
| Qwen3.5 Flash (= Qwen3.5-35B-A3B) | **37**      | 93.3 (MMLU-Redux)\*\*\* | **85.3**             | [AAII for Qwen3.5-35B-A3B](https://artificialanalysis.ai/models/qwen3-5-35b-a3b), [MMLU-Redux + MMLU-Pro from HF model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |
| Gemini 2.5 Flash-Lite             | 13          | not in our search; vendor publishes mixed-eval composites only | not in our search    | [AAII (composite v4.0 used by analysis notebook)](https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index) |
| Xiaomi MiMo-V2-Flash              | **30**      | not officially reported | **84.9** (post-training) / 73.2 (base) | [AAII for MiMo-V2-Flash](https://artificialanalysis.ai/models/mimo-v2-flash), [MMLU-Pro from XiaomiMiMo HF model card](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash) |

### Footnotes

\* Mistral did not publish a v0.3-specific MMLU; the 60.1 figure is for the original Mistral 7B Instruct (the v0.x suffixes share the same base weights and similar instruction-tuning, so the score is treated as representative).

\*\* AAII does not have a dedicated page for Phi-3.5-mini; the score is taken from the Phi-3-mini family page and matches what the analysis notebook used as the OOS-only S8 panel reference. If you want a Phi-3.5-mini-specific number, it should be looked up directly on artificialanalysis.ai/comparisons.

\*\*\* Qwen reports MMLU-Redux (93.3) instead of vanilla MMLU; MMLU-Redux is a cleaned-up version of MMLU and the two are not directly comparable. Vanilla 5-shot MMLU is not in the official model card.

---

## Earlier-noted analysis-notebook AAII values

The `S8_AAII` dict in `analysis/analysis.ipynb` (cell starting line 8416) has:

| slug | analysis-notebook value | AAII source value | reconciliation |
|---|---|---|---|
| `judge_mistral7b` | 7 | 7 | ✓ |
| `judge_gemini_2_5_flash_lite` | 13 | 13 | ✓ |
| `judge_cmdr7b` | 25 | (no AAII page; was likely from a comparison view) | unverified |
| `judge_qwen3_5_flash_02_23` | 26 | 37 (open weight) / 26 (proprietary "Qwen3.5 Omni Flash" March 2026 release) | use **37** to match the open-weight Qwen3.5-35B-A3B per user instruction |
| `judge_mimo_v2_flash` | 36 | **30** | analysis notebook value is wrong; correct to 30 |

The display label `'judge_mimo_v2_flash': 'MiniMax M2'` has been corrected to `'Xiaomi MiMo-V2-Flash'`.

## Code changes

- `analysis/analysis.ipynb`:
  - Display label fix: `judge_mimo_v2_flash` → `'Xiaomi MiMo-V2-Flash'` (was `'MiniMax M2'`)
  - AAII fix: `judge_mimo_v2_flash` → `30` (was `36`)
  - AAII fix: `judge_qwen3_5_flash_02_23` → `37` (was `26`; matches the open-weight Qwen3.5-35B-A3B AAII)
  - Recommend regenerating `analysis/figures/pct_rewrite_lower_than_orig.csv` after the label fix lands.
