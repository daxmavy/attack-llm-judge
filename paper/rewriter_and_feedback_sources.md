# Sources for rewriter + feedback-judge tables — Zotero import list

For each row in `tables/rewriter_models.tex` and `tables/feedback_models.tex`, the URLs below cover parameter count, release date, and AAII score. Paste these into Zotero's "Add Item by URL" dialog. The Hugging Face and Artificial Analysis pages parse cleanly; vendor blogs may need a manual title/date adjustment.

## Rewriter models (`tables/rewriter_models.tex`)

### Qwen2.5-1.5B-Instruct

- Params, release date: <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>
- AAII: not on Artificial Analysis (no dedicated page) — entry left blank.

### LFM2.5-1.2B-Instruct

- Params, release date: <https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct>
- AAII score (8): <https://artificialanalysis.ai/models/lfm2-5-1-2b>

### Gemma-3-1B-it

- Params, release date: <https://huggingface.co/google/gemma-3-1b-it>
- AAII score (6): <https://artificialanalysis.ai/models/gemma-3-1b>

### Qwen3-8B

- Params, release date: <https://huggingface.co/Qwen/Qwen3-8B>
- AAII score (11): <https://artificialanalysis.ai/models/qwen3-8b>

## Feedback judges (`tables/feedback_models.tex`)

### Qwen-3.5-9B (small panel)

- Params, release date: <https://huggingface.co/Qwen/Qwen3.5-9B>
- AAII score (32, reasoning-mode variant only): <https://artificialanalysis.ai/models/qwen3-5-9b>

### Llama-3.1-8B-Instruct (small panel)

- Params, release date: <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>
- AAII score (12): <https://artificialanalysis.ai/models/llama-3-1-instruct-8b>

### Gemma-2-9B-it (small panel)

- Params, release date: <https://huggingface.co/google/gemma-2-9b-it>
- AAII: not on Artificial Analysis (no dedicated page) — entry left blank.

### Mistral-Small 24B Instruct 2501 (medium panel, FP8-dynamic)

- Params, release date (base): <https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501>
- FP8-dynamic variant actually served: <https://huggingface.co/RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic>
- AAII score (13): <https://artificialanalysis.ai/models/mistral-small-3>

### Gemma-3-27B-it (medium panel, FP8-dynamic)

- Params, release date (base): <https://huggingface.co/google/gemma-3-27b-it>
- FP8-dynamic variant actually served: <https://huggingface.co/RedHatAI/gemma-3-27b-it-FP8-dynamic>
- AAII score (10): <https://artificialanalysis.ai/models/gemma-3-27b>

### Phi-4 (medium panel, FP8-dynamic)

- Params, release date (base): <https://huggingface.co/microsoft/phi-4>
- FP8-dynamic variant actually served: <https://huggingface.co/RedHatAI/phi-4-FP8-dynamic>
- AAII score (10): <https://artificialanalysis.ai/models/phi-4>

## AAII methodology (shared with the OOS-judge table)

- <https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index>

## Notes

- The slug `judge_qwen95b` in the codebase resolves to `Qwen/Qwen3.5-9B` per `grpo_run/run_pilot_len_pen.py`. Earlier appendix text referred to this model as "Qwen-2.5-9.5B"; that label is inconsistent with the registry and should be updated to "Qwen-3.5-9B" when the appendix is next revised.
- `Qwen2.5-1.5B-Instruct` and `Gemma-2-9B-it` do not have dedicated Artificial Analysis pages (verified 2026-04-29). The `---` entry follows the same convention used in `tables/oos_judge_panel.tex` for Command-R 7B.
- Quantisation provenance: the medium panel runs the RedHatAI FP8-dynamic variants per `config/models.py` on the `mission-2026-04-22` branch. The small panel runs in bf16 (default vLLM dtype), confirmed via the `JUDGE_REGISTRY` in `grpo_run/run_pilot_len_pen.py`.
