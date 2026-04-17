# Agent coordination

This file is written by in-repo Claude agents so we don't step on each other.
Keep entries short. Update the "Status" line when you start / finish. The most
recently updated section is the most authoritative.

Conventions:
- One section per agent, headed by a UTC timestamp (ISO 8601) + agent handle.
- Under each section, list: Goal, Touching (paths), Status, Notes.
- Do NOT modify another agent's section except to note a conflict.
- Before writing to a path, grep this file for the path — if another active
  agent claims it, coordinate or wait. Log conflicts under Notes.

---

## 2026-04-17 — agent:judge+rewriters (started 2026-04-17)

- **Goal:** Build LLM-as-a-judge (clarity + informativeness, Llama 3.3 70B + Gemini 2.0 Flash via OpenRouter), evaluate against paul_data human ratings, then implement and run the two judge-free rewriters from plan.md item (3).
- **Touching (write):**
  - `judge/` (new directory; rubrics.py, client.py, run_eval.py, results/)
  - `rewriters/` (new directory; rewrite.py, evaluate.py, results/)
  - `.env` (created; contains OpenRouter key)
  - `plan.md` ACTIVE AGENTS section only
- **Touching (read-only):** `paul_data/`, `background_docs/`, `agreement_model/`
- **Not touching:** `agreement_model/` (owned by other agent training DeBERTa)
- **Status:** IN PROGRESS — scaling up to a full evaluation suite.
- **Notes:** OpenRouter spend through v2_tight ~$0.30. Rewriter is Qwen 2.5 72B (not in judge pool, not in paul_data original-author pool). See `plan.md` ACTIVE AGENTS entry for numbers.

---

## 2026-04-17 — agent:eval-suite (same operator, second phase)

- **Goal:** Systematic evaluation suite covering clarity (parameterised so informativeness can be added). 10 rewriter methods × ~4503 writer paragraphs. Metrics: (1) 5 small "attack" LLM judges, (2) 5 strong "gold" LLM judges, (3) word-count change, (4) learned agreement-score, (5) learned clarity regressor, (6) AI-detector probability; plus (a) embedding similarity and (c) hallucinated-specifics (approved additions).
- **Touching (write):**
  - `data/` (new) — SQLite paragraphs.db + methods.yaml + parquet view
  - `eval_suite/` (new) — schema, ingest, judge runners, metric modules, detector, regressors
  - `rewriters/methods_5_10/` or extend rewrite_prompts.py with 6 more methods
  - `criterion_model/` (new) — clarity regressor (DeBERTa-v3-base)
  - `plan.md` ACTIVE AGENTS section only
- **Touching (read-only):** `paul_data/`, `background_docs/`, `agreement_model/runs/main/final/` (regressor weights; needed for metric 4)
- **Not touching:** `agreement_model/train.py` and runs (owned by other agent)
- **Running sub-agents:**
  - research: best open-source AI-text detector (2025-26) — expects Binoculars + install plan
  - design: rewriting methods 5-10 (ICIR, BoN, injection, rules-explicit, rubric-aware, scaffolded)
- **Cost flag:** user set $100 budget for gold eval (option ii: Sonnet 4.6 / Gemini 2.5 Pro / GPT-5-mini / DeepSeek-V3 / Llama-3.1-405B on full corpus). At current OpenRouter pricing this panel on ~45k paragraphs is likely $200-300. Will project exact cost after smoke test and flag to user before launching the full run.
- **Status:** scaffolding DB + ingest while sub-agents run.


---
