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
- **Status:** in progress — judge smoke test passed; about to run eval sample.
- **Notes:** Using the OpenRouter budget sparingly (~$0.30 cap for this run). Judge rubric + client are in place; sampling 300 docs stratified by paragraph_type × model_name.

---
