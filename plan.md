AIM: build and evaluate different methods for rewriting paragraphs (containing political opinions) to achieve better results under LLM-as-a-judge.

There are multiple coding agents working in parallel in this environment; each agent is responsible for a specific task.

Evaluate each method based on two criteria:
1) How effective is it if we don't know the exact prompt or model used for the LLM-as-a-judge?
2) How much semantic drift does it introduce? (operationalised as 'level of agreement with the proposition')

Parts of project:
1) Create judge: Find a realistic LLM-as-a-judge rubric and set up LLM-as-a-judge (clarity, informativeness to proposition)
2) Agreement model: Estimate agreement_score for new documents robustly:
    - Can it estimate agreement_score accurately when used on out-of-sample models?
        - train against human-only and check performance on AI-rewritten
        - train against human-only and only 1 AI model, check performance on the other two

3) Judge-free rewriting: Build and evaluate judge-free rewriting methods: (do with highly capable but not-too-expensive model)
    - Ask LLM to rewrite the text (no further prompting)
    - Ask LLM to rewrite the text (prompt draws from existing findings in literature on improving judgement results)
    - 

Implementation details:
- Need to figure out best way to use GPUs - might need to move this onto a remote server that I rent
- All generations should attempt to control the word count so that it is approximately the same as the original paragraph to be re-written.
- .env file has OpenRouter key
- To test generalisability, start small: 2 LLM judges, and 2 prompts (one judging for 'clarity' and one judging for 'informativeness'), for 4 combinations total.
- LLM judge should return a score between 0 (bad) and 100 (good) for quality judgements.
- LLM judges: Llama 3.3 70B (~$0.10/$0.32 per 1M in/out) and Gemini 2.0 Flash (~$0.10/$0.40 per 1M in/out), both via OpenRouter. Held-out OOD judges for transfer evaluation: GPT-4o-mini and Claude Haiku 3.5.
- Base model for RL/SFT training (single A100 40GB): Qwen 2.5 1.5B — full fine-tuning fits without LoRA, fast GRPO loops, strong reasoning baseline at this scale. Similar-size comparison models: Llama 3.2 1B, Gemma 2 2B.

Known problems:
- Is it possible to estimate the deviation from human judgement accurately? We could learn a model to estimate `criterion` of paragraphs using paul_data - but problem is, very high-quality generations might be created, whose true quality is above that observed in the distribution used to train the quality-estimating model. This suggests that we shouldn't try to estimate the deviation between LLM judge estimates and the 'true' estimate; am leaving this out of scope for the moment.

*ACTIVE AGENTS* (write down when you started, what your goal is, and what you're working on):
- 2026-04-17 (agent: agreement-model, status: IN PROGRESS training):
    - Goal: plan item (2) — train agreement-score regressor on paul_data/prepared/documents.csv (proposition + paragraph -> agreement_score, 0-1 using aggregated rater stance, not writer self-report).
    - Files owned: `agreement_model/` (train.py, runs/, model checkpoints). Please don't modify.
    - External side effects: GPU (A100, ~15 GB used during training). Will release when run finishes.
    - Main run (runs/main): DeBERTa-v3-base fp32, 4 epochs, group-by-proposition 80/10/10. Test MAE=0.080, Pearson=0.937 (overall). Saved weights at runs/main/final/.
    - Hold-out sweep done (13 configs, 3 epochs each). See `agreement_model/runs/sweep/SUMMARY.md`. Headlines for plan item (2):
        - Human-only training already transfers to AI paragraphs (MAE 0.089, Pearson 0.94); adding any single AI source closes most of the remaining gap.
        - Leave-one-AI-out: held-out AI model scores at MAE 0.07-0.085, Pearson 0.94-0.96 — essentially no distribution-shift cost.
        - Hardest subgroup is human writers, not AI (reverse of what we might have expected). Single-AI-only training is the one failure mode (Pearson drops to 0.68-0.82).
    - Status: IDLE, GPU released. No further runs queued.
- 2026-04-17 (agent: judge+rewriters, status: IN PROGRESS):
    - Goal: plan items (1) LLM-as-a-judge and (3) judge-free rewriters.
    - Judge built in `judge/` — clarity + informativeness rubrics with 0-100 anchored scoring and JSON chain-of-thought output; OpenRouter client for Llama 3.3 70B + Gemini 2.0 Flash (the two in-scope judges; GPT-4o-mini and Claude Haiku 3.5 are held out).
    - Baseline eval on 300 stratified docs (tag `baseline`, `judge/results/`): parse OK 99.9%, spend ~$0.11, overall Pearson vs human means: Llama 0.49/0.61 (clarity/inf), Gemini 0.55/0.60; judge-judge Pearson 0.90/0.77. Weak cell: clarity on writer paragraphs (both judges under-rate human-written paragraphs).
    - Files owned: `judge/`, `rewriters/` (new), `COORDINATION.md`, `.env`, `.gitignore`.
    - Next: implement + run the two judge-free rewriters (naive, literature-informed) and re-score with the judges.
    - UPDATE 2026-04-17 (rewriters done): `rewriters/` with Qwen 2.5 72B as rewriter (not a judge, not a paul_data original author — avoids self-preference both ways). Two prompts: `naive` (rewrite with stance + word-count constraint) and `lit_informed` (GEO/Prometheus/G-Eval rewrite heuristics: specific facts, explicit reasoning chain, authoritative register, main-claim-first structure; same constraints). Ran on 100 stratified baseline docs × 2 methods × 2 judges × 2 criteria = 800 judge calls. Rewrite success 100%, judge parse 100%, word ratio 0.87-0.90 vs originals, spend $0.09 (rewrites+judging). Key results (mean delta vs original):
        - naive: clarity +5.3 (Gemini) / +6.3 (Llama); informativeness +2.9 / +1.7.
        - lit_informed: clarity +8.0 / +6.6; informativeness +6.5 / +6.9.
        - lit_informed beats naive on informativeness by ~+4-5 and on clarity by ~+2 (Gemini) / 0 (Llama).
        - Gains are dominated by writer-paragraphs (+18-25 clarity, +7-14 informativeness); AI-authored originals already score high and improve only marginally.
        - Suggests the two prompts are similarly good at clarity but lit_informed is noticeably better at informativeness — consistent with literature on what judges reward.
        - Outputs: `rewriters/results/rewrites_v1.csv`, `rewrite_judge_scores_v1.csv`, `combined_v1.csv`, `summary_v1.csv/json`.
        - Total spend so far (judge+rewriters): ~$0.20 of $30 budget.
    - UPDATE 2026-04-17 (length-controlled variants): Word-count fidelity of v1 was weak — naive hit 56% within ±10% (mean pct_err 11%); lit_informed only 28% within ±10% (mean pct_err 15.5%), with 25% of outputs <0.80× original. Added two length-controlled variants that keep the originals' prompts intact (`naive_tight`, `lit_informed_tight` in `rewriters/rewrite_prompts.py`) plus a one-shot length retry in `run_rewrite.py`: if the first draft is outside ±10%, re-prompt once with the actual count and the required range. Tight prompts use explicit min/max, explicit "paragraph is X words" anchor, and forbid headings/bullets. Re-ran on the same 100-doc sample (tag `v2_tight`). New fidelity (`rewriters/results/wordcount_fidelity.csv`):
        - naive → naive_tight: pct_err 11.0% → 5.2%; within ±10% 56% → 90%; within ±20% 86% → 100%; under 0.80× 14% → 0%.
        - lit_informed → lit_informed_tight: pct_err 15.5% → 7.0%; within ±10% 28% → 76%; within ±20% 74% → 99%; under 0.80× 25% → 1%.
        - Judge deltas held up or improved slightly: lit_informed_tight vs lit_informed Llama informativeness delta +9.7 vs +6.9, clarity about the same; same on Gemini. Length control did not cost quality.
        - Old v1 methods kept in place. Total rewriter+judge spend incl. v2: ~$0.30.
    - UPDATE 2026-04-17 (second phase — evaluation suite): scope expanded per operator. Now building a systematic eval suite (`eval_suite/` + `data/`):
        - Criterion: CLARITY only for now, but all code parameterised so informativeness can be added by adding a row to the criteria config.
        - Attack panel (5, family-diverse, small): meta-llama/llama-3.1-8b-instruct, qwen/qwen-2.5-7b-instruct, mistralai/mistral-7b-instruct, google/gemma-2-9b-it, microsoft/phi-3.5-mini-instruct. Leave-one-out generalisation across the 5.
        - Gold panel (5, disjoint, strong): anthropic/claude-sonnet-4.6, google/gemini-2.5-pro, openai/gpt-5-mini, deepseek/deepseek-v3, meta-llama/llama-3.1-405b-instruct. Option (ii) per operator.
        - Rewriter methods: 10 total. The 4 we already have (naive, lit_informed, naive_tight, lit_informed_tight) plus 6 new ones designed by a sub-agent (ICIR, BoN-ensemble, injection, rules-explicit, rubric-aware, structure-scaffolded). All inference-only; training-based methods out of scope for now.
        - Storage: SQLite at `data/paragraphs.db`, tables: `paragraphs` (originals + rewrites, document_id + base_document_id + method info + stance/type), `methods` (rewriter config), `evaluations` (long format: paragraph_id × metric × judge/model × value), `criteria` (clarity now, informativeness later). Plus a parquet denormalised view for analysis.
        - Extra metrics beyond user's 6: embedding cosine similarity to base original (e5-large-v2), hallucinated-specifics rate (regex + spaCy NER, new entities/numbers in rewrite but not in original).
        - Top-10% flag: WILL COMPUTE for writer paragraphs as "top-10% by `mean_human_clarity` within each (proposition × agreement-score quintile) cell." If operator wanted a different ranking metric, update this entry; operator flagged ambiguity in last message but moved on.
        - Two sub-agents in progress: (a) AI-detector research → Binoculars install plan; (b) methods 5-10 design. Will integrate when they return.
        - Gold-panel cost at full corpus (10 methods × 4503 writer paragraphs + 4503 originals + ~5500 other-type-originals ≈ 55k paragraphs × 5 gold × 1 criterion ≈ 275k gold calls) is projected to exceed the $100 limit from earlier; will compute exact number after smoke test and flag to operator before launching the full run.

    - CURRENT SPRINT (2026-04-17, still running):

        **What's already done (this sprint):**
        - DB schema `eval_suite/schema.py` with tables: `paragraphs`, `methods`, `evaluations` (long format), `criteria`. SQLite + WAL at `data/paragraphs.db`. Schema is append-only and criterion-agnostic.
        - Ingest `eval_suite/ingest.py` — populated DB with 10,008 originals (4,503 writer + 4,503 model + 1,002 edited). Computed top-decile flag for each writer paragraph as: within (proposition × agreement-score quintile), top 10% by `mean_human_clarity`. Per-quintile counts: 917/898/889/901/898, top-deciles 102/100/100/100/100.
        - Judge panels `eval_suite/panels.py`: 5 attack + 5 gold configs with OpenRouter pricing for cost estimation.
        - Judge runner `eval_suite/judge_runner.py`: parameterised by criterion + panel, idempotent via PRIMARY KEY on (paragraph_id, metric, criterion, source), writes straight to DB.
        - Research outputs saved: `background_docs/methods_5_10_design.md` (6 new methods with exact prompts + algorithms + expected transfer ranking), `background_docs/ai_detector_plan.md` (Binoculars install + caveat + Fast-DetectGPT backup).

        **Next steps, in order:**
        1. Metric modules (`eval_suite/metrics/`): word_count, embed_sim (e5-large-v2), hallucinated_specifics (regex + spaCy NER), agreement-model inference wrapper, clarity-regressor inference wrapper, Binoculars wrapper. Each metric exposes `.score(paragraph_ids) -> dict` AND `.benchmark()` that runs on paul_data's labeled human-vs-AI split (per operator's latest request).
        2. Implement 6 new rewriter methods (`rewriters/rewrite_prompts.py` extend + `rewriters/pipeline.py`, `panel_scorer.py`, `iterative.py`, `parsing.py` helpers). All 10 methods registered in `methods` DB table.
        3. Train a clarity regressor on `mean_human_clarity` (DeBERTa-v3-base, group-by-proposition split; same pattern as the existing agreement_model).
        4. Integrate Binoculars (runs on A100 when free; co-tenant with agreement_model weights loaded in memory).
        5. End-to-end smoke test on 50 paragraphs × 10 methods. Populate DB. Compute each metric's paul_data benchmark.
        6. Project full-corpus cost (especially gold panel). **Flag to operator before launching the expensive gold-panel eval.** Likely ~$250–300 for gold at current spec; operator's earlier stated budget was $100. Will propose: downgrade gold to cheaper strong models (e.g. swap Sonnet 4.6 → Claude Haiku 4.5, Gemini 2.5 Pro → Flash), OR subsample to a fixed N per method.

        **Files this sprint will own:** `eval_suite/*`, `data/paragraphs.db`, `criterion_model/*`, `background_docs/methods_5_10_design.md`, `background_docs/ai_detector_plan.md`, extend `rewriters/*`. Will NOT touch `agreement_model/*`.

    - PROGRESS UPDATE 2026-04-17 late-afternoon:
        - DB schema + ingest: ✅ done. 10,008 originals, top-decile flags ~100/quintile.
        - 10 rewriter methods: ✅ implemented (naive, lit_informed, naive_tight, lit_informed_tight, injection_leadin, rules_explicit, rubric_aware, scaffolded_cot_distill, icir_single, bon_panel).
        - Clarity regressor: ✅ trained. Test Pearson 0.758 / MAE 0.072 (writer cell 0.68, model cell 0.74). Weights at criterion_model/clarity/final/ (704MB, gitignored).
        - Judge panels calibrated to live OpenRouter slugs:
          - Attack: llama-3.1-8b, qwen-2.5-7b, ministral-8b-2512, gemma-3-4b-it, grok-3-mini
          - Gold (operator downgraded sonnet→haiku + pro→flash to fit $100): claude-haiku-4.5, gemini-2.5-flash, gpt-5-mini, deepseek-v3.2, llama-4-maverick
          - Reasoning-model handling (reasoning.effort=low) added to judge client for Gemini 2.5 Pro / GPT-5-mini.
        - Stratified sample `main_20pct`: ✅ 903 writer paragraphs across 10 strata (5 quintiles × 2 top-decile classes).
        - Main run in progress: rewrites + attack panel + gold panel + non-LLM metrics on 903 writers + 9030 rewrites = 9933 paragraphs. `naive` complete (903 in 693s, $0.08). 9 methods to go. Estimated total wall ~2–3 hrs, spend ~$40.
        - AI detector: switched from Binoculars (28GB Falcon-7B pair didn't fit 20GB disk) to Fast-DetectGPT-lite with GPT2-XL (single model, analytic form). Vendored Binoculars into `eval_suite/binoculars/` in case disk grows later. Currently running detector on all 10k originals for calibrator + benchmark.
        - AutoCLI `gh` installed at `~/.local/bin/gh`; operator asked to run `gh auth login` to unblock GitHub push.
        - Next after main_run finishes: (a) detector rescoring over rewrites, (b) regressor inference on full sample scope, (c) benchmark run producing the paul_data benchmark JSON.