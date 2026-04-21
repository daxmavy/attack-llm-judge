# scripts/ — canonical copies of the mission pipeline

These are the source-of-truth copies of every script used in the current
attack experiment. The live working copies sit at:

- `/workspace/grpo_run/*.sh` and `/workspace/grpo_run/*.py` (the driver/runner scripts)
- `/workspace/grpo_run/run_pilot_len_pen.py` (mirrors `training/scripts/run_pilot_len_pen.py`)
- `/home/max/*.py` (the post-hoc DB scripts: scoring, agreement, backfill, preflight)

When you clone this repo onto another machine, copy these scripts into your
working directory (preferably `/workspace/grpo_run/` or equivalent scratch dir)
before running. They assume:

- `sys.path.insert(0, "/workspace/grpo_run")` to import `run_pilot_len_pen` → rebind this to wherever you put the training script on the new host.
- `sys.path.insert(0, "/home/max/attack-llm-judge")` to import `rewriters.rewrite_prompts` → rebind to the repo path on the new host.

## Script index

### Attack generation
- `run_mission_attacks.py` — CLI with subcommands `feedback_free`, `bon_generate`, `bon_score FOLD`. Used by the full-pipeline shell script. Accepts `--rewriter` to swap base model, `--criterion clarity|informativeness`.
- `run_icir.py` — ICIR (4-iteration iterative rewrite with judge feedback). Accepts `--folds`, `--criteria`, `--rewriter`, `--in-panel-override`, `--method-tag`.

### GRPO orchestration
- `run_3fold_grpo.sh` — the original overnight driver: 3 folds × single criterion using the hardcoded base rewriter. Superseded by `run_grpo_3fold_with_rewriter.sh` for parameterised runs.
- `run_grpo_3fold_with_rewriter.sh <HF_ID> <short> <criterion>` — newer 3-fold GRPO driver that accepts rewriter base and short name. Pushes each model to HF and calls `backfill_grpo_rewriter.py` to ingest post-eval rewrites + held-out scores into the DB.
- `run_full_criterion_with_rewriter.sh <HF_ID> <short> <criterion>` — chains feedback-free → bon_generate → 3×bon_score → 3×icir (fresh subprocess per fold to avoid vLLM memory-leak OOM) → 3-fold GRPO.

### DB ingestion / scoring
- `backfill_grpo_rewrites.py` — legacy backfill for the original clarity GRPO runs (hardcoded fold 1 posthoc scoring logic).
- `backfill_grpo_rewriter.py` — general-purpose: `--short SHORT --fold N --criterion CRIT --rewriter HF_ID --held-out JUDGE_SLUG`. Idempotent.
- `score_all_missing.py` — mission-panel scoring: scans `attack_rewrites`, finds missing (judge × criterion) cells, loads one vLLM judge at a time, rubric-swaps across criteria in-place. `--methods`, `--judges`, `--criteria`, `--include-candidates`.
- `apply_agreement_score.py` — DeBERTa-v3-base regressor scoring for stance agreement. Idempotent.

### Smoke tests / tooling
- `preflight_rewriter.py` — vLLM compatibility check: load → generate → single GRPO step with live judge. Produces `/home/max/attack-llm-judge/PREFLIGHT.md`.
- `build_controversial_subset.py` — regenerates the 40-prop stratified train/eval split. Only needed if you need a new subset.

## Running order for a new rewriter × criterion

```
bash scripts/run_full_criterion_with_rewriter.sh <HF_ID> <short> clarity
bash scripts/run_full_criterion_with_rewriter.sh <HF_ID> <short> informativeness
# then:
python3 scripts/score_all_missing.py   # fill in all judge/criterion gaps
python3 scripts/apply_agreement_score.py
```
