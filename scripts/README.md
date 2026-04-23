# scripts/ — canonical source of truth for the mission pipeline

**Repo is canonical.** Every script lives here (or in `training/scripts/`).
The runtime directory `/workspace/grpo_run/` holds byte-exact copies of the
active subset; populate it with `bash scripts/sync_to_workspace.sh` after
each pull. `/home/max/` no longer contains pipeline scripts — everything
that used to live there moved here (2026-04-23 de-dup).

Live copies (kept in sync via `sync_to_workspace.sh`):

```
/workspace/grpo_run/
  run_pilot_len_pen.py          ← training/scripts/run_pilot_len_pen.py
  run_manifest.py               ← training/scripts/run_manifest.py
  length_penalty.py             ← training/scripts/length_penalty.py
  run_mission_attacks.py        ← scripts/run_mission_attacks.py
  run_icir.py                   ← scripts/run_icir.py
  stop_signal.py                ← scripts/stop_signal.py
  run_full_criterion_with_rewriter.sh
  run_grpo_3fold_with_rewriter.sh
```

Path bindings at the top of each script:

- `sys.path.insert(0, "/workspace/grpo_run")` — to import `run_pilot_len_pen.JudgeVLLM` and friends. Rebind to your scratch dir on a new host.
- `sys.path.insert(0, "/home/max/attack-llm-judge")` — to import `rewriters.rewrite_prompts`. Rebind to the repo path on a new host.

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

### Fidelity scoring (NLI and embedding)
- `apply_nli_scores.py` — bidirectional entailment scoring via `MoritzLaurer/ModernBERT-large-zeroshot-v2.0`. Populates `attack_nli_scores` with raw `nli_fwd`, `nli_bwd` probabilities. Idempotent; `--reset` to rescore.
- `e5_prefix_ablation.py` — one-off ablation of `"query: "` vs `"passage: "` e5-large-v2 prefix on attack-rewrite fidelity. Historical; results in EXPERIMENT_NOTES.md.

### Smoke tests / tooling
- `preflight_rewriter.py` — vLLM compatibility check: load → generate → single GRPO step with live judge. Produces `/home/max/attack-llm-judge/PREFLIGHT.md`.
- `build_controversial_subset.py` — regenerates the 40-prop stratified train/eval split. Only needed if you need a new subset.
- `init_db.py` — bootstrap a fresh `paragraphs.db` from the committed JSON + schema.
- `backup_db.py` — snapshot `paragraphs.db` to private HF dataset `daxmavy/attack-llm-judge-db` (SQLite online backup + gzipped SQL dump). Run whenever the DB has grown.
- `sync_to_workspace.sh` — copy canonical scripts into `/workspace/grpo_run/`.
- `stop_signal.py` — graceful-stop primitives. `touch <pilot_dir>/STOP` mid-run to save + exit after the next training step.

## Current reward modes

- `--embed-sim` — subtractive fidelity penalty: `reward = ej − length_pen − β·max(0, threshold − cos_sim_e5)`. Used for all overnight/Qwen2.5 + LFM2.5 + Gemma-3-1b runs.
- `--nli-fidelity` — additive bidirectional-entailment bonus: `reward = ej + 100·(P(rew→orig)+P(orig→rew))/2 − length_pen`. Used for the 2026-04-23 NLI retrain. Mutually exclusive with `--embed-sim`.

## Running order for a new rewriter × criterion

```
bash scripts/run_full_criterion_with_rewriter.sh <HF_ID> <short> clarity
bash scripts/run_full_criterion_with_rewriter.sh <HF_ID> <short> informativeness
# then:
python3 scripts/score_all_missing.py   # fill in all judge/criterion gaps
python3 scripts/apply_agreement_score.py
```
