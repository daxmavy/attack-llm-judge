Mission for tonight (2026-04-19 → 2026-04-20):
- Finish informativeness training
- For both the 3 clarity RL models, and the 3 informativeness RL models, create rewrites for the evaluation dataset (so 6*714 generations). If any of the other attack methods don't have rewrites stored, do the same for them.
- Once you have done that, you will have rewrites across all attack methods for both clarity and informativeness (including the 6 RL models and the 6 BoNs across 3 folds * 2 criteria). This is the evaluation set of rewrites.
- Make sure that you have clarity and informativeness scores, from all three judges, for the evaluation set of rewrites. Additionally, make sure that the original human-written paragraphs (which provided the base for these rewrites) have scores (6 scores per paragraph)
- Apply the agreement_score regressor to estimate the agreement_score of the evaluation set of rewrites. Additionally, calculate the agreement_score of the original (human-written) evaluation_set paragraphs.
- Get the 3 judges' scores for both clarity and informativeness for all human-written paragraphs in the 40 propositions used for the evaluation set (note: including the paragraphs using in training dataset). 
- If you complete all of the above, here are some more tasks:
    - implement ICIR across all 3 folds and 2 criteria. Include its final rewrites in the evaluation set.
    - download 3 more models to use as out-of-sample judges on the evaluation set & original human set. Do it one at a time: download model, run on paragraphs, download next, etc.
    -  run one of the RL models for clarity for another 400 steps to see what happens. Try using G=8 for this experiment, and half the training data (I want to see if it's possible to fine-tine more aggressively and see what happens, considering that last time this did actually result in both better held-out performance and a larger hold-out performance gap).
    - If you really run out of tasks, do everything using a different base rewriter model. Choose a base model which is similarly-sized to the current one (so we don't run into memory issues), then run inference-only and RL methods across all three folds, for both clarity and informativeness, using the same training / evaluation split. I suspect that you will not have time to do this before I wake up, though - no problems if this is the case. 

Make sure that you don't run into a) network storage issues (I increased storage to 200 GB), b) stupid bugs (such as the default-logic which used the wrong qwen model), or c) VRAM limits. In particularly, make sure that a silly mistake doesn't cost hours of training - verify that what you're doing makes sense, and that you don't have, and don't introduce, bugs in code or logic which goes against the spirit of what I'm trying to achieve.

If you run into a bug which kills a run, don't just assume it's ok - if it would compromise the necessary requirements above, try to fix it before continuing.

Make sure you push models to HF, so you are able to safely delete them locally if necessary. Before you do this, check that you can safely restore them from HF.

---

# Follow-up additions (2026-04-20)

## Single-judge in-training ablation (task #39, currently running)
Run BoN, ICIR, and GRPO attacks with 1 in-training judge (qwen95b) instead of 2. Measure held-out generalisation gap on llama8b + gemma9b. Clarity only. Hypothesis: less diverse in-training signal → worse held-out transfer.

## 2-rewriter replication experiment (task #40, queued after #39)

**Goal.** Replicate the core 3-fold × 2-criteria experiment with two alternative rewriter base models so results are directly comparable to the existing Qwen2.5-1.5B-Instruct run.

**New rewriter bases:**
- `LiquidAI/LFM2.5-1.2B-Instruct`
- `google/gemma-3-1b-it`

Both are instruction-tuned and similarly-sized to `Qwen/Qwen2.5-1.5B-Instruct`.

**Pre-flight (MUST come first, before any attack-method execution).**
Before launching any long runs, run a compatibility smoke test for BOTH new rewriters:
1. Load the model in the current vLLM (0.18.1) — no version changes.
2. Generate one paragraph (short prompt).
3. Run one step of GRPOTrainer with a live judge (the same TRL + vLLM stack used by the existing experiment).
4. Produce a written report listing any issues encountered for each model. Do NOT attempt workarounds that change the library stack; if a model fails, flag it for user review. User is open to swapping in an alternative if needed.

**Execution order.**
Sequential per rewriter — LFM2.5 first (it's more likely to work; Gemma family has had vLLM friction historically), then Gemma-3. Within each rewriter: clarity 3-fold first, then informativeness 3-fold. This guarantees at least one complete comparable dataset early.

**Core attack methods only (no ablations).** naive, lit_informed_tight, rubric_aware, BoN K=16, ICIR iter=4, GRPO 400-step.

**Comparability requirements.**
- Same dataset (`controversial_40_3fold.json`: 1091 train / 714 eval, 40 propositions).
- Same mission judge panel (qwen95b / llama8b / gemma9b) in identical fold rotations.
- Same GRPO hyperparameters (lr=5e-6, alpha=100, penalty_shape=asymm_cubic, embed_sim β=200, etc.).
- Same chat-template handling — rely on each model's own `apply_chat_template`; use the existing `supports_system_role` fallback for models without a system role (Gemma-3 will likely need this).
- Tag rows in `attack_rewrites` by the `rewriter_model` field (existing column). Keep method names identical (`grpo_400step`, `bon_panel`, `icir`, etc.) so queries `GROUP BY method, fold, criterion, rewriter_model` produce the comparison tables directly.

**After each GRPO run completes.**
1. Use the freshly-trained model to generate rewrites on the eval-set and save them to `attack_rewrites` BEFORE any local cleanup — so we never have to re-download from HF just to produce eval rewrites.
2. Push the model to HF at `daxmavy/grpo-{rewriter-short-name}-fold{N}-{criterion}` (e.g. `daxmavy/grpo-lfm25-fold1-clarity`).
3. Verify HF round-trip (download + load + generate) before deleting anything locally.
4. Free local disk by deleting `final_*` dirs only after (1)+(2)+(3) are confirmed successful.

**Scoring + agreement_score.** Apply the full 3-mission-judge × 2-criteria scoring pass AND the DeBERTa agreement_score regressor to every new rewrite, identical to the Qwen2.5-1.5B setup. This is what makes the cross-rewriter comparison valid.

**Budget.** ~18 h GPU-hours per rewriter (≈15 h of GRPO training + ~3 h of attack generation + scoring). ~36 GPU-hours total for both rewriters. Spec clarity-first so even a partial completion has one full criterion worth of comparable data.
