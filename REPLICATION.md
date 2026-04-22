# Replication guide — 3-fold × 2-criteria attack experiment

This document tells another agent how to replicate the experiment. The target setup is **2× A100 80GB** — GPU 0 serves the 2 in-panel judges via vLLM, GPU 1 runs the rewriter (vLLM rollouts + QLoRA GRPOTrainer). Single-GPU co-location is a known-bad configuration (see §9); don't launch a full 400-step run there.

## 0. Executive summary

- **Goal**: test how well different rewriter-side "attack" methods can lift LLM-judge scores on short argumentative paragraphs, for two judging criteria (clarity, informativeness), with held-out-judge transfer evaluated across 3 cross-validation folds.
- **Attack methods under test** (6 total): `naive`, `lit_informed_tight`, `rubric_aware`, `bon_panel` (BoN K=16), `icir` (4-iteration iterative refinement), `grpo_400step` (GRPO RL for **400 steps** — locked, CLAUDE.md GRPO step-budget rule).
- **Judge panel**: 3 vLLM judges. Identities and fold rotation live in **`config/models.py`** (`JUDGE_REGISTRY` and `FOLDS`). Rotated per fold: 2 are in-panel training judges, 1 is held-out evaluator. Do not re-declare judge IDs anywhere else — see §7 and EXPERIMENT_NOTES.md §2026-04-22 refactor.
- **Rewriter base**: also lives in `config/models.py` (`REWRITER`). Historical rewriters previously tested (for reference): `Qwen/Qwen2.5-1.5B-Instruct`, `LiquidAI/LFM2.5-1.2B-Instruct`, `google/gemma-3-1b-it`, `Qwen/Qwen3-14B`. The pipeline is agnostic to the rewriter choice so long as it works in vLLM and fits on one A100 under QLoRA.
- **Dataset**: 40 controversial political/policy propositions, stratified 1091 train / 714 eval. Same split across all folds. Stratification is 3×3 on (human-labelled clarity tertile × human-stance-agreement tertile) to equalise difficulty across folds.

---

## 1. Fold rotation

All 3 mission judges are used in every fold, but the role rotates. The authoritative mapping lives in `config/models.py::FOLDS` — the table below is illustrative shape only (slugs depend on whatever `JUDGE_REGISTRY` currently holds):

| fold | in-panel training judges (reward signal) | held-out eval judge |
|------|------------------------------------------|---------------------|
| 1    | `judgeA` + `judgeB`                       | `judgeC`            |
| 2    | `judgeA` + `judgeC`                       | `judgeB`            |
| 3    | `judgeB` + `judgeC`                       | `judgeA`            |

The **reward signal at training time is the mean of the two in-panel judges**, minus a length penalty, minus a fidelity (embedding-similarity) penalty. At eval time the held-out judge scores the 714 post-rewrites to measure out-of-distribution transfer.

---

## 2. Attack methods (what gets stored)

Per rewriter × criterion × (where applicable) fold, the following rewrite tables are produced:

| method | criterion-aware prompt? | fold-specific? | n rewrites |
|---|---|---|---|
| `naive` | no (same text reused across criteria) | no | 714 |
| `lit_informed_tight` | yes | no | 714 per criterion |
| `rubric_aware` | yes (full rubric verbatim in prompt) | no | 714 per criterion |
| `bon_candidate` | yes | no | 11,424 per criterion (K=16 × 714) |
| `bon_panel` | yes | yes | 714 per fold per criterion (argmax of K=16 under that fold's 2 in-panel judges) |
| `icir` | yes | yes | 714 per fold per criterion (final of 4 iterations — iter 0 = lit_informed_tight seed, iters 1-3 = refinement with judge reasoning feedback) |
| `grpo_400step` | no (one RL run per fold × criterion) | yes | 714 per fold per criterion (model-generated rewrites on eval set after 400 training steps) |

All methods run on the **same 714-paragraph eval set**. Only GRPO uses the 1091-paragraph training set.

The rewriter-side prompt templates are in `rewriters/rewrite_prompts.py`. The GRPO rewriter prompt is hard-coded in `training/scripts/run_pilot_len_pen.py` at `make_rewrite_prompt()` (note: it says "clearer and easier to read" even in informativeness runs — the criterion swap happens at the reward-side judge rubric, not in the rewriter instruction. Keep this quirk unless you want to invalidate comparability).

---

## 3. Reward function (GRPO)

The `reward_fn` in `training/scripts/run_pilot_len_pen.py` does the following for each completion `r` generated in a rollout:

```
reward = ensemble_judge_score(r) - length_penalty(r) - fidelity_penalty(r)
```

Where:

- **`ensemble_judge_score(r)`** = `mean(judge1.score(prop, r), judge2.score(prop, r))`. Each judge is a `JudgeVLLM` instance wrapping vLLM.`LLM.generate` with the rubric-specific prompt template and `SamplingParams(temperature=0.0, max_tokens=180)`. Parses `{"reasoning": ..., "score": int}` from the model output.
- **`length_penalty(r)`** = `asymm_cubic` shape with `alpha=100`, `tol=0.1` of target word count, `over_tol=0.15` (asymmetric — overshooting by 15%+ costs cubically). See `training/length_penalty.py`.
- **`fidelity_penalty(r)`** = `embed_beta=200 × max(0, embed_threshold - cos_sim(embed(orig), embed(r)))`. `embed_beta=200`, `embed_threshold=0.85`. Embedder is `intfloat/e5-large-v2`.

**Rubric**: the full text is in `training/scripts/run_pilot_len_pen.py` as `CLARITY_RUBRIC` and `INFORMATIVENESS_RUBRIC`. 0-100 integer scores with 5 anchor bands.

**Judge prompt quirk**: Qwen3 family defaults to thinking mode, which burns the 180-token output budget before the JSON score appears. The JudgeVLLM class passes `enable_thinking=False` to `apply_chat_template` when the model id contains "qwen3". Preserve this fix.

---

## 4. GRPO hyperparameters (current)

From `run_3fold_grpo.sh`:

```
--max-steps 400
--alpha 100
--lr 5e-6
--beta 0.01 (KL coefficient)
--temperature 1.0
--num-generations 4  (G)
--per_device_train_batch_size = 2 * G = 8
--scale-rewards group
--loss-type dapo
--penalty-shape asymm_cubic --penalty-gamma 1000 --penalty-over-tol 0.15
--embed-sim --embed-beta 200 --embed-threshold 0.85
--criterion {clarity|informativeness}
--train-judges {fold's 2 in-panel judge slugs}
--heldout-judge {fold's held-out slug}
--dataset-json /workspace/grpo_run/controversial_40_3fold.json
--base-model {HF id}  (optional; falls back to config.models.REWRITER)
```

Defaults to `save_strategy="no"` (periodic step-50 saves caused silent failures when disk quota was tight).

TRL version: 1.2.0. vLLM version used for judges + rewriter rollouts: **0.18.1** — note TRL claims "supported vLLM 0.11.0–0.18.0" so you're one minor version out of range; it still works but expect deprecation warnings.

---

## 5. Dataset

Shipped in this repo at `data/controversial_40_3fold.json` (1.9 MB, pinned in git — the build script in `scripts/build_controversial_subset.py` is for regeneration only and defaults to a 25-prop variant; do **not** regenerate, use the committed JSON).

Pipeline scripts still load it from `/workspace/grpo_run/controversial_40_3fold.json` (hardcoded path). On the replication host, copy it over:
```
cp <repo>/data/controversial_40_3fold.json /workspace/grpo_run/controversial_40_3fold.json
```

Contents:

```
{
  "n_propositions": 40,
  "n_train": 1091,
  "n_eval": 714,
  "seed": 17,
  "stratification": "...",
  "propositions": [40 proposition texts],
  "rows": [1805 rows, each with `document_id`, `proposition`, `text`, `split`='train'|'eval'],
}
```

Rows are contiguous: `rows[:1091]` are train, `rows[1091:]` are eval. Use this JSON as the single source of truth — do NOT re-sample. The 714 eval paragraphs are the same across every run.

The underlying documents live in `data/paragraphs.db`, table `paragraphs`, keyed by `document_id`. The 40 propositions are a curated top-controversial subset of the `paul_data` corpus.

---

## 6. Data schema (SQLite at `data/paragraphs.db`)

- **`paragraphs`**: master table. Human-written originals with `document_id`, `proposition`, `text`, `word_count`, etc. Treat as read-only.
- **`attack_rewrites`**: every generated rewrite gets a row. Columns: `rewrite_id` (PK, stable), `source_doc_id`, `method`, `fold` (nullable), `criterion`, `config_json`, `rewriter_model`, `judge_panel_json`, `text`, `word_count`, `run_metadata_json`.
- **`attack_judge_scores`**: (`rewrite_id`, `judge_slug`, `criterion`) PK. `score` float 0-100. Written by the post-hoc scoring pass.
- **`attack_agreement_scores`**: `rewrite_id` → `score` float 0-1 from DeBERTa-v3-base regressor trained to predict `agreement_score` from (proposition, paragraph).

**Comparability across rewriters**: queries `GROUP BY method, fold, criterion, rewriter_model`. The `rewriter_model` column is the HF id string and is set correctly by the parameterised scripts.

---

## 7. Pipeline scripts (run in this order per rewriter × criterion)

**Before launching anything**: populate `config/models.py` with `REWRITER`, `JUDGE_REGISTRY`, `FOLDS` and confirm `python3 -c "from config.models import require_config; require_config()"` exits 0. Every pipeline script below calls `require_config()` at entry and will abort with a clear error if placeholders remain. Do NOT hardcode model ids or fold rotation inside any script or shell wrapper — the 2026-04-22 refactor centralises this so one edit propagates to every downstream tool (postmortem: EXPERIMENT_NOTES.md §2026-04-22 refactor).

All under `/workspace/grpo_run/`:

1. **`run_mission_attacks.py feedback_free --criterion CRIT --rewriter HF_ID`** — generates naive, lit_informed_tight, rubric_aware on eval set. ~3-5 min on a 1-2B model. Skips naive for informativeness if clarity-naive rewrites already exist (same text).
2. **`run_mission_attacks.py bon_generate --criterion CRIT --k 16 --rewriter HF_ID`** — generates 11,424 candidates (714 × K=16). Temperature=1.0 for diversity. ~2-5 min.
3. **`run_mission_attacks.py bon_score FOLD --criterion CRIT --rewriter HF_ID`** (run for FOLD in 1, 2, 3) — scores all 11,424 candidates with that fold's 2 in-panel judges, picks argmax per source doc. Saves 714 panel winners as `method='bon_panel'`, fold=FOLD. ~30-45 min per fold (qwen95b is the bottleneck at ~25-30 min).
4. **`run_icir.py --folds 1 2 3 --criteria CRIT --rewriter HF_ID`** — 4-iteration ICIR, uses lit_informed_tight template as seed and `ICIR_ITER_TEMPLATE` with feedback. Uses the fold's 2 in-panel judges for iterative scoring. *Known quirk*: run ONE fold per subprocess — vLLM engines don't fully release memory between `del j; torch.cuda.empty_cache()` calls on the same process, causing OOM on the second fold. Wrapping bash loop spawns fresh processes per fold.
5. **`run_pilot_len_pen.py --base-model HF_ID ...`** per fold × criterion — GRPO 400-step training with in-panel judges for reward, held-out judge for post-train eval. Saves final model to `/workspace/grpo_run/final_...`.
6. **HF push + DB backfill**: `run_grpo_3fold_with_rewriter.sh <HF_ID> <short> <criterion>` chains all 3 GRPO folds + `api.upload_folder` to `daxmavy/grpo-{short}-fold{N}-{criterion}` + calls `/home/max/backfill_grpo_rewriter.py` to ingest `eval_summary.json` post-rewrites + held-out scores into `attack_rewrites` / `attack_judge_scores`.
7. **Full mission-panel scoring**: `/home/max/score_all_missing.py` — scans `attack_rewrites`, finds cells missing (judge × criterion) scores, scores with vLLM judge models. Idempotent. Excludes `bon_candidate` by default (only panel winners need evaluation). For all 3 judges × 2 criteria on a new rewriter: ~1-1.5h.
8. **Agreement-score regressor**: `scripts/apply_agreement_score.py` — loads a DeBERTa-v3-base regression head (single-logit, sigmoid → [0,1]) from `/home/max/attack-llm-judge/agreement_model/runs/main/final`. On a new host, `huggingface_hub.snapshot_download("daxmavy/attack-llm-judge-agreement-regressor", local_dir=...)` pulls it (712 MB, 4 files: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`). Scores every row in `attack_rewrites` not yet present in `attack_agreement_scores`. Very fast (~150 rows/s on single GPU, ~1 GB VRAM).

**One command that runs steps 1-5 (generation side, not the final scoring pass)**:
```
bash run_full_criterion_with_rewriter.sh <HF_ID> <short_name> <criterion>
```
This chains feedback_free → bon_generate → 3×bon_score → 3×icir (one subprocess per fold) → 3-fold GRPO → HF push + DB backfill.

---

## 8. Versions currently in use

Full pinned list in `requirements.txt` at repo root. Critical pins:

```
torch         2.10.0
vLLM          0.18.1
TRL           1.2.0
transformers  4.57.6
accelerate    1.13.0
datasets      4.8.4
CUDA          12.x  (matches torch 2.10.0 cu128 wheel)
Python        3.11.10
```

Install into a fresh venv:
```
python3.11 -m venv /workspace/pylocal && source /workspace/pylocal/bin/activate
pip install -r requirements.txt
```

Dependencies are **not** managed by poetry/pdm/uv on this host — pure pip into a bare venv. The `agreement_model/train.py` and `agreement_model/run_experiments.py` additionally expect `scikit-learn` and `scipy` for metrics; add those if retraining the regressor (not needed for inference-only runs).

The `transformers` version in the current venv has a known bug loading the DeBERTa tokenizer in *some* environments; if you see a tokenizer error during `apply_agreement_score.py`, fall back to the system `python3` with a plain `pip install transformers tokenizers` (not the pylocal PYTHONPATH).

**vLLM compatibility notes observed:**
- LFM2.5-1.2B-Instruct and Gemma-3-1b-it both work on vLLM 0.18.1 (pre-flight tested).
- Gemma-3-1b-it supports system-role chat templates in this vLLM version. Gemma-2 family did NOT in earlier versions — the code has a `supports_system_role(tokenizer)` guard that falls back to concatenating system text into the user turn. Preserve this pattern for any new rewriter base.
- Qwen3 family reasoning mode is suppressed via `enable_thinking=False` in `apply_chat_template`. Required.

---

## 9. Two-GPU topology (mandatory; architectural gap still open)

The pipeline is designed to run on **2× A100 80GB** (CLAUDE.md GPU-topology rule). The mapping:

- **GPU 0 (judges)**: both `JudgeVLLM` instances live here, pinned via `CUDA_VISIBLE_DEVICES=0`. With a dedicated GPU, `gpu_memory_utilization` can go to 0.45-0.50 per judge (up from the co-located 0.22-0.28). A 14B-class judge fits comfortably; 2 × 14B ≈ 60 GB total.
- **GPU 1 (rewriter)**: GRPOTrainer + rollout vLLM, pinned via `CUDA_VISIBLE_DEVICES=1`. With a dedicated 80 GB you have headroom for **~3-4B full bf16 RL**, or **~15-30B with LoRA**, or **30-70B with QLoRA**. The historical single-A100 runs showed ~30 GB on a 1.5B; the 2026-04-21 14B co-located run consumed all 80 GB because judges shared the device.

Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` in every launch script to avoid nvidia-smi/CUDA index drift on mixed compute-capability hosts. Co-locating rewriter + judges on a single A100 is a **known-bad configuration**: the 2026-04-21 run was forced to rewriter≈0.38 + judges≈0.28 each, which silently overcommits when the judge panel rotates and which contributed to the length-penalty failure in fold 1 (`frac_outside_tol = 0.75`).

### Architectural gap (not yet closed as of 2026-04-22)

`training/scripts/run_pilot_len_pen.py::reward_fn` still instantiates `JudgeVLLM(...)` in the **same Python process** as the GRPOTrainer. All downstream pipeline scripts (`run_icir.py`, `run_mission_attacks.py`, `score_all_missing.py`) follow the same single-process pattern. That means that on a 2-GPU host today, both judges and the rewriter still compete for a single `CUDA_VISIBLE_DEVICES` by default — the repo does not yet have a real split.

To actually execute the dual-GPU topology, one of these refactors is required (pick (a), it's less invasive):

1. **(a) Multiprocessing**: spawn judges in a child `multiprocessing.Process` with `CUDA_VISIBLE_DEVICES=0` set *before* torch/vLLM import, keep the trainer in the parent with `CUDA_VISIBLE_DEVICES=1`. IPC via `multiprocessing.Queue` for prompts / score tuples. Keep the same JSON parsing (`parse_score`).
2. **(b) HTTP judge server**: spawn `CUDA_VISIBLE_DEVICES=0 python3 judge_server.py --judges … --port 8123` as a sibling process, and have `reward_fn` call that endpoint. More code, more failure modes (port conflicts, startup race), but isolates the trainer Python env so the judge server can stay on a pinned vLLM version if the trainer wants a newer one.

Either way, flag it at launch — running 400-step GRPO under co-location is a mission-invalidating mistake.

### Other two-GPU tuning knobs

3. **Judge batch size**: with a dedicated judge GPU, bump `gpu_memory_utilization` to 0.45 and raise `max_model_len` to 4096 or 6144. That roughly doubles the KV-cache concurrency and cuts wall time per scoring call by ~30-40%. `JudgeVLLM.__init__` auto-picks the budget; override via the `EVAL_VLLM_MEM_UTIL` env var.
4. **Rewriter size bump**: if using a 3-4B base with full bf16, raise `--per-device-batch` to 16 (`2 × G = 16` at G=8, or 8 at G=4), and keep `--num-generations 4`. If going to LoRA/QLoRA with a bigger base, reduce G to 2 to control rollout KV.
5. **vLLM version**: if the judge or rewriter architecture needs a newer vLLM than 0.18.1, isolate the two processes' Python environments so one doesn't force a break of the other. Pre-flight smoke-test any rewriter base with `scripts/preflight_rewriter.py`.

---

## 10. Current HF artifacts (for reference/download)

All under `daxmavy/`:

- **Qwen2.5-1.5B base (current overnight run)**:
  - `grpo-400step-fold{1,2,3}` (clarity)
  - `grpo-400step-fold{1,2,3}-informativeness`
  - `grpo-400step-fold1-g8-halfdata-clarity` (G=8 continuation ablation)
  - `grpo-400step-q95only-clarity` (single-judge ablation)
- **LFM2.5-1.2B (new rewriter base)**:
  - `grpo-lfm25-12b-fold{1,2,3}-clarity`
  - `grpo-lfm25-12b-fold{1,2,3}-informativeness`
- **Gemma-3-1b (running)**:
  - `grpo-gemma3-1b-fold{1,2,3}-{criterion}` (will appear as training completes)

Use `huggingface_hub.snapshot_download(repo_id, token=HF_TOKEN)` to restore.

---

## 11. Known gotchas / traps

1. **vLLM memory leak between subprocess runs**: `del` + `empty_cache` does not fully release vLLM engine state. Always spawn a fresh subprocess for each fold if multiple vLLM engines need to load in sequence (e.g., ICIR which loads rewriter + 2 judges, then wants to reload judges for the next fold).
2. **Naive is criterion-agnostic**: its prompt doesn't mention criterion, so the text is identical for clarity and informativeness runs. `cmd_feedback_free` detects this and skips regenerating for the second criterion. Preserve this — it's correct and saves 714 generations.
3. **Judge clamp to 50.0 on parse failure**: `parse_score` returns 50.0 if no JSON/score extractable. Monitor `score_s` in the training log — if it's consistently 50.0 something's broken with the judge's output.
4. **BoN candidate rewrite_id vs panel rewrite_id**: bon_panel does NOT share rewrite_id with bon_candidate. The panel row's `run_metadata_json` has `selected_candidate_rid` pointing back to the source candidate, and the text is identical. If you need per-panel judge scores, join on text + source_doc_id OR on the `selected_candidate_rid` field, rather than rewrite_id.
5. **GRPO `judge_panel_json` column**: currently only the held-out judge is saved there for `method='grpo_400step'` rows. The 2 in-panel judges that trained the model are inferred from the fold via the FOLDS dict. Don't rely on `judge_panel_json` for GRPO rows.
6. **Disk quota**: the single-GPU workspace has 200GB. Each final GRPO model dir is ~3GB, so 6 per rewriter = 18GB. `ckpt_*` dirs from intermediate saves are auto-overwritten with `save_strategy="no"`. Push to HF, then delete `final_*` locally when the DB backfill + HF round-trip-verify are both confirmed.

---

## 12. What "replication" means for the other server

At minimum, the other agent should:
1. Use the same `controversial_40_3fold.json` dataset (copy it over, don't regenerate).
2. Use the same 3 mission judges (same HF model IDs). If they want larger judges, keep the fold rotation structure identical but swap the model IDs; record the swap in `rewriter_model` / `judge_panel_json` for traceability.
3. Use the same 6 attack methods with the same prompt templates and hyperparameters.
4. Produce the same DB schema (`attack_rewrites`, `attack_judge_scores`, `attack_agreement_scores`) so results can be merged into this DB or compared side-by-side.
5. Push GRPO models to HF under a distinct namespace (e.g. `{username}/grpo-2gpu-...`) to avoid conflicts with the current single-GPU run's repos.

The **diff they can reasonably introduce**, and flag in their notes:
- vLLM / TRL version bumps (compatible with their hardware / larger models).
- Rewriter base model size (3-4B full bf16 or bigger with LoRA/QLoRA).
- Judge model size (14B-class instead of 9B).
- GRPO `num_generations` (could bump to 8 with more VRAM headroom on GPU 1).
- `per_device_train_batch_size` scaling.
- `max_model_len` for judges (3072 → 4096/6144 with more KV headroom).

The **diff they should NOT introduce** if the goal is direct comparability:
- Change the 40 propositions or the train/eval split.
- Change the attack method prompt templates.
- Change the length/fidelity penalty formulas.
- Change the judging rubric text.
- Change the `enable_thinking=False` setting for Qwen3 judges.

---

## 13. Quick smoke-test recipe (before long runs)

For any new rewriter base:

```
python3 /home/max/preflight_rewriter.py --models <new_rewriter_hf_id>
```

That runs: (a) vLLM load + chat-template + generate 1 paragraph, (b) single GRPOTrainer step with a live judge. Both must pass before committing to a full 3-fold run. Outputs a markdown report at `PREFLIGHT.md`.

---

## 14. Contact for questions

Max Davy. The code is in the `main` branch of this repo. Experiment notes (design decisions, bug fixes) live in `EXPERIMENT_NOTES.md`. The paper being written from this is at `paper/main.tex`.
