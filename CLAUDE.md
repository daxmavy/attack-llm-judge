# Project instructions

- Read `STRUCTURE.md` before making any changes. It is the source of truth for what to work on next.
- Make small, focused git commits for the changes you make. Prefer many small commits over one large one.
- Push all commits to Github remote, so that if this server is lost for some reason I can recover the code.
- Add experiment notes (design decisions, bugfixes, etc) to EXPERIMENT_NOTES.md
- Literature review is at LITERATURE.md
- default to autonomy — act first, ask only when a decision is genuinely blocking, irreversible, or ambiguous in a way that cheap exploration can't resolve. Don't ask just because you can.

## Mission status (2026-04-17 → 2026-04-18)

**All three mission goals achieved by 03:49 UTC** (5h 10min before the 09:00 UTC deadline):

- **(a) vLLM working** — via the fallback path (vLLM judges + HF rewriter). TRL `use_vllm=True` rewriter path was NOT solved; its IS-ratio collapse bug remains open.
- **(b) Length-controlled pilot** — quadratic penalty `α·(r−1)²` with α=100 (not the tolerance-band additive) converged to ratio ~1.0. This config frozen for the full run.
- **(c) Full 3-fold training** — all three leave-one-out judge rotations trained, held-out evaluated, and pushed to HF (`daxmavy/attack-llm-judge-grpo-fold{1,2,3}-20260418`, private). Held-out Δ: +8.14 / +6.29 / +6.23 (mean +6.89).

Full mission log in `EXPERIMENT_NOTES.md` under "Tonight's mission log".

## Follow-up investigations (pending)

Ordered by how cheaply each resolves an open question about the results.

### A. Gold-panel + drift evaluation on the 3 trained checkpoints (highest priority)

The mission proves "in-panel reward climbs and held-out judges agree" but does NOT prove "the rewriter learned real clarity vs gamed judge preferences." Need to run on the ~105 post-train rewrites (35 eval prompts × 3 folds):
1. **Gold panel** (Claude Haiku 4.5, Gemini 2.5 Flash, GPT-5-mini, DeepSeek V3.2, Llama-4-Maverick) — ~\$3–5 on OpenRouter, ~5 min.
2. **Agreement-score regressor** (`agreement_model/runs/main/final/`) for stance drift — ~30 sec on GPU.
3. **AI-generation detector** (`data/ai_detector_tfidf.pkl`) for "this now sounds more AI-written than the original" signal — ~30 sec.
4. **Embedding-cosine similarity** to original paragraph — ~1 min.
5. **Hallucinated-specifics rate** (regex + spaCy NER) — ~1 min.

Together these produce the 4-quadrant gold-vs-in-panel decomposition chart described in `STRUCTURE.md` §5.

### B. Fix the hardcoded wandb key bug in `reward_fn`

`training/scripts/run_pilot_len_pen.py`'s `reward_fn` uses **hardcoded** keys `reward/judge_qwen7b_mean` and `reward/judge_llama8b_mean` regardless of which judges the ensemble actually contains. For fold 2 (Qwen+Gemma) and fold 3 (Llama+Gemma), Gemma's scores are logged under those hardcoded names (the user noticed Gemma was missing from charts). Fix: dynamic keys like `reward/{judge.name}_mean`. Re-log fold-2/3 summary stats afterwards if needed.

### C. Qualitative review of stance drift

Looking through fold-1 post-train rewrites already surfaced suggestive stance flips (e.g. writer argued FOR trans-recognition → trained-rewriter argues AGAINST). Quantify with the agreement regressor (see A2) and then hand-inspect cases where `|agreement_post − agreement_pre| > 0.15` to confirm whether they're flips, softening, or regressor noise.

### D. Tried-but-untested knobs

None of these were exercised in the mission; all have CLI flags already in `run_pilot_len_pen.py`:
- **Dr-GRPO** (`--scale-rewards none --loss-type dr_grpo`): targets low within-group variance, which we measured as `frac_reward_zero_std` climbing to ~0.25 by end of fold 1. Worth a 25-step pilot.
- **Higher rollout temperature** (`--temperature 1.2` or 1.3): widens within-group diversity → cleaner group-relative advantage.
- **Higher `num_generations`** (G=8 instead of 4): cuts the rate of zero-variance groups directly. Memory-feasible if `--per-device-batch 4`.

### E. Longer-training pilot on winning config

Pilots showed reward plateauing by ~step 20 and the 200-step fold runs showed further gain but also growing stylistic homogenisation ("The proposition that the United Kingdom should…" opener everywhere). A 500–1000-step run on one config (probably fold 1 recipe) would clarify whether more training keeps improving held-out transfer or just deepens the template collapse.

### F. Save individual G=4 rollouts for diversity analysis

Current `eval_summary.json` only contains the single deterministic post-train rewrite per prompt. To compute token-level or semantic diversity within G=4 rollouts (what actually drives GRPO's advantage signal), the training script needs a `--save-rollouts` flag that dumps each step's per-rollout text + reward to JSONL. ~10-line code change.

### G. Re-investigate TRL `use_vllm=True` with sleep_mode=False + smaller per_device

The sleep_mode=False attempt OOM'd at step 2 before we could see whether the IS-ratio collapse persisted. With smaller per_device (4 or even 2) the footprint fits; worth a clean 5-step diagnostic to settle whether the IS issue is sleep_mode-specific or fundamental kernel drift. If fundamental, we're stuck on vLLM judges + HF rewriter for real training (3–4× slower) and need a different plan for scaling.

### H. Broaden corpus beyond top-decile

All training was on the top-10% clearest human writers (deliberate — least headroom for honest improvement, so any reward gain is more likely gaming). For a paper claim, repeat the experiment on bottom-decile or uniform-sampled writers and compare the gold-vs-in-panel decomposition.

## Operating rules (still active)

- Speed matters; inspect after 5 steps, not 25; make git commits after each milestone; manage VRAM aggressively (OOM kills the run); write notes to `EXPERIMENT_NOTES.md` as incidents happen.
- **Run-monitoring rule:** every time you launch a training or evaluation run, schedule a check-in for yourself every **5 minutes** until the run has either concluded or been killed. "Check-in" = read the log tail, confirm the process is alive, look for OOM / errors / stalled progress, and grab the latest metric values. Don't let a run drift for 20+ minutes without looking — the feedback loop is what catches silent failures (IS-ratio collapse, length drift, OOM) early. If nothing has changed in the log since the last check, note that and look again in another 5 minutes.