# EXPERIMENT_NOTES.md — iteration log, design decisions, bugfixes

Append-only running log. Newest entries at the bottom of each section. Cross-reference design decisions by their **D-N** label and bugfixes by **B-N** so other docs (e.g. STRUCTURE.md §7) can point at them stably.

For the research framing those decisions sit inside, see `STRUCTURE.md`.
For prior art, see `LITERATURE.md`.
For model identities and exact slugs, see `MODELS.md`.

---

## A. Major design decisions

### D-1. Single fold, 2 proxy judges + 1 held-out (replacing original 5-attack + 5-gold panel design)
**Date:** 2026-04-17 (mid-day).
**Original spec:** 5-judge attack panel (small, family-diverse) + 5-judge gold panel (strong) with leave-one-out generalisation across the 5 attack judges.
**Revised:** 2 fixed proxy judges + 1 held-out judge for the *transfer* axis, plus the gold panel kept separate at its own role.
**Why:** the LOO design fragments the unknown-judge signal across 5 cells, each with weak statistical power at the available sample size; collapsing to a fixed proxy/held-out trio concentrates the signal and makes the cross-method comparison legible.
**Cost:** one fold means no error bar on the unknown-judge penalty across judges. Mitigation: report within-panel LOO over the 2 proxy judges as a robustness check (cheap).

### D-2. Method panel cut to 5 (from 10)
**Date:** 2026-04-17 (late afternoon).
**Original spec:** 10 inference-only methods (`naive`, `lit_informed`, `naive_tight`, `lit_informed_tight`, `injection_leadin`, `rules_explicit`, `rubric_aware`, `scaffolded_cot_distill`, `icir_single`, `bon_panel`).
**Revised:** 5 methods chosen to span the method-complexity axis: naive, lit-informed, BoN, SFT, GRPO. See `STRUCTURE.md` §4.
**Why:** the 10-method panel is a wide horizontal comparison without depth on training-based methods, which are required for any "ranking reversal" prediction. The 5-method panel keeps one method per family (prompt-only / inference-search / offline-training / online-RL).
**Status of dropped methods:** code remains in `rewriters/` from the earlier sprint. May appear in an appendix table if useful but not in headline charts.

### D-3. Training-based arms (SFT, GRPO) added to scope
**Date:** 2026-04-17.
**Reason:** without weight-updating methods, the headline ranking-reversal prediction is untestable. Inference-only methods cluster too tightly on unknown-judge transfer (most prompt tweaks raise quality universally).
**Risk to manage:** GRPO may reward-hack the proxy judges in ways that don't transfer (One Token to Fool, arXiv:2507.08794). Tripwires on rising proxy reward + rising drift / detector signals are required diagnostics.

### D-4. Reward model dropped from all methods
**Date:** 2026-04-17 (most recent).
**Original spec:** every judge-feedback method (BoN, SFT, GRPO) had two variants: (a) learned reward model distilled from the 2 proxy judges, (b) online proxy-judge ensemble.
**Revised:** only the online variant remains.
**Why:** RM arms add methodological depth that is not central to the research question ("does optimising against proxy judges transfer to an unknown judge"). The online variant directly answers it; the RM variant adds resolution on *how* an attacker would do it under realistic budget constraints — interesting but secondary. Cutting reduces the experiment grid from 8 cells to 5 and drops the iterated-RM-refresh / RM training time entirely.
**Cost:** loses the "RM distribution-shift hacking" sub-question and the cheap-fallback hedge if online-GRPO hits API rate limits. If online-GRPO turns out to be infrastructure-fragile in practice, RM-GRPO may need to come back.

### D-5. Keep SFT alongside GRPO
**Date:** 2026-04-17 (most recent).
**Decision:** despite scope pressure, keep SFT as a separate arm.
**Why:** SFT and GRPO probe different regimes — *imitation of BoN-selector preferences* vs. *online exploration that can drift to arbitrary high-RM regions of token space*. Without both, a null result on either is uninterpretable. SFT is also cheap (hours, not tens of GPU-hours) and shares infrastructure with BoN.
**Pre-registered prediction:** SFT achieves lower uplift but smaller transfer gap than GRPO. Either outcome is publishable; reversal would be more surprising.

### D-6. Online judge = OpenRouter for GRPO rollouts
**Date:** 2026-04-17.
**Decision:** during online-GRPO, query the 2 proxy judges via OpenRouter rather than running them locally on the same A100 as the rewriter.
**Why:** empirical benchmark (see B-7) showed OpenRouter at conc=16 hits ~5 calls/sec vs ~4.25 calls/sec for local HF transformers at bs=16, while *not* stealing GPU cycles from the policy's forward/backward pass. ~20% throughput deficit is more than offset by the architectural separation.
**Revisit if:** vLLM benchmark (still pending) shows large enough margin to justify co-tenancy, or a second GPU is added.
**Risk:** ties online-GRPO availability to OpenRouter's rate limits and uptime. See R-1 in §C.

### D-7. AI detector — supervised classifier, not zero-shot
**Date:** 2026-04-17.
**Decision:** the AI-detection metric is a supervised TF-IDF (word 1-2g + char 3-5g) → LogisticRegression trained on `paul_data` writer-vs-model split. AUROC 0.989 held-out (group-by-proposition).
**Original plan:** Binoculars (Falcon-7B + Falcon-7B-Instruct, ~28GB combined).
**First fallback:** Fast-DetectGPT-lite with GPT2-XL, then Qwen-2.5-1.5B as the scoring LM.
**Why supervised won:**
- Binoculars: requires ~28GB on disk; the RunPod overlay disk is 20GB. Vendored at `eval_suite/binoculars/` for if the disk ever grows.
- Fast-DetectGPT with GPT2-XL: AUROC 0.60. With Qwen-2.5-1.5B: AUROC 0.57. Zero-shot detectors need a scoring LM of comparable capability to the AIs they're detecting (here Claude Sonnet 4 / DeepSeek V3 / GPT-4o-latest); we have no disk for a comparable scoring LM.
- Supervised classifier on the labelled writer-vs-model split is more honest and dramatically better.
**Caveat to write into limitations:** the supervised detector learns the stylistic signatures of Claude Sonnet 4 / DeepSeek V3 / GPT-4o-latest specifically. A rewriter that fools it may just be moving away from those three signatures, not becoming generally human-like.

### D-8. Gold panel cost-downgraded to fit $100 budget
**Date:** 2026-04-17.
**Original spec:** `claude-sonnet-4.6`, `gemini-2.5-pro`, `gpt-5-mini`, `deepseek/deepseek-v3`, `meta-llama/llama-3.1-405b-instruct`.
**Revised:** `claude-haiku-4.5`, `gemini-2.5-flash`, `gpt-5-mini`, `deepseek/deepseek-v3.2`, `meta-llama/llama-4-maverick`.
**Reasons:** Sonnet 4.6 + Gemini 2.5 Pro on the full corpus blew the $100 cap. Llama-3.1-405B is not on OpenRouter; Llama-4-Maverick is the closest available substitute.
**Cost to write into limitations:** "gold" is family-diverse but not frontier. Don't claim "frontier reference" in the paper.

### D-9. Reasoning-model handling in the judge client
**Date:** 2026-04-17.
**Issue:** Gemini 2.5 Pro and GPT-5-mini emit hidden reasoning tokens that consume `max_tokens` before the answer fits.
**Fix:** `judge/client.py` recognises a `REASONING_MODELS` set and, for those, sets `reasoning.effort=low` and bumps `max_tokens=500`. Other models unaffected.

### D-10. Word-count fidelity tightening
**Date:** 2026-04-17.
**Issue:** v1 rewriters (`naive`, `lit_informed`) only hit ±10% of original word count 56% / 28% of the time; lit_informed had 25% of outputs under 0.80× original — large enough to confound clarity uplift with simple compression.
**Fix:** added `naive_tight` and `lit_informed_tight` with explicit min/max anchors and a one-shot length-retry in `run_rewrite.py` (re-prompt with the actual count if outside ±10%).
**Result:** within-±10% climbed to 90% / 76%; under-0.80× collapsed to 0% / 1%; clarity deltas held or improved slightly. Length control did not cost quality.
**Implication for current scope:** the headline 5 methods inherit this pattern — naive and lit-informed in the panel are the *tight* variants.

### D-11. Top-decile flag for sample stratification
**Date:** 2026-04-17.
**Definition:** within each (proposition × agreement-score quintile), flag the top 10% by `mean_human_clarity`. Per-quintile counts: 917/898/889/901/898 writers; ~100 top-deciles per quintile.
**Why this metric:** clarity is the in-scope criterion, so ranking exemplars within a stance quintile by clarity is the right signal. If informativeness is added, a parallel flag should be computed using `mean_human_informativeness`.

### D-12. RM iterated-refresh design (now obsolete after D-4)
**Date:** 2026-04-17 (designed); deprecated 2026-04-17 by D-4.
**Originally specified:** after ~1 epoch of GRPO, draw ~500 on-policy rewrites, label with the 2 in-panel judges, append, retrain RM (~30 min DeBERTa-v3-base), continue. Guards against the biggest RM risk: distribution shift, not dataset volume.
**Status:** retained here for context. If the RM arms come back (see D-4 reversal condition), this is the refresh recipe.

---

## B. Bugfixes and infrastructure quirks

### B-1. OpenRouter `response_format: json_object` not universally supported
**Symptom:** some OpenRouter providers reject the `json_object` response_format with HTTP 400.
**Fix:** judge client retries the call once without the parameter. Logged as a soft warning, not a failure.

### B-2. OpenRouter 200-OK with no choices
**Symptom:** OpenRouter occasionally returns HTTP 200 but with an empty `choices[]` array under provider quota blips. Caused the rewriter run to crash on `rubric_aware` mid-run.
**Fix:** rewriter client now handles empty-choices as a transient error and retries.

### B-3. SQLite locks under concurrent writers
**Symptom:** parallel API workers writing to `data/paragraphs.db` triggered `database is locked` errors intermittently.
**Fix:** main thread batches writes after parallel API calls complete. Don't pass the same `sqlite3.Connection` across threads. WAL journaling is on but isn't sufficient on its own.

### B-4. Working directory required for package imports
**Symptom:** `python3 -m eval_suite.<module>` fails with `ModuleNotFoundError` when invoked from elsewhere.
**Fix:** `cd /home/max/attack-llm-judge` before any `python3 -m eval_suite.*`. The package needs the project root on `sys.path`.

### B-5. RunPod overlay disk is 20GB (not unlimited)
**Symptom:** `pip install` of large dependencies fills the root filesystem and fails.
**Fix:** install with `PYTHONUSERBASE=/workspace/pylocal pip install --user <pkg>`. `/workspace` is the persistent 830TB volume; root fs is the 20GB ephemeral overlay.

### B-6. Model weights not committable
**Symptom:** trained DeBERTa weights are >100MB; `git add` fails.
**Fix:** `.gitignore` excludes `*.safetensors`, `criterion_model/*/final/`, `criterion_model/*/checkpoint-*/`. Push only code; weights live on `/workspace`.

### B-7. API vs local judge throughput benchmark
**Date:** 2026-04-17.
**Setup:** Qwen-2.5-7B-Instruct, real rubric, 30 paul_data paragraphs.

| Path | calls/sec | out-tok/sec | 1k calls |
|---|---|---|---|
| OpenRouter @ conc=16 | 5.06 | 328 | 3.3 min |
| Local HF transformers bs=16 (bf16, A100-80GB) | 4.25 | 275 | 3.9 min |
| Local vLLM (installed, pending benchmark) | ~10–12 est. | ~700–900 est. | ~1.5–2 min est. |

Drove decision D-6.

### B-8. Judge panel slugs needed swapping (retired models)
**Date:** 2026-04-17.
**Original attack panel:** `meta-llama/llama-3.1-8b-instruct`, `qwen/qwen-2.5-7b-instruct`, `mistralai/mistral-7b-instruct`, `google/gemma-2-9b-it`, `microsoft/phi-3.5-mini-instruct`.
**Revised attack panel** (after OpenRouter retirement / hosting issues):
- `mistralai/mistral-7b-instruct` → `mistralai/ministral-8b-2512`
- `google/gemma-2-9b-it` → `google/gemma-3-4b-it`
- `microsoft/phi-3.5-mini-instruct` → not on OpenRouter; tried NVIDIA Nemotron (returned empty content under `response_format=json_object`); landed on `x-ai/grok-3-mini`
**Note:** since D-1 cut to 2 proxies + 1 held-out, only a subset of these are still active. See `MODELS.md` for the live trio.

### B-9. `gh` not in package manager (no sudo on RunPod)
**Fix:** installed manually to `~/.local/bin/gh`. Operator must run `~/.local/bin/gh auth login` once to unblock GitHub push.

---

## C. Open risks (live)

### R-1. Online-GRPO is single-point-of-failure under D-4 + D-6
With RM cut and judges on OpenRouter, GRPO's training loop depends on OpenRouter's rate limits and provider uptime for ~72k rollouts (G=8 × ~4.5k training prompts × 2 epochs per estimate from the original training plan; the training agent should confirm actual numbers). One bad provider hour mid-run can wedge the experiment.
**Mitigations to consider:** keep RM-GRPO as a hedge (re-introduces D-4 trade-off), or cache rollout responses aggressively, or pre-warm a vLLM fallback for the judges.

### R-2. Clarity dynamic range may null out the headline
Pilot data: clarity v1 deltas of +5 to +8 across methods, judge-judge Pearson 0.90. If the held-out vs proxy gap is <1 judge-score point, it sits inside judge-judge noise and the paper has no significance. Monitor early, decide whether to add informativeness for training-based arms.

### R-3. Agreement-score regressor may not catch the failure mode that matters
The DeBERTa-v3 regressor was trained on `paul_data` (writer + model + edited). It generalises across AI models in the training distribution (LOO Pearson 0.94–0.96), but its hardest subgroup is *human writers* — exactly what we're rewriting. A trained rewriter may push outputs into a region where the regressor extrapolates poorly. Cross-check rewrites against held-out human-judged subsets if any exist.

### R-4. Reward hacking on online-GRPO
Standard concern (Pan et al. 2024 ICRH; One Token to Fool 2507.08794). Required diagnostics: rising proxy reward with rising drift, hallucinated-specifics rate, AI-detector confidence. Promote "rising score + rising drift" as a headline diagnostic, not a side metric.

---

## D. Decision log shorthand

When adding a new entry to A or B above, also append a one-line shorthand here so the running narrative stays scannable:

- 2026-04-17 — D-1 narrow panel to 2+1+5; D-2 cut method panel to 5; D-3 add training-based arms; D-10 word-count tightening; B-1/B-2/B-3 OpenRouter quirks
- 2026-04-17 — D-7 AI detector switched to supervised; D-8 gold panel cost-downgraded; D-9 reasoning-model handling
- 2026-04-17 — B-7 API vs local benchmark; D-6 OpenRouter for online-GRPO
- 2026-04-17 — D-4 RM dropped from all methods; D-5 SFT kept alongside GRPO; D-12 RM-refresh deprecated
- 2026-04-18 — B-10 TRL `use_vllm=True` IS-ratio collapse; D-13 length-penalty in reward; D-14 proven-path (vLLM judges + HF rewriter) adopted for tonight's mission

---

## Tonight's mission log (2026-04-17 evening → 2026-04-18 morning; deadline 09:00 UTC)

### Problem at start of mission
Successful GRPO runs earlier in the day showed +5–8 reward Δ on training judges + ~40% length increase — reward gain is partly length-bias. User wants: (a) replicate reward climb with vLLM; (b) pilot reward climb without length drift; (c) full training before deadline.

### B-10. TRL `use_vllm=True` + vllm 0.18.1 + torch 2.10 numerical drift
**Symptom.** Rewriter loss function's importance-sampling ratio collapses from ~0.97 (step 1) → 0.065 (step 5). `sampling_logp_difference` mean grows 0.025 → 0.17 over 5 steps, max reaches 3.4. Reward does not climb; slightly decreases.
**Diagnosis.** vLLM-sampled token logprobs and HF-trainer token logprobs diverge rapidly as the policy updates. Cap at 3.0 means many tokens get zeroed out in the policy update via IS correction. Observed in `run_min_repro_vllm.py` (25-step attempt) and confirmed in a 5-step targeted diagnostic.
**Attempted fixes.** `vllm_enable_sleep_mode=False` (test A): IS ratio *was* healthy at step 1 (0.97) but trainer OOM'd on step 2 — keeping the rewriter vLLM resident during optim step pushes trainer process to 40 GB alone (vs ~20 GB with sleep_mode). Didn't get to 5 steps to see if drift still happens.
**Decision (D-14).** Skip further debugging of this TRL integration path for tonight. Fall back to the proven **vLLM judges + HF rewriter** path (`run_min_vllm.py`), which showed reward climb earlier today (+3.48 Qwen / +8.56 Llama on 25 steps).

### D-13. Length-penalty in reward (tolerance-band additive)
**Formula.** `reward = mean(judge1, judge2) − α × max(0, |len_ratio − 1| − tol)` where `len_ratio = generated_wc / original_wc`. Defaults `α=25, tol=0.10`.
**Intuition.** Inside ±10% tolerance no penalty; at ±40% deviation penalty is 7.5 points (≈ the length-driven gain seen in the HF run).
**Module.** `/workspace/grpo_run/length_penalty.py`. Wired into `/workspace/grpo_run/run_pilot_len_pen.py` via a custom `reward_fn`. CLI args `--alpha`, `--tol`, plus GRPO knobs `--loss-type`, `--scale-rewards`, `--temperature`, `--beta`, `--lr` for Dr-GRPO style sweeps.

### D-14. Proven-path strategy for tonight
Use `run_min_vllm.py` pattern (vLLM for judges only, HF rewriter via TRL's default `model.generate`). Add length penalty. Run 5-step smoke → 15-step pilot → full run. Full run scope determined by time budget (target: finish before 09:00 UTC).
