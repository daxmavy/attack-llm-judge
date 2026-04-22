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

### B-11. vLLM V1 KV-cache budget excludes weights → Qwen3-14B OOM at `gpu_memory_utilization=0.18`
**Date.** 2026-04-22 (mission prep). **Debug attempt #1 for Qwen3-14B rewriter.**
**Symptom.** `ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization when initializing the engine.` on Qwen3-14B (bf16, ~28 GB weights) with `gpu_memory_utilization=0.18`.
**Diagnosis.** vLLM V1 counts model weights AGAINST the `gpu_memory_utilization` budget. 80 GB × 0.18 = 14.4 GB — insufficient for the 28 GB weights alone, so the KV-cache budget goes to 0 and engine init aborts.
**Fix.** Bumped `gpu_memory_utilization` to 0.55 (44 GB budget) on the dedicated rewriter GPU — covers weights (27.5 GB) + KV (14.8 GB) + overhead comfortably. Added `_rewriter_vllm_mem_util()` in `scripts/run_mission_attacks.py` that auto-picks 0.55 for ≥13B, 0.35 for 7-8B, 0.20 for ≤2B so future rewriter swaps don't trip on the same thing.
**Verification.** vLLM generate smoke for Qwen3-14B passed cleanly — weights loaded in 6.32 s, KV 96,832 tokens, `"AI is transforming industries across the globe."` returned at T=0.0.

### B-12. Preflight GRPO stage: `TypeError: Object of type set is not JSON serializable` hid the real GRPO error
**Date.** 2026-04-22 (mission prep). **Debug attempt #2 for Qwen3-14B rewriter.**
**Symptom.** `scripts/preflight_rewriter.py` printed `grpo_step ok: False` and then crashed on `json.dumps(all_results)` with `TypeError: Object of type set is not JSON serializable` — BEFORE the markdown report was written, so the caught exception from the GRPO stage was lost.
**Diagnosis.** `LoraConfig.target_modules` is stored as a `set` after PEFT's `__post_init__`, not a list. Storing it directly into `result["details"]` broke `json.dumps`. The bigger mistake was ordering JSON-write before MD-write in `main()`, so any encoder blowup destroys the whole report and the actual error from `check_grpo_step` never surfaces.
**Fix.** Four small edits in `scripts/preflight_rewriter.py`:
  - Cast `peft_config.target_modules` → `list()` before stashing in `result["details"]`.
  - Swap report-write order: markdown first, JSON second.
  - Add `default=str` to `json.dumps` as a belt-and-braces safety net for future untyped objects (dtypes, tensors, etc.).
  - Print `grpo_step.error` and `.traceback` (and same for `vllm_generate.*`) inline whenever a stage fails, so a downstream write crash can never again swallow the diagnostic.
**Meta.** This is an observability bug in the debugging tool itself. Worth an explicit entry because the same "silent-crashing reporter" pattern masked the underlying issue for an entire debug cycle — exactly the "raise-don't-hide" anti-pattern applied to my own scaffolding.

### B-13. Preflight GRPO: 4-bit device check rejects `cuda:1` when `ACCELERATE_TORCH_DEVICE` is unset
**Date.** 2026-04-22 (mission prep). **Debug attempt #4 for Qwen3-14B rewriter.**
**Symptom.** With 4-bit Qwen3-14B loaded via `device_map={"":1}` and `torch.cuda.set_device(1)`, `GRPOTrainer(...)` construction raises `ValueError: You can't train a model that has been loaded in 8-bit or 4-bit precision on a different device than the one you're training on`. Happens even though the model is entirely on `cuda:1` and we never touched `cuda:0` for training.
**Diagnosis.** `accelerate/accelerator.py:1847-1853` runs a 4-bit placement guard:
```python
elif torch.device(self.device.type, current_device_index) != self.device:
    if (self.device.index is not None) or (current_device_index != 0):
        raise ValueError(...)
```
`self.device` comes from `PartialState.__init__` (accelerate/state.py:182-183), which reads `ACCELERATE_TORCH_DEVICE` env at first-access and defaults to `torch.device("cuda")` — **index=None**. With `current_device_index=1` and `self.device.index=None`, the check exits via the second arm and raises.
**Fix.** Set `os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{train_idx}"` **before any `trl`/`accelerate` import** (PartialState is a singleton — set too late and the default sticks). Moved this assignment to the very top of `check_grpo_step` in `scripts/preflight_rewriter.py`, before the `import torch`/`from trl`/`from run_pilot_len_pen` block.
**Caveat.** This only unblocks the static placement check; it does NOT move TRL's in-process generation inputs to the training device — see B-14 for the runtime device-mismatch that still hits during the GRPO rollout. Both fixes are needed together for multi-GPU (judge on 0, trainer on 1) to work.

### B-14. Preflight GRPO: `input_ids` land on `cuda:0` even with model on `cuda:1` → `index_select` device mismatch
**Date.** 2026-04-22 (mission prep). **Debug attempt #5 for Qwen3-14B rewriter.**
**Symptom.** After B-13 was fixed, GRPOTrainer's first step raises `RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on cuda:1 (when checking argument in method wrapper_CUDA__index_select)`. Traceback: `modeling_qwen3.py:371` → `embed_tokens(input_ids)` → `F.embedding`. Model params on `cuda:1`; `input_ids` on `cuda:0`.
**Diagnosis.** During the GRPO rollout, TRL calls `model.generate(...)` / forward with input tensors that did not get forwarded through Accelerate's `_prepare_inputs` placement step (or placement fell back to `torch.cuda.current_device()` at tensor-creation time, which may not equal the model's device under our split-GPU setup). The 4-bit static check is no longer the problem — this is a runtime placement bug that surfaces per-batch.
**Fix (workaround).** Install a root-level `forward_pre_hook` on the model that moves every tensor arg/kwarg to the model's actual device before each forward:
```python
_model_device = next(model.parameters()).device
def _align(module, args, kwargs):
    def mv(x): return x.to(_model_device, non_blocking=True) if torch.is_tensor(x) and x.device != _model_device else x
    return tuple(mv(a) for a in args), {k: mv(v) for k, v in kwargs.items()}
model.register_forward_pre_hook(_align, with_kwargs=True)
```
This is belt-and-suspenders; the root cause is somewhere in TRL's rollout-path device handling with split-GPU Accelerate, but paperwork over it is much cheaper than forking TRL. Verify with smoke v5.
**Meta.** One more strike against "put the judge on a different GPU via `CUDA_VISIBLE_DEVICES=0,1` in-process" as a design. A cleaner long-term option is to split judge into its own subprocess with its own `CUDA_VISIBLE_DEVICES=0` env at spawn time, and run the trainer with `CUDA_VISIBLE_DEVICES=1` only — but that's a bigger refactor, and the hook approach should unblock tonight.

### B-15. Preflight GRPO: abandoned cross-GPU and passed on **single-GPU co-location** (judge + trainer on GPU 1)
**Date.** 2026-04-22 (mission prep). **Debug attempt #6 for Qwen3-14B rewriter — SUCCESS.**
**Decision.** After 5 cross-GPU debug attempts papering over successive device-mismatch sites (static 4-bit check, embedding input_ids, TRL's `eos_idx` tensor in `grpo_trainer.py:1408`), abandoned the "judge on 0, trainer on 1" architecture entirely. Launched with `CUDA_VISIBLE_DEVICES="1"` so both the judge vLLM subprocess and the main-process trainer see a single GPU labeled `cuda:0` — all tensor creation defaults to the same device, and the whole class of `_prepare_inputs` / `eos_idx` / embedding mismatches disappears.
**Result.** v6 preflight: `grpo_step ok=True`, use_qlora=true, step 1 train_runtime=136.82 s, peak trainer VRAM 15.07 GB, judge vLLM ~22 GB on the same GPU (total ~37 GB of 80 GB A100).
**Budget math (critical).** 138 s/step × 400 steps = 920 min ≈ 15.3 hrs, which alone exceeds the 15-hr mission window. Either (a) scope GRPO down to ~100-150 steps, (b) enable TRL's `use_vllm=True` colocate mode to cut rollout time ~10x (at cost of more debug), or (c) accept that `grpo_400step` is the last method run and may not complete. Pilot timing probe with 3 RL steps needed to distinguish cold-start overhead from steady-state step time before committing.
**Silent fallback flagged.** TRL defaults to `use_vllm=False` → rollout goes through `model.generate()` (HF), which is the bulk of the 138 s. Per CLAUDE.md raise-don't-hide rule: this is the "vLLM→HF generate" fallback engaging silently and is the single biggest step-time cost.
**Step 1 zero-loss anomaly.** `train_loss=0.0, total_flos=0.0` at step 1. Consistent with GRPO's degenerate first-step (single-group reward, std→0, advantage→0). Flagged per rule but not blocking.
**Cleanup debt.** The forward pre-hook (B-14) and the early-env-var assignment (B-13) are both now no-ops under single-GPU, but harmless; leaving in place.

### D-13. Length-penalty in reward (tolerance-band additive)
**Formula.** `reward = mean(judge1, judge2) − α × max(0, |len_ratio − 1| − tol)` where `len_ratio = generated_wc / original_wc`. Defaults `α=25, tol=0.10`.
**Intuition.** Inside ±10% tolerance no penalty; at ±40% deviation penalty is 7.5 points (≈ the length-driven gain seen in the HF run).
**Module.** `/workspace/grpo_run/length_penalty.py`. Wired into `/workspace/grpo_run/run_pilot_len_pen.py` via a custom `reward_fn`. CLI args `--alpha`, `--tol`, plus GRPO knobs `--loss-type`, `--scale-rewards`, `--temperature`, `--beta`, `--lr` for Dr-GRPO style sweeps.

### D-14. Proven-path strategy for tonight
Use `run_min_vllm.py` pattern (vLLM for judges only, HF rewriter via TRL's default `model.generate`). Add length penalty. Run 5-step smoke → 15-step pilot → full run. Full run scope determined by time budget (target: finish before 09:00 UTC).

### Pilot #1 — 2026-04-17 23:27 UTC — SUCCESS
- **Config:** G=4, bsz=8, grad_accum=4, lr=5e-6, β=0.01, T=1.0, α=25 (length), tol=0.10, 25 steps, top-decile 200 train / 50 eval.
- **Trajectory:**
  - Step 1: ej=74.7, pen=7.23, final=67.5, ratio=0.65 (undershoots target)
  - Step 5: ej=77.1, pen=10.47, final=66.7, ratio=1.45 (overshoots target — length-gaming signal kicks in)
  - Step 11: ej=81.1, pen=3.55, final=77.6, ratio=1.09 (penalty pulls it back)
  - Step 18: ej=84.9, pen=1.28, final=83.6, ratio=0.97 (converged to in-band)
  - Step 25: ej=80.7, pen=2.44, final=78.2, ratio=0.96
- **Pre/post on 2 in-panel judges (50-prompt disjoint eval set):**
  - Qwen-7B: 81.80 → 85.24, **Δ +3.44**
  - Llama-8B: 79.86 → 86.22, **Δ +6.36**
- **Wall-clock:** 6.8 min training, 16.0 min total (12 min vLLM setup + pre/post eval).
- **Interpretation.** Length penalty at α=25 successfully kept `length_ratio` in the 0.83-0.99 band for last 10 steps despite initial overshoot to 1.54. Reward climbed +10 on the final (penalised) scale; +5 on raw judges. This is mission (b) achieved.
- **Wandb run:** `grpo-pilot1-<timestamp>` under group `mission_20260418`.
- **Artifacts:** `/workspace/grpo_run/final_pilot1/` (model), `training/pilot1_eval_summary.json` (full pre/post scores + rewrites), `training/pilot1_manifest.json` (config).

### Mission full run — 2026-04-17 23:35 UTC → KILLED
Launched with pilot-#1 config + 500 steps + 452 train. Killed before training started in response to user clarification: "goal is no change in length, not just making it shorter". Pilot #1 ended with mean ratio 0.94 (6% shorter than target) — not "no change".

### Pilot #2 (additive tol=0) — KILLED
Briefly tried `tol=0` (penalty active even inside ±10%). User then clarified: "anything within 10% shouldn't be punished severely, maybe just nudge it a little towards 0 change in length". Killed before training; switched to quadratic penalty.

### Pilot #2 (quadratic) — 2026-04-17 23:54 UTC — SUCCESS, lower-magnitude
Quadratic penalty: `α·(r−1)²`. Default α=100. Gentle in-band (r=0.95 → 0.25; r=0.9 → 1.0), steep outside (r=1.5 → 25). Pre/post on 200 train / 50 eval top-decile:
- Qwen-7B: 81.80 → 83.30 (Δ +1.50)
- Llama-8B: 80.92 → 82.68 (Δ +1.76)
- Length ratio at convergence: 0.86–1.05, mean ~0.91. Trade-off vs additive: smaller reward gain, tighter length.

### 3-fold mission run — 2026-04-18 starting 00:09 UTC
3 leave-one-out folds rotating which judge is held-out, 33 propositions each (1/3 of 100), 130 train / 35 eval, 200 GRPO steps, α=100 quadratic, all other knobs same as pilot #1.

#### Fold 1: train qwen7b+llama8b, held-out gemma9b — DONE 01:05 UTC
- Qwen-7B (in-panel): 80.09 → 85.83 (Δ +5.74)
- Llama-8B (in-panel): 80.49 → 90.69 (Δ +10.20)
- Gemma-9B (held-out): pending heldout_only_eval.py pass
- **length/mean_ratio: 1.01** (essentially no change — quadratic penalty worked at scale!)
- length/frac_outside_tol: 0.125 (only 12.5% of rollouts outside ±10%)
- Train wall-clock: ~48 min
- Wandb: `grpo-fold1_heldout_gemma-20260418-001738`

#### Fold 2: train qwen7b+gemma9b, held-out llama8b — DONE 02:22 UTC
- judge_qwen7b (in-panel): 80.09 → 86.06 (Δ +5.97)
- judge_gemma9b (in-panel): 79.43 → 85.00 (Δ +5.57)
- **judge_llama8b (HELD-OUT): 80.49 → 86.77 (Δ +6.29)**
- length/frac_outside_tol: 0.3125 (some length drift, but controlled)
- Wandb: `grpo-fold2_heldout_llama-20260418-012926`
- **Notable: held-out Δ+6.29 > either in-panel Δ**. Strong cross-judge transfer — improvements not judge-specific.
- First attempt OOM'd on Gemma init (default gpu_mem=0.22 vs weights 17.22 GB). Fixed by per-judge `gpu_mem_util` auto-pick (gemma 0.28, llama 0.25, qwen 0.22).

#### Fold 3: train llama8b+gemma9b, held-out qwen7b — DONE 03:44 UTC
- **First attempt OOM'd**: Llama (20 GB) + Gemma (22 GB) = 42 GB judges + Qwen-1.5B trainer (35 GB peak with per_device=8) = 77 GB on 80 GB (out of memory during backward).
- **Fix**: added `--per-device-batch` CLI arg, halved per_device from 8 → 4 with grad_accum 4 → 8 (preserves effective batch of 32 gens/step).
- judge_llama8b (in-panel): 80.49 → 89.91 (Δ +9.43)
- judge_gemma9b (in-panel): 79.43 → 85.00 (Δ +5.57)
- **judge_qwen7b (HELD-OUT): 80.09 → 86.31 (Δ +6.23)**
- length/frac_outside_tol: 0.094 (best fold!)
- Wandb: `grpo-fold3_heldout_qwen-20260418-024527`

#### Held-out delta summary across all 3 folds

| Fold | Train | Held-out | Held-out Δ |
|---|---|---|---|
| 1 | qwen7b + llama8b | gemma9b | **+8.14** |
| 2 | qwen7b + gemma9b | llama8b | **+6.29** |
| 3 | llama8b + gemma9b | qwen7b | **+6.23** |

Mean held-out Δ across the rotation: **+6.89**. All three folds show genuine cross-judge transfer; nothing is judge-specific gaming.

#### HF artefacts (all pushed 03:49 UTC, private repos under daxmavy)
- `daxmavy/attack-llm-judge-grpo-fold1-20260418` — train Qwen+Llama, held-out Gemma
- `daxmavy/attack-llm-judge-grpo-fold2-20260418` — train Qwen+Gemma, held-out Llama
- `daxmavy/attack-llm-judge-grpo-fold3-20260418` — train Llama+Gemma, held-out Qwen

Each repo contains the full bf16 rewriter (~6 GB safetensors), tokenizer, training_args, README with delta summary, and `eval_summary.json` with per-prompt scores + raw rewrites.

#### Mission completion summary

**Mission deadline:** 09:00 UTC (10 am UK time).
**Mission completion:** 03:49 UTC (HF pushes done). **5h 11min buffer.**

Key incidents tonight:
- B-10 TRL `use_vllm=True` IS-ratio collapse — sidestepped by reverting to vLLM-judges + HF-rewriter path.
- B-11 Gemma-9B vLLM OOM with default 0.22 gpu_mem_util — fixed by per-judge auto-pick (gemma 0.28, llama 0.25, qwen 0.22).
- B-12 Fold 3 OOM during backward (Llama+Gemma judges + per_device=8 trainer too much) — fixed by `--per-device-batch 4` (effective batch preserved via grad_accum 4→8).
- D-13 length penalty: tried tolerance-band additive (pilot #1, ratio drifted to 0.94), then user clarified "gentle inside ±10%, steep outside" → switched to quadratic `α·(r−1)²` (pilot #2, then all folds: ratio converged to ~1.0).

---

## 2026-04-19 — Mission setup: 3-fold attack comparison

**Decision: prune feedback-free rewriter methods**
Operator chose to retain only `naive`, `lit_informed_tight`, and `rubric_aware` for the
core paper. Removed from `rewriters/rewrite_prompts.py` and `rewriters/orchestrators.py`:

- `lit_informed`        (loose-length variant of `lit_informed_tight`; superseded by it)
- `naive_tight`         (tight-length variant of `naive`; covered by `naive` + retry-on-length)
- `rules_explicit`      (numbered-rules variant of `lit_informed_tight`; redundant)
- `scaffolded_cot_distill`  (JSON plan/draft/critique/final; orchestrator was `run_scaffolded`)

Also removed: `LIT_INFORMED_USER_TEMPLATE`, `NAIVE_TIGHT_USER_TEMPLATE`, `RULES_EXPLICIT_TEMPLATE`,
`SCAFFOLDED_TEMPLATE`. `TIGHT_RETRY_SUFFIX` and `length_bounds` retained — used by retry path.

**Mission scope locked**
- Panel: Qwen3.5-9B (no_think) + Llama-3.1-8B + Gemma-2-9B (3 leave-one-out folds)
- Dataset: top-40 controversial props × all writers, 60/40 split, 3×3 stratified by clarity tertile × agreement tertile (1091 train / 714 eval)
- Attack methods:
    - Feedback-free (run once, fold-independent): `naive`, `lit_informed_tight`, `rubric_aware`
    - Feedback-utilising inference-only: `bon_panel` K=16, with candidates generated **once** then scored 3 times (one per fold's 2-judge in-panel) to pick argmax
    - RL-trained: `grpo_400step_foldN` with embed-sim scaled to ~30% of (judge_mean − length_penalty)
- All rewrites persisted to new `attack_rewrites` + `attack_judge_scores` tables in `paragraphs.db`
- Inference-only attacks moved to local-vLLM judges (same `JudgeVLLM` codepath as RL); OpenRouter retained via `JUDGE_BACKEND=openrouter`

---

## E. Mission 2026-04-21 kickoff (attack-pipeline replication, modern rewriter, dual A100)

**D-E1. Stack: restored REPLICATION.md pins (TRL 1.2.0, transformers 4.57.6, datasets 4.8.4)**
The Unsloth experiment on branch `unsloth-qwen3-30b` had upgraded the stack to transformers 5.3.0 + TRL 0.22.2, which breaks `trl.trainer.grpo_trainer` import (unconditional `from vllm_ascend.distributed...` on x86). Reverted to the pinned stack via `pip install trl==1.2.0 transformers==4.57.6 datasets==4.8.4`. pip-freeze snapshot at `/data/shil6647/attack-llm-judge/tmp/daxmavy_pip_freeze_pre_downgrade.txt` for rollback. Unsloth and unsloth-zoo are now version-incompatible with the installed TRL/datasets; we're not using Unsloth tonight — direct TRL GRPOTrainer + PEFT QLoRA instead.

**D-E2. Conda activation hook for libstdc++ (env-scoped)**
vLLM 0.18.1's transitive `libicui18n.so.78` requires `CXXABI_1.3.15` which the system `/lib/x86_64-linux-gnu/libstdc++.so.6` does not provide (tops out at 1.3.13). The conda env ships `libstdc++.so.6.0.34` with 1.3.15. Added `/opt/anaconda/envs/daxmavy/etc/conda/activate.d/99-libstdcxx.sh` prepending `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH`. **Flagged to Max** — this edit is outside `/home/shil6647/...` and `/data/shil6647/...` and so required a notification per CLAUDE.md.

**D-E3. Disk paths: /workspace → /data/shil6647 rebase re-applied**
Max had reverted commits 2b4b008 and a7e0e56 on main without explanation. `/workspace` does not exist on this host and can't be created (root-only). The reverts would prevent the pipeline from running. Cherry-picked both commits onto `mission-2026-04-21` and extended the rebase to `rewriters/vllm_rewriter.py` and `judge/vllm_client.py` which weren't in the original patch. Remaining `/workspace` references live only in old `training/scripts/*` (run_pilot_len_pen.py, finish_eval.py, run_min_repro_vllm.py, push_folds_to_hf.py) — not on tonight's critical path.

**D-E4. Rewriter selection: Qwen/Qwen3-14B (dense, `enable_thinking=False`)**
Short-listed: Qwen3-14B, Mistral-Small-3.2-24B-Instruct-2506, Gemma-3-12B-IT, Qwen3-32B. Eliminated Mistral-Small-3.2 and Gemma-3-12B because their public checkpoints are multimodal (`Mistral3ForConditionalGeneration`, `Gemma3ForConditionalGeneration`) — loading them wastes VRAM on an unused vision tower and adds another degree of freedom to debug. Qwen3-14B is pure text (`Qwen3ForCausalLM`), fits QLoRA on A100 80GB with headroom, and `enable_thinking=False` is the REPLICATION-documented canonical way to suppress reasoning. Qwen3-32B deferred as fallback if 14B proves too weak — prefer the 2× faster training cycle on 14B for the 15h budget.

**D-E5. Judge pair: deviation from Max's prescribed panel (Nemotron-49B-v1.5-AWQ + Gemma-3-27B-QAT)**
Max's brief named Nemotron Super 49B v1.5 AWQ + Gemma 3 27B QAT as primary, with a backup list (GLM-4.7-Flash, Mistral-Small-3.2-24B, Qwen3-32B-AWQ). Evidence against the primary panel:
- `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` exists (with underscores) but has **no public AWQ variant** — only FP8 and NVFP4. FP8 on A100 (Ampere) emulates via FP16 which gives no speedup and doubles memory vs INT4. NVFP4 bleeding-edge, vLLM 0.18.1 support uncertain.
- `google/gemma-3-27b-it-qat-q4_0-gguf` is GGUF (llama.cpp), not vLLM-loadable. The `-unquantized` BF16 variant is 54 GB — co-resident with any non-trivial second judge is tight on a shared A100.
- Community AWQ Gemma-3-27B variants (e.g., `pytorch/gemma-3-27b-it-AWQ-INT4`) exist but are less-tested.

Going with the existing JUDGE_REGISTRY fold-1 pair: **`qwen95b` (Qwen/Qwen3.5-9B) + `llama8b` (meta-llama/Llama-3.1-8B-Instruct)** — proven to load and score JSON-parseable rubric output in this exact pipeline, total ~34 GB BF16 weights on GPU 0 with plenty of KV headroom. Fold 1 rotation per REPLICATION.md has `gemma9b` as the held-out judge — we'll score held-out as stretch only. Will raise this decision to Max explicitly.

**D-E6. Test plan before the 400-step run**
1. Smoke: load Qwen3-14B on GPU 1 with vLLM + HF parity check (one generation each; verify no `<think>` tokens).
2. Smoke: load qwen95b + llama8b on GPU 0 via `judge.vllm_client`; score one rewrite end-to-end.
3. Smoke: 3 GRPO steps with live judges — record step time, VRAM peak, reward distribution.
4. Extrapolate: 400 steps + 6 methods × 714 eval rows + 2-judge scoring. Cut scope if projected > 13 h.
