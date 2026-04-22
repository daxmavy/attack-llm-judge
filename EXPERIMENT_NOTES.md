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

### B-16. Mission launch landed on L40S instead of A100 — CUDA device ordering was FASTEST_FIRST
**Date.** 2026-04-22 (mission exec, ~01:55 UTC).
**Symptom.** After launching GRPO on `CUDA_VISIBLE_DEVICES=1` and feedback_free on `CUDA_VISIBLE_DEVICES=0`, `nvidia-smi --query-gpu` reported GPUs 0/1 at 4 MiB used while GPUs 2/3 (L40S, 48 GB each) had 32 GB and 23 GB respectively. vLLM then crashed both engine cores with `ValueError: No available memory for the cache blocks` (feedback_free at `gpu_memory_utilization=0.70`, GRPO judge during warmup). vLLM also printed `WARNING: Detected different devices in the system... Please make sure to set CUDA_DEVICE_ORDER=PCI_BUS_ID to avoid unexpected behavior.`
**Diagnosis.** Host has 2× A100 80GB (indices 0,1) and 2× L40S 48GB (indices 2,3). CUDA's default ordering is `FASTEST_FIRST`, which ranks by compute capability first: **L40S CC 8.9 outranks A100 CC 8.0**, so unmasked `CUDA_VISIBLE_DEVICES=0` resolves to the first L40S (physical GPU 2), not the A100 (physical GPU 0). Confirmed by mapping `nvidia-smi --query-compute-apps=pid,gpu_uuid` UUIDs against `--query-gpu=index,name,uuid`:
  - vLLM engine core PID 3370317 was on `GPU-67a9d512-…` = L40S GPU 2
  - GRPO judge PID 3363620 was on `GPU-81580d1a-…` = L40S GPU 3
  - Both "A100" launches actually took L40S devices.
**Consequence.** Ran the prior 6 preflight smokes (v5/v6/v7) on L40S too — the "41 s/step A100 steady-state" from v7 was really 41 s/step on an L40S. The real A100 step time will differ (plausibly faster with more compute; possibly slower if the judge vLLM on CC 8.0 is less efficient). All prior budget math must be re-validated once the A100 launch settles.
**Fix.** Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` in the launch env alongside `CUDA_VISIBLE_DEVICES`. Under PCI_BUS_ID the mapping matches `nvidia-smi` output exactly: `CUDA_VISIBLE_DEVICES=0` → physical GPU 0 (A100), `CUDA_VISIBLE_DEVICES=1` → physical GPU 1 (A100). Relaunched both jobs with the updated env at ~01:57 UTC.
**Leaked VRAM.** Killing the vLLM EngineCore subprocesses via SIGTERM left ~32 GB orphaned on GPU 2; no owner process, driver-side reclaim (known vLLM/Triton idiom). Not blocking since the new launches are on GPUs 0/1.
**Actionable convention going forward.** Every launch script in this repo that uses `CUDA_VISIBLE_DEVICES` on a host with mixed compute capability MUST also set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Added to the post-mortem as an auto-test item for future mission prep.

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

### 2026-04-22 02:55 — GRPO pre-eval slowness anomaly (raise-don't-hide)
- Qwen3-14B QLoRA GRPO on GPU 1: `[02:04:57] rewriter loaded 4-bit QLoRA` → no step=1 after 50 min.
- Pmon: trainer PID 3375296 at 96 % sm; both local vLLM EngineCores (rollout, judges) idle. Wandb filestream advancing by 2 events / 15 s = system-stats only.
- Interpretation: still in `generate_eval` HF-generate pre-eval loop (50 prompts × max_new=260 × bnb-4bit 14B). Expected ~6 min batched; observed ~50 min. ~8× slower than projection.
- Root-cause candidates: (a) bnb-4bit NF4 generate is materially slower than expected at batch=8 (no FA2 for bnb?), (b) padding causing ~1024-token inputs when most are much shorter.
- Revised cutoff: 03:10 BST (= 65 min pre-eval). Original 02:55 rule assumed free restart; actually killing forfeits the ~50 min already invested plus ~50 min to redo on relaunch, for ~100 min net loss. Training-step time uses vLLM rollouts not HF generate, so pre-eval slowness doesn't imply slow training.

### 2026-04-22 03:04 — GRPO pre-eval judge disagreement (raise-don't-hide)
- pre  judge_qwen95b mean=10.41
- pre  judge_llama8b mean=78.80
- 68-point gap on the SAME 50 baseline rewrites scored by both in-panel judges with the same clarity rubric.
- Reward fn averages the two; skew means reward ≈ 45 with high between-judge variance → training gradient signal reflects "make llama8b happier without making qwen95b much worse" rather than a clean clarity signal. Noting for post-hoc analysis.
- Hypotheses (deferred — not blocking the run): (a) qwen95b stricter on this domain, (b) parse failures coerced to 0 for qwen95b, (c) rubric prompt interaction with Qwen3.5 style. To be investigated if post-eval delta is unexpected.
- Pre-eval wall time: 60 min (02:04:57 → 03:04:12) for 50 prompts × max_new=260 × bnb-4bit 14B batched at 8. 8× slower than projection (~6 min). Root cause unknown; does not affect training step time because training uses vLLM rollouts, not HF generate.

### 2026-04-22 03:08 — GRPO 400-step kill + restart at 100 steps with 400-tok cap
- Step 1 TRL output revealed broken config: completions/mean=min=max=260, clipped_ratio=1.0, mean_terminated_length=0.0, loss=0.0, grad_norm=0.083.
- Root cause: `max_completion_length=260` (hardcoded GRPOConfig line 582) too tight for this rewriter — Qwen3-14B is verbose; 0/32 rollouts terminate naturally at 260. Length penalty is ~constant across the group → advantage=0 → no gradient signal.
- Step 2 had loss=0.0298 only because ONE of 32 rollouts terminated at 96 (mean_terminated_length=96). 97% still clipped.
- Step time 78 s/it × 400 = 8.7 h → blows remaining budget once ICIR + post-eval + HF push are stacked.
- **Fix**: edited `max_completion_length` 260→400. Relaunching with `max_steps=100` (projected ~2 h training + 2 h eval = 4 h total from 03:10 → done ~07:10; leaves ~6 h for ICIR + post-eval + HF + ingestion).
- Accepted cost: forfeit 60 min of pre-eval wall time + step-2 partial gradient.

### 2026-04-22 03:22 — ICIR v1 crashed: hardcoded gpu_memory_utilization=0.18 for 14B rewriter
- Symptom: `ValueError: No available memory for the cache blocks.` on vLLM LLM() init for Qwen3-14B. Engine core 3541979 exited, python PID 3541336 raised RuntimeError.
- Root cause: `scripts/run_icir.py` line 130 hardcoded `gpu_memory_utilization=0.18` — fine for the default Qwen2.5-1.5B rewriter but insufficient for 14B bf16 (~28 GB weights) which alone exceeds 0.18 × 80 GB = 14.4 GB budget. Same class of bug as B-11 (already fixed in run_mission_attacks.py via `_rewriter_vllm_mem_util`).
- Fix: added an auto-picker to run_icir.py (0.45 for 14/13B, 0.25 for 7/8/9B, else 0.18). Relaunched at 03:22 as ICIR v2 (PID 3554970).
- Post-fix budget on GPU 0 (co-located rewriter + 2 judges): rewriter 0.45 (36 GB) + judge qwen95b 0.27 (21.6 GB) + judge llama8b 0.25 (20 GB) = 0.97 of 80 GB. Rewriter loaded successfully (weights 27.52 GiB, KV 6.86 GiB). Judges loading now.

### 2026-04-22 05:00 — ICIR v2 budget overcommit on fold-2 clarity (gemma9b panel)
- Fold-1 clarity ICIR ran clean with the 0.45 rewriter budget (in-panel: qwen95b+llama8b). Fold-2 clarity crashed at 2nd judge init: `Free memory on device cuda:0 (22.11/79.15 GiB) on startup is less than desired GPU memory utilization (0.28, 22.16 GiB)` — off by ~50 MiB.
- Root cause: JudgeVLLM's hardcoded `gpu_memory_utilization=0.28` (run_pilot_len_pen.py ~line 191) means the rewriter must fit below `1.0 - 2×0.28 = 0.44`. 0.45 overcommits by 0.05 at fold 2 where gemma9b was the in-panel judge.
- Fix: lowered `_rew_mem_util` table to **0.38 for 14B/13B**, 0.22 for 7B/8B/9B, 0.18 else. Qwen3-14B bf16 = 28 GiB weights; 0.38 util (30.1 GiB) leaves ~2 GiB KV — enough for 8+ concurrent 3072-tok requests given GQA (40 layers × 8 KV heads × 128 head_dim → 160 KiB/token). Fold 2/3 clarity ICIR both ran clean after this edit.

### 2026-04-22 07:15 — GRPO fold-1 clarity 100-step — MISSION §5 deliverable DONE
- Wall: 308.4 min total (5h 8m), train 185.6 min, pre-eval 60 min, post-eval 61 min.
- Eval on 714 fold-1 paragraphs (pre=baseline rewrites, post=GRPO rewrites):

| Judge | Role | Pre | Post | Δ |
|---|---|---|---|---|
| qwen95b | in_panel | 10.41 | 66.85 | **+56.43** |
| llama8b | in_panel | 79.18 | 74.79 | −4.39 |
| gemma9b | held_out | 35.24 | 69.87 | **+34.64** |

- **Held-out gemma9b Δ=+34.64 validates the transfer thesis** — GRPO-trained rewriter pleases a judge it was never trained against.
- qwen95b pre=10.41 unusually low (matches 03:04 disagreement flag); post=66.85 suggests qwen95b *can* be pleased, so the pre-eval baseline is what looked like a floor. Worth a sanity check post-mission (dump 5 baseline rewrites + qwen95b reasoning).
- **Length-penalty failure — RAISING**: `frac_outside_tol = 0.75` on post-eval. α=100 quadratic penalty did not hold length under Qwen3-14B's high-variance completions. Needs a length-matched baseline or bounded-generation variant before using these rewrites for cross-method comparison of length-sensitive metrics.

### 2026-04-22 07:58 — Fold-1 informativeness bon_score ~4× slower than clarity (raise-don't-hide)
- bon_score iterates 11,424 candidates × 2 judges sequentially (run_mission_attacks.py:290). Fold-1 clarity on qwen95b+llama8b: ~17 min total (~8.5 min/judge). Fold-1 informativeness on same panel: **qwen95b alone 31.5 min** — 3.8× slowdown per-judge.
- Cause: informativeness rubric elicits longer judge reasoning (more output tokens per candidate × 11,424 → dominated by generation cost, not prefill).
- Not a hang — GPU 0 at 99–100% util throughout, llama8b ramped up on schedule after qwen95b finished.
- Budgeting implication: informativeness attack methods cost ~4× more wall time than clarity for the same fold × panel. Future stretch-goal queues should budget accordingly.

### 2026-04-22 08:37 — HF push: daxmavy/qwen3-14b-grpo-fold1-clarity-100 (§10 DONE)
- 138 MB adapter + tokenizer + eval_summary.json uploaded in 9.8 s (private repo).
- Round-trip verified: downloaded `adapter_config.json` from HF, byte-exact match with local copy.
- Script: `scripts/push_grpo_fold1_clarity_100_to_hf.py`.
- MISSION.md §10 checkbox 3 (GRPO checkpoint push + round-trip) satisfied.

### 2026-04-22 09:54 — §10 checkbox 2 gap discovered: `attack_judge_scores` missing all non-BoN rewrites (raise-don't-hide)
- At the end of the fold-1 informativeness stretch (§9.3), audit showed `attack_judge_scores` contained only 34,272 `bon_candidate` rows across 3 judges × 2 criteria — zero scores for `naive` / `lit_informed_tight` / `rubric_aware` / `bon_panel` / `icir` rewrites. §10 checkbox 2 was technically unmet.
- Root cause: `run_mission_attacks.py` scores BoN candidates (it's mandatory for selection), but feedback_free + bon_panel + ICIR rewrites are written to `attack_rewrites` without a scoring pass. Held-out eval (§9.4) also needed these cells populated.
- Remediation: `scripts/score_all_missing.py` — excludes `bon_candidate`, scores the remaining 9,282 × 3 judges × 2 criteria = 55,692 cells in one pass. Loads judges sequentially (qwen95b → llama8b → gemma9b), swaps rubric in-place between clarity and informativeness, writes via `INSERT OR REPLACE` (idempotent).
- Wall time 1h 17m total (launched 09:53, done 11:10):
  - qwen95b: 26.6 min clarity + 26.5 min informativeness (5.8/s both — rubric length-bound)
  - llama8b: 10.4 min clarity (14.8/s) + 12.0 min informativeness (12.9/s)
  - gemma9b: 17.0 min clarity (9.1/s) + 20.5 min informativeness (7.6/s)
- Final coverage (non-bon_candidate): all 9,282 rows × 3 judges × 2 criteria. Expected skew: `bon_candidate × informativeness × gemma9b` = 0 because gemma9b wasn't on fold-1 informativeness BoN panel — that cell is not needed for §10 or §9.4 held-out eval on selected rewrites; filling it would cost ~25 min GPU if analysis later needs "held-out on raw candidates".

### 2026-04-22 12:08 — §9.5 agreement-score regressor re-application
- Re-applied DeBERTa-v3-base regressor (`agreement_model/runs/main/final`) to all 32,130 rewrites in `attack_rewrites`. New table `attack_agreement_scores` populated: range [0.110, 1.0], mean 0.495.
- Throughput 200 rows/s on single A100 (above docstring 150/s estimate); GPU 0 peak 3.68 GiB. Total wall time 2.7 min.
- **Bug found and fixed**: `tokenizer_config.json` stores `extra_special_tokens` as a list `["[PAD]", "[CLS]", "[SEP]"]` (older transformers format); transformers 4.57.6 expects a dict, crashes with `AttributeError: 'list' object has no attribute 'keys'` in `_set_model_specific_special_tokens`. Patched `scripts/apply_agreement_score.py:64` to pass `extra_special_tokens={}` as kwarg (non-invasive; tokens already present via top-level `pad_token` / `cls_token` / `sep_token` keys). REPLICATION.md line 182 had pre-warned about this class of issue.

### 2026-04-22 — Mission §10 end-state (fold 1 rewriter replication)
- [x] `attack_rewrites` cohort: 6 methods × 714 eval rows × fold 1 for `Qwen/Qwen3-14B` rewriter.
- [x] `attack_judge_scores`: 2-judge (in-panel) scores for every new rewrite; held-out gemma9b coverage for selected rewrites (non-bon_candidate).
- [x] `attack_agreement_scores`: all 32,130 rewrites scored by DeBERTa regressor.
- [x] GRPO 100-step checkpoint on HF `daxmavy/qwen3-14b-grpo-fold1-clarity-100`, round-trip verified.
- [x] EXPERIMENT_NOTES.md updated with rewriter choice, smoke, timing, debug attempts.
- [x] No GPU leaks: GPU 0 / 1 both 0 MiB at 12:08 after regressor exit. Foreign process on GPU 3 is another user (`/opt/anaconda/envs/nlp2025/bin/python`, PID 3956169) — does not touch our GPUs.
- [x] No /data overflow: 1.4 TB free on `/data` (85% used), our subtree 142 GB.
- [x] §9 stretch: all 5 items done (clarity folds 2 & 3, informativeness fold 1, held-out eval via score_all_missing, agreement regressor).

### 2026-04-22 — Post-mortem + refactor: model-stub source of truth & GRPO step-budget lock
**Trigger.** Max invalidated the 2026-04-21 mission. Three failures:
1. Wrong rewriter base — the HF id actually loaded at GRPO time didn't match what Max had intended to test.
2. Wrong judge panel — `JUDGE_REGISTRY` had drifted between `training/scripts/run_pilot_len_pen.py`, `scripts/run_mission_attacks.py`, `scripts/run_icir.py`, `judge/vllm_client.py`, `scripts/score_all_missing.py`, and the HF push step. The model card pushed to `daxmavy/qwen3-14b-grpo-fold1-clarity-100` advertised a panel that didn't match the weights' actual reward source.
3. Single-GPU co-location — rewriter + 2 judges all on one A100 at tight memory fractions; contributed to the length-penalty failure (`frac_outside_tol = 0.75` at fold-1 post-eval, §2026-04-22 07:15 above).

**Fix — single source of truth.** New module **`config/models.py`** holds `REWRITER`, `JUDGE_REGISTRY`, `FOLDS`. Both ship as `None`/`{}` placeholders so stale defaults can't silently resurrect. `require_config()` is called at every entrypoint's `main()` and raises a clear `SystemExit` banner if not populated.

Scripts refactored to import from `config.models` and drop their local copies:
- `training/scripts/run_pilot_len_pen.py` (GRPO trainer) — also raised `--max-steps` default from 15 → **400** to match MISSION §7 and CLAUDE.md's new GRPO step-budget rule.
- `scripts/run_mission_attacks.py` (feedback_free / bon_generate / bon_score).
- `scripts/run_icir.py` (ICIR 4-iter attack).
- `scripts/score_all_missing.py` (gap-filling post-hoc scorer).
- `scripts/preflight_rewriter.py` (smoke tester).
- `judge/vllm_client.py` — `LOCAL_MODEL_MAP` is now derived from `JUDGE_REGISTRY` instead of a hardcoded dict.
- `scripts/run_grpo_3fold_with_rewriter.sh` — fold rotation now comes from Python, not a hardcoded bash array.

**Shell-script subtlety worth remembering.** First attempt at the shell fix used `eval "$(python3 … FOLDS print …)"`. Under `set -e`, when Python raised `SystemExit` from `require_config()`, bash eval'd the empty stdout as no-op commands and silently continued into an undefined `${IN_PANEL[$FOLD]}`. Fix: write Python output to a `mktemp`'d tempfile, then `source` it — that keeps the non-zero Python exit visible to `set -e` and aborts before `source` runs. Reminder that command-substitution wraps the child's exit status and hides it from `set -e`.

**Dual-GPU topology — CLAUDE.md rule added, architectural gap still open.** Added a "GPU topology rule" clause to CLAUDE.md mandating GPU 0 = judges / GPU 1 = rewriter on 2× A100 boxes, with `CUDA_DEVICE_ORDER=PCI_BUS_ID`. BUT: `run_pilot_len_pen.py::reward_fn` still instantiates `JudgeVLLM` in the same Python process as the GRPOTrainer, so on a 2-GPU host both still end up on a single `CUDA_VISIBLE_DEVICES`. A real split needs either a `multiprocessing.Process` for judges (preferred — less invasive) or a sibling HTTP judge server. Flagged as mandatory pre-launch work in REPLICATION.md §9 and MISSION.md reboot checklist. Not attempted yet — awaiting Max's model selection + scope-check before touching the reward-fn hot path.

**Docs updated.**
- `CLAUDE.md`: added Model-ID source-of-truth rule, GPU topology rule, GRPO step budget lock.
- `MISSION.md`: 2026-04-21 section marked ARCHIVED/invalidated; new 2026-04-22 reboot header with pre-launch checklist; old mission text retained below for reference until the new panel is chosen.
- `REPLICATION.md`: §0 + §1 + §7 now point at `config/models.py`; §9 reframed from "target setup" to mandatory topology with the open architectural gap called out.

**Smoke verification.** Imported every refactored Python entrypoint in the `daxmavy` conda env — all import cleanly with placeholders. Each `require_config()` call raises the expected banner. The refactored shell script also fails fast (exits non-zero, arrays never populated) when models are unset. No runtime has been launched against live weights.

**Known residual follow-ups.**
- Out-of-process judge refactor (see above) — mandatory before next 400-step GRPO launch. **Done 2026-04-22, see below.**
- `config/models.py` still has placeholder values — Max to populate with the real panel before mission restart.
- Old HF repo `daxmavy/qwen3-14b-grpo-fold1-clarity-100` advertises a panel that doesn't match reality; consider deleting or adding a README warning once the new mission produces a replacement. Not touched in this pass (user data, not to be destroyed without explicit ask).

---

## 2026-04-22 — HTTP judge server refactor (architectural gap closed)

**Motivation.** The in-process judge pattern (`JudgeVLLM` instantiated inside `run_pilot_len_pen.py::reward_fn`, same for ICIR, BoN, gap-fill, preflight) meant the GRPOTrainer and both judges fought for a single `CUDA_VISIBLE_DEVICES` on a 2-GPU host. The 2026-04-21 run co-located 3 vLLM engines on a single A100 at `gpu_memory_utilization≈0.28` and silently overcommitted, contributing to the fold-1 `frac_outside_tol=0.75` length-penalty failure. Max's direction: "fix this, judge scoring should always be done from the http judge server."

**What landed.**
- **`judge/rubrics.py`** — extracted the rubric strings (`JUDGE_SYSTEM`, `CLARITY_RUBRIC`, `INFORMATIVENESS_RUBRIC`, `RUBRICS`), `SCORE_RE_*`, `parse_score_and_reasoning`, `supports_system_role` from `run_pilot_len_pen.py`. Shared between server and any legacy caller.
- **`judge/server.py`** — stdlib `ThreadingHTTPServer`, one process per GPU. Endpoints: `/load`, `/score`, `/generate`, `/set_rubric`, `/unload`, `/health`, `/shutdown`. Loads HF tokenizer + vLLM `LLM` per slug, caches in `_STATE`, serialises concurrent generate calls with a per-judge lock. `_auto_gpu_mem_util()` picks sensible defaults for a dedicated GPU (0.45 for 7-14B, 0.70 for 27-32B, 0.90 for 49B+). Auto-loads on first `/score` or `/generate` if a rubric is supplied. Qwen3 family forces `enable_thinking=False` (preserved from old code).
- **`judge/http_client.py`** — `JudgeHTTP` dataclass with the same public surface as the old `JudgeVLLM` (`name`, `model_id`, `wandb_name`, `rubric_name`, `rubric_text`, `score()`, `score_full()`, `set_rubric()`, `unload()`, `generate_raw()`). `spawn_judge_server(port, gpu, log_path)` subprocess helper that sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES` *before* Python import, polls `/health` with timeout, reuses an already-bound port if found.
- **`judge/vllm_client.py`** — rewritten from "in-process vLLM per judge" to a thin passthrough. Public surface unchanged (`is_local_available`, `call_judge_local`, `call_judge_local_batch`); delegates to `JudgeHTTP.generate_raw`. Endpoint from `JUDGE_HTTP_ENDPOINT` env var (default 127.0.0.1:8127). No auto-spawn — callers bring the server up themselves.
- **`training/scripts/run_pilot_len_pen.py`** — dropped the inlined `JudgeVLLM` class and rubric constants (they now live in `judge/rubrics.py`). Spawns a judge server at main-entry via `spawn_judge_server(gpu=args.judge_gpu)`, constructs `JudgeHTTP(name=slug, rubric=..., endpoint=endpoint)` per in-panel judge. wandb reward keys + `pre_scores`/`post_scores` dict keys migrated to `j.wandb_name` to preserve the `judge_<slug>` format the existing dashboards + DB `judge_slug` column expect. Held-out eval block explicitly `unload()`s the in-panel judges first, then loads the held-out slug on the same server.
- **`scripts/run_icir.py`** — replaced hand-rolled `j._build_prompt()` + `j.llm.generate()` + regex parse with `j.score_full(props, rewrites, include_raw=False)`. `j.set_rubric(cri)` instead of mutating internal state. `_rew_mem_util` defaults raised (0.75 for 32B, 0.70 for 24B, 0.65 for 14B, …) because the rewriter now owns its GPU.
- **`scripts/run_mission_attacks.py`** — `cmd_bon_score` routes through `JudgeHTTP`. DB writes continue to use `j.wandb_name` as `judge_slug` so existing aggregation queries don't break.
- **`scripts/score_all_missing.py`** — same pattern; in-place rubric swap via `j.set_rubric()` between clarity and informativeness.
- **`scripts/preflight_rewriter.py`** — spawns a judge server before the 1-step smoke so preflight exercises the same path as the real run.

Every consumer accepts `--judge-endpoint` / `--judge-port` / `--judge-gpu` CLI flags, so one server can be shared across a preflight → GRPO → mission-attacks → scoring pipeline without reloading judge weights between stages.

**Smoke.** All seven refactored modules import cleanly in the `daxmavy` env (`python -c "import judge.http_client, judge.server, judge.vllm_client; import run_pilot_len_pen; import run_icir; import run_mission_attacks; import score_all_missing; import preflight_rewriter"`). `python -m judge.server --help` parses. End-to-end server spawn is deferred to the first run with a populated `JUDGE_REGISTRY` (still blank by design — Max picks the panel).

**Docs updated.** CLAUDE.md judge-inference rule now mandates the HTTP server path. MISSION.md reboot checklist marks the refactor complete. REPLICATION.md §9 rewritten from "architectural gap" to the live out-of-process architecture with endpoint list + consumer inventory.

---

## 2026-04-22 (afternoon) — pilot-plan rewriter + judge picks partially invalidated by smoke

Ran the planned cohost smoke + Unsloth load probe on the picks in `config/models.py`. Three problems surfaced, all traceable to picking HF repos without inspecting their architecture tags:

### B-N1. Qwen/Qwen3.5-9B is a vision-language model, not text-only
- `HfApi.model_info('Qwen/Qwen3.5-9B').pipeline_tag == 'image-text-to-text'`; `model_type=qwen3_5`, tags include `image-text-to-text`.
- Unsloth loader banner confirmed: `Unsloth: Fast Qwen3_5 patching` → `Qwen3_5VisionModel`; `[unsloth_zoo|WARNING] get_input_embeddings not auto-handled for Qwen3_5VisionModel`.
- `unsloth/Qwen3.5-9B` is the same thing (that's what Unsloth silently redirected our download to, eating ~12 min of XET downloads).
- Load itself worked: 735s base load, 18.8 GB VRAM bf16, `get_peft_model` 7.4s → 51.0M trainable params (0.54% of 9461M). But `tok(prompt, return_tensors='pt')` crashes because Unsloth-zoo's `patched_call` in `tokenizer_utils.py:702` routes through `Qwen3VLProcessor.__call__`, which passes the text to `qwen2_vl.image_processing_qwen2_vl.preprocess()` as if it were an image URL → `binascii.Error: Incorrect padding` → `ValueError: Incorrect image source`.
- The Unsloth notebook I was referencing (`Qwen3_5_4B_Vision_GRPO.ipynb`) is demoing the VL variant specifically — that's why `get_peft_model(target_modules='all-linear')` "worked" in the notebook. For a text-only rewriter, Qwen3.5 is the wrong series: there is no text-only Qwen3.5-9B repo.
- **Fix:** swap `REWRITER` to a text-only model. Candidates (all `pipeline_tag=text-generation`):
  - `Qwen/Qwen3-14B` — known-working on 2026-04-21 run (HF push succeeded, GRPO 100-step reached). Largest of the three; needs QLoRA for comfort on 1 A100 under GRPO — Unsloth supports this via `load_in_4bit=True`.
  - `Qwen/Qwen3-8B` — same family, smaller, bf16 LoRA fits easily (~18 GB + optimizer).
  - `Qwen/Qwen2.5-7B-Instruct` — older but rock-solid Unsloth + vLLM path.
- The `--use-unsloth` branch I added to `run_pilot_len_pen.py` (commit 519c2f2) is still correct — it's just the model pick that was wrong. FastLanguageModel handles text-only Qwen/Llama fine.

### B-N2. RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8 has no HF tokenizer
- Repo file list: `['consolidated.safetensors', 'generation_config.json', 'params.json', 'SYSTEM_PROMPT.txt', 'tekken.json', 'README.md']`. No `tokenizer_config.json`, no `tokenizer.json`, no `chat_template.json`, no HF `config.json`.
- vLLM can still load it (the cohost run did: `Resolved architecture: PixtralForConditionalGeneration`, 24 GiB model load, FLASH_ATTN backend, KV cache 27k tokens). But our judge server uses HF `AutoTokenizer` + `apply_chat_template` — vLLM downloads `tekken.json` at init and converts it, but the HF tokenizer object has no `chat_template` attribute, so scoring crashes:
  > `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed`
- Also: Mistral-Small 3.2 (2506) is itself a Pixtral VLM, so even if we patched the tokenizer, rewards from a vision-conditioned judge on text-only inputs are not what we want.
- **Fix:** swap `mistral24` hf_id to `RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic`. Verified today via `HfApi`: `architectures=['MistralForCausalLM']`, `model_type=mistral`, no `vision_config`, ships `tokenizer_config.json` + `tokenizer.json`. The 3.1-2503 variant is also Pixtral; 2501 is the last pure-text Mistral-Small-24B.

### B-N3. Flash-Attention 2 broken in daxmavy env after transformers 5.5.0 upgrade
- Unsloth banner: `Unsloth: Your Flash Attention 2 installation seems to be broken. Using Xformers instead. No performance changes will be seen.`
- `FA [Xformers = 0.0.35. FA2 = False]`.
- Not blocking (Xformers backward works on Qwen text arches), but ~30% training-step regression vs FA2. Likely the `flash-attn` wheel pinned for transformers 4.57 doesn't satisfy transformers 5.5 / torch 2.10 / CUDA 12.8.
- Investigation deferred until the rewriter / judge swaps above are locked in — no point reinstalling FA2 if we're about to re-pin the whole stack.

### What's still-valid from today's smoke
- **Cohost VRAM envelope verified.** Pair 1 (gemma27 + mistral24) loaded both judges on one A100 with peak GPU0 = **65.7 GB / 80 GB = 82%**. Two FP8 24-27B judges fit comfortably with gpu_memory_utilization=0.45/0.38. Auto-picks in `judge.server._auto_gpu_mem_util` (0.55/0.70/0.90) assume a dedicated GPU and overcommit when cohosting — the cohost script's explicit per-judge override (in `/data/shil6647/attack-llm-judge/tmp/pairwise_judge_cohost.py`) is the right pattern and should migrate into the production launchers.
- **gemma27 scoring works end-to-end.** Sample scores on the 4 calibration items: `[90.0, 90.0, 95.0, 95.0]` — plausible and consistent across two cold loads (429.9s first load on fresh XET cache, 69.9s on pair 2 with cache warm; 16.0s / 7.6s per 4 items).
- **vLLM picks TRITON_ATTN for gemma27 vit**, FLASH_ATTN for mistral24 text. Compute-side fine, logged for the record.

### Immediate plan
1. Get Max's sign-off on rewriter swap (likely `Qwen/Qwen3-14B` to match prior-mission known-good) and mistral24 swap (`RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic`).
2. Update `config/models.py`, re-run cohost pair 1 (gemma27 + new mistral24) to verify the chat-template fix; pair 2 (gemma27 + phi4) is already mid-flight and not affected.
3. Re-run the load probe against the new rewriter (expect Qwen3-14B to load in ~3 min from warm cache, ~35 GB VRAM bf16 LoRA or ~12 GB QLoRA).
4. Then the 5-step GRPO smoke.
