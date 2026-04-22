# Mission — 2026-04-22 (reboot)

The 2026-04-21 attempt (below) is **invalidated**. Three failures made the run unusable:

1. **Wrong rewriter base** — model chosen was not the one Max intended to test; the "Qwen3-14B" label was applied to weights that came from a different stub.
2. **Wrong judge panel** — `JUDGE_REGISTRY` drifted between `run_pilot_len_pen.py`, `run_mission_attacks.py`, `run_icir.py`, `judge/vllm_client.py`, and the HF-push step. The model card pushed to `daxmavy/qwen3-14b-grpo-fold1-clarity-100` advertised a panel that does not match the one the GRPOTrainer actually optimised against.
3. **Single-GPU co-location** — all 3 vLLM engines (2 judges + rewriter) were crammed onto one A100 at tight memory fractions (rewriter 0.38, judges 0.28 each), producing throughput-starved training and contributing to the length-penalty failure (`frac_outside_tol = 0.75` on the fold-1 post-eval).

The mission restarts tonight. Model IDs are TBD pending Max's selection — the workspace has been refactored so that once Max fills in `config/models.py`, every downstream script picks up the new panel from a single source of truth.

## Reboot: what changed in the repo (2026-04-22)

- **`config/models.py`** is the single source of truth for `REWRITER`, `JUDGE_REGISTRY`, and `FOLDS`. `require_config()` fails fast with a clear error if the placeholders are still set. See EXPERIMENT_NOTES.md §2026-04-22 refactor for the postmortem.
- All scripts (`training/scripts/run_pilot_len_pen.py`, `scripts/run_mission_attacks.py`, `scripts/run_icir.py`, `scripts/score_all_missing.py`, `scripts/preflight_rewriter.py`, `judge/vllm_client.py`, `scripts/run_grpo_3fold_with_rewriter.sh`) now import from `config.models` and call `require_config()` at `main()` entry.
- GRPO `--max-steps` default raised from `15` to **`400`** (CLAUDE.md GRPO step-budget rule).
- Dual-GPU topology is now mandatory — see §1 Hardware assignment and §9b for the architectural gap that still needs to be closed before a full 400-step run can launch on 2 GPUs.

## Reboot: pre-launch checklist

Before running anything heavier than a smoke test on the new mission:

- [ ] Max has populated `config/models.py` with the intended `REWRITER` + `JUDGE_REGISTRY` + `FOLDS`.
- [ ] `python3 -c "from config.models import require_config; require_config()"` exits 0.
- [ ] `nvidia-smi` shows both A100s idle (0 MiB) or has an explicit carve-out.
- [ ] `scripts/preflight_rewriter.py` passes for the new rewriter (vLLM load + 1 GRPO step).
- [ ] The out-of-process judge refactor in §9b is either complete or explicitly deferred with Max's sign-off.

---

# Mission — 2026-04-21 (ARCHIVED — invalidated, see reboot notes above)

Goal: replicate the full `attack-llm-judge` attack pipeline on a fresh, modern rewriter, on the hardware we have tonight, and land a minimum-viable cross-comparison dataset within 15 hours.

The earlier mission for 2026-04-19 is archived at `archive/MISSION_2026-04-19_overnight.md` for reference.

---

## 1. Hardware assignment
Two A100 80GB GPUs are reserved for this mission:
- **GPU 0** — judge panel (2 judges, both served via vLLM).
- **GPU 1** — rewriter (vLLM for rollouts + inference, Unsloth/QLoRA GRPOTrainer for RL).

L40S GPUs 2/3 are shared with other users; do not rely on them.

Artifacts must live under `/data/shil6647/attack-llm-judge/` (see CLAUDE.md disk rule). `/home` is full.

Python env: `daxmavy` (modify freely).

---

## 2. Rewriter requirements (hard constraints)
Any candidate rewriter must satisfy **all four**:
- **(a) No thinking mode**, or a reliable way to disable it. No `<think>` tokens, no harmony reasoning, no hidden CoT in the visible output. `enable_thinking=False` on the chat template is acceptable.
- **(b) Works in vLLM _and_ HF `generate()`**. We need inference parity because attack methods use both code paths (vLLM for bulk rollouts + BoN; HF for some evaluators). Verify both actually produce sensible outputs on a real prompt before committing.
- **(c) Fits on a single A100 80GB** for QLoRA GRPO training with batch size large enough to be useful. QLoRA is fine — lower 4-bit memory footprint is explicitly allowed if it lets us fit a bigger/more-capable rewriter or speeds up training.
- **(d) Modern but debuggable**. Released in the last ~12 months, well-supported in HF/vLLM/TRL, large user base so issues are googleable. Avoid brand-new releases (<2 weeks) that the library stack hasn't caught up to.

**Candidate short-list** (to evaluate in order):
1. Qwen3-14B (no-think variant or `enable_thinking=False`) — QLoRA comfortably fits on one A100.
2. Mistral-Small-3.2-24B-Instruct-2506 — no reasoning, solid vLLM coverage.
3. Gemma-3-12B-IT — no reasoning mode, has had vLLM friction historically, verify first.
4. Qwen3-32B (no-think) — QLoRA fits at 80 GB, more capable but slower.

Backup moves if the first candidate fails the smoke: step through the short-list sequentially. Up to **10 genuine debug attempts** per candidate before declaring it dead (per CLAUDE.md rewriter-swap rule).

---

## 3. Judge panel
**Primary (per tonight's brief):**
- Judge A: Nemotron Super 49B v1.5 AWQ — `nvidia/Llama-3.3-Nemotron-Super-49B-v1.5` (AWQ variant if available; otherwise FP8/INT4).
- Judge B: Gemma 3 27B QAT — `google/gemma-3-27b-it-qat-q4_0`.

**Backups, in priority order** (if a primary judge fails vLLM load or JSON-parseable rubric output):
1. GLM-4.7-Flash (`zai-org/GLM-4.7-9B` class or similar flash variant).
2. Mistral-Small-3.2-24B-Instruct.
3. Qwen3-32B-AWQ (no-think).

Both judges must fit on GPU 0 together. If they don't (Nemotron-49B AWQ + Gemma-27B-QAT ≈ 25 + 14 GB on top of KV cache budget), demote to two smaller backups and note the decision in EXPERIMENT_NOTES.md.

Judges are invoked via vLLM only (CLAUDE.md judge-inference rule). Hard ban on HF `generate()` for judges.

---

## 4. Attack methods (6 total)
Replicate exactly the methods documented in `REPLICATION.md` §attacks:
1. `naive` — baseline rewrite, no attack knowledge.
2. `lit_informed_tight` — literature-informed rewrite prompt.
3. `rubric_aware` — rewriter sees the rubric text.
4. `bon_panel` — best-of-N=16 ranked by the in-panel judges.
5. `icir` — 4-iteration iterative criticism-informed rewrite.
6. `grpo_400step` — 400-step GRPO-trained rewriter (QLoRA).

Same chat-template handling, reward function, and DB schema as the existing pipeline. Tag every row in `attack_rewrites` with the new rewriter's short name in the `rewriter_model` field so cross-rewriter comparison queries work unchanged.

---

## 5. Minimum deliverable (must-hit)
**1 fold × 2 in-panel judges × 1 criterion (clarity).**

That is: run the 6 attack methods on fold 1 only, with the primary 2 judges as the in-panel pair, for clarity only. Score all resulting rewrites with the same 2 judges and push rows to `attack_rewrites` / `attack_judge_scores`.

Held-out judge scoring and informativeness can be added if the 15 h budget has slack. They are stretch, not minimum.

---

## 6. Time budget
**15 hours end-to-end**, covering:
- rewriter candidate selection + smoke (vLLM + one GRPO step)
- timing probe
- 6 attack methods × 1 fold × clarity (including GRPO 400-step training)
- scoring pass with 2 judges
- DB ingestion and a sanity-check query

If the timing probe projects the GRPO 400-step stage alone would blow the budget, **shrink scope before launching** (fewer steps, smaller rewriter, or skip ICIR's iter-4 in favor of iter-2). Surface the decision to Max via the report, don't just silently cut.

---

## 7. Timing probe (mandatory before the full run)
After the candidate rewriter smoke passes:
- Run **3 GRPO steps** end-to-end with judges live and count wall time per step + VRAM peak.
- Run **5 vLLM inference generations** on real prompts, count tok/s and per-request latency.

Extrapolate:
- 400 GRPO steps × step_time → training budget.
- 714 eval-set rewrites × per-method × per-criterion → inference budget.
- Sum with judge-scoring time (each rewrite scored by 2 judges).

If the projection exceeds ~13 h (leave 2 h of safety margin on the 15 h ceiling), cut scope before launch.

---

## 8. Rewriter debug-loop discipline
Per CLAUDE.md rewriter-swap rule: up to ~10 genuine debug attempts per candidate before dropping to the next short-list entry. Log each attempt (symptom / hypothesis / fix / result) in EXPERIMENT_NOTES.md so retries don't repeat work.

---

## 9. Stretch goals (if minimum lands with wall-time remaining)
Per Max's 2026-04-22 note: "if for some reason you complete what I've asked, continue onto the 2nd and 3rd folds". Order:
1. Clarity fold 2 (swap in-panel judges per the rotation in `REPLICATION.md`).
2. Clarity fold 3.
3. Informativeness fold 1 (only after all three clarity folds are done).
4. Held-out judge evaluation pass (one judge per fold).
5. Agreement-score regressor re-application.

Push each fold's GRPO checkpoint to HF and verify round-trip **before** starting the next fold so a crash doesn't cost the earlier work.

## 9b. Out of scope for tonight
- 2nd rewriter replication (explicitly future work).

## 9c. GPU-sharing caveat
Per Max's 2026-04-22 note: "the A100s are shared - others might jump on them, look out for this and tell me if so". Every monitoring check-in must include `nvidia-smi`; if a foreign process lands on GPU 0 or GPU 1 mid-run, raise it to Max at the next user turn with PID/user/free-VRAM/run-stability context. Don't kill other users' jobs.

---

## 10. End-state checklist
When the mission is complete, the following should be true:
- [ ] `attack_rewrites` has a new `rewriter_model` cohort with 6 methods × 714 eval rows × 1 fold.
- [ ] `attack_judge_scores` has 2-judge clarity scores for every new rewrite.
- [ ] The 400-step GRPO checkpoint is pushed to `daxmavy/<rewriter>-grpo-fold1-clarity` on HF, round-trip verified.
- [ ] EXPERIMENT_NOTES.md has a section for 2026-04-21 covering rewriter choice, smoke result, timing probe numbers, and any debug attempts.
- [ ] No GPU leaks; `nvidia-smi` shows 0 MiB used when done.
- [ ] No /data overflow (headroom >= 50 GB at end of run).
