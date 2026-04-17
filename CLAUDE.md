# Project instructions

- Read `STRUCTURE.md` before making any changes. It is the source of truth for what to work on next.
- Make small, focused git commits for the changes you make. Prefer many small commits over one large one.
- Push all commits to Github remote, so that if this server is lost for some reason I can recover the code.
- Add experiment notes (design decisions, bugfixes, etc) to EXPERIMENT_NOTES.md
- Literature review is at LITERATURE.md
- default to autonomy — act first, ask only when a decision is genuinely blocking, irreversible, or ambiguous in a way that cheap exploration can't resolve. Don't ask just because you can.

## Tonight's mission (2026-04-17 → 2026-04-18, hard deadline 09:00 UTC / 10 am UK)

Three goals, in order. Don't stop to ask questions until all three are done or the user asks one directly.

**a) Get vLLM working** — replicate (closely enough) the reward-climbing behaviour we saw in the non-vLLM minimum runs. If the TRL `use_vllm=True` rewriter path can't be stabilised, fall back to the proven **vLLM judges + HF rewriter** setup (`run_min_vllm.py` pattern). Replicate the reward climb *before* adding a length penalty.

**b) Get pilot working** — with vLLM-judges path running, add a length penalty to the reward and run short pilots until reward climbs *without* corresponding length drift (a small amount of length increase is acceptable; a ~40% drift like the original HF run is not). Work incrementally: one knob per pilot, check at 5 iterations rather than waiting for 25. Options to consider from prior discussion: Dr-GRPO loss/scale_rewards, higher generation temperature, tightened IS cap, different dataset (e.g. uniformly-sampled human writers instead of top-decile) as a last resort.

**c) Run full training before 09:00 UTC** — once (b) produces a config that climbs without length drift, launch the full run with those hyperparameters frozen. Scope (prompt count, epochs, % of corpus) is negotiable — cut whatever's needed to finish inside the time budget. Push the trained checkpoint to HF and log full pre/post eval on all 3 judges.

**Operating rules:** speed matters; inspect after 5 steps, not 25; make git commits after each milestone; manage VRAM aggressively (OOM kills the run); write notes to `EXPERIMENT_NOTES.md` as incidents happen.