# Project instructions
DO NOT EDIT FILES OUTSIDE OF /home/shil6647/attack-llm-judge/* or /data/shil6647/* EVER!!!!!!
IF YOU THINK YOU HAVE TO, ASK MAX (HUMAN OPERATOR) FIRST!
DO NOT REMOVE OR DELETE DATA IN THE DATABASE!

You are on a shared environment: don't add additional GPU usage without asking Max.

- Make small, focused git commits for the changes you make. Prefer many small commits over one large one.
- Push all commits to Github remote, so that if this server is lost for some reason I can recover the code. Similarly, push model checkpoints to Hugging Face (my username is daxmavy)
- Add experiment notes (design decisions, bugfixes, etc) to EXPERIMENT_NOTES.md
- Literature review is at LITERATURE.md
- default to autonomy — act first, ask only when a decision is genuinely blocking, irreversible, or ambiguous in a way that cheap exploration can't resolve. Don't ask just because you can.

**Autonomy** Within Max's workspace (described above), you have complete autonomy. Don't wait to ask questions to Max if you can progress with a non-destructive action.

**Run-monitoring rule:** every time you launch a training or evaluation run, schedule a check-in for yourself every **5 minutes** until the run has either concluded or been killed. "Check-in" = check whether your process is competing with other users' processes, read the log tail, confirm the process is alive, look for OOM / errors / stalled progress, and grab the latest metric values. Don't let a run drift for 20+ minutes without looking — the feedback loop is what catches silent failures (IS-ratio collapse, length drift, OOM) early. If nothing has changed in the log since the last check, note that and look again in another 5 minutes.

**Raise-don't-hide rule:** a crash-free run is not a successful run. Every time you report on a run, actively scan its output for *anomalies that a human operator wouldn't spot without you pointing them out*, and surface each one as a problem to investigate — not as a footnote, caveat, or "cosmetic" aside. Examples of things you MUST raise (not explain away):
- Performance well outside the expected envelope (step time, tok/s, load time) relative to similar-sized models/configs.
- Fallback paths engaging silently: FA2→xformers, bnb→CPU offload, vLLM→HF generate, "fast_inference disabled", "kernel not found, using reference impl".
- Metrics that reconcile to zero (`loss=0`, `grad_norm=0`, `lr=0`, `reward_std=0`, `frac_reward_zero_std=1`) — even when mathematically expected, these mean no learning happened on that step and are a design-smell worth flagging.
- Reporting inconsistencies that hint at misconfiguration: `Trainable parameters = 0 of N`, adapter size vs. target-module count mismatches, dtype mismatches in the banner.
- Deprecation warnings from the exact library versions you monkey-patched (they often foreshadow the next break).

The test: "Would Max notice this from the SMOKE DONE line alone?" If no, raise it explicitly in your report and propose the investigation/fix. Monitoring exists to *surface* problems early, not to confirm that the run technically finished. Don't make Max ask why something looks slow or wrong — if it looks slow or wrong to you, say so up front.

**Disk-headroom rule:** `/workspace` has a hard **150 GB quota**. Before every training or eval run that will write to `/workspace` (checkpoints, logs, caches), check `du -sh /workspace` and confirm headroom >= the expected write volume (saved model shards + wandb + logs, typically 5-10 GB). If close to the limit, clean up first — `/workspace/pip_cache`, orphan HF blobs (e.g. duplicate `consolidated.pth` alongside safetensors), and superseded `grpo_run/final_*` / `ckpt_*` dirs are the usual suspects. Never leave <10 GB headroom going into a save-heavy run; silent partial-write failures from quota are how T9/T10 died. When you're done with a checkpoint, push it to HF.

**Judge inference rule:** never run judge models via HF `model.generate()` — it's too slow for our per-step / per-eval scoring volumes. Judge inference must go through vLLM. If vLLM doesn't support a given judge model, downgrade to a worse judge the current vLLM does support rather than falling back to HF generate. (Applies to both training-time scoring and held-out eval scoring.)

---
