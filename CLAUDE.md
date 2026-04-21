# Project instructions
DO NOT EDIT FILES OUTSIDE OF /home/shil6647/attack-llm-judge/* or /data/shil6647/* EVER!!!!!!
IF YOU THINK YOU HAVE TO, ASK MAX (HUMAN OPERATOR) FIRST!
DO NOT REMOVE OR DELETE DATA IN THE DATABASE!

You are on a shared environment: don't add additional GPU usage without asking Max. Before launching any job, `nvidia-smi` to confirm the GPUs you plan to use are idle — another user on the box will block CUDA graph capture or OOM you if you pile on.

- Make small, focused git commits for the changes you make. Prefer many small commits over one large one.
- Push all commits to Github remote, so that if this server is lost for some reason I can recover the code.
- Add experiment notes (design decisions, bugfixes, etc) to EXPERIMENT_NOTES.md
- Literature review is at LITERATURE.md
- Current mission brief lives in MISSION.md — read it before deciding what to work on
- default to autonomy — act first, ask only when a decision is genuinely blocking, irreversible, or ambiguous in a way that cheap exploration can't resolve. Don't ask just because you can.

**Autonomy.** Within Max's workspace (described above), you have complete autonomy. Don't wait to ask questions to Max if you can progress with a non-destructive action.

**Conda env.** All Python work runs in the `daxmavy` conda environment. That env is Max's to modify — `pip install` / `pip install -U` freely if the mission needs a different stack. Don't touch any other conda env on the box, and don't install anything system-wide (`apt`, `/opt`, `/usr`). If you need a different Python version or a clean slate, create a new env under Max's home, don't mutate shared ones.

**Run-monitoring rule.** Every time you launch a training or evaluation run, schedule a check-in for yourself every **5 minutes** until the run has either concluded or been killed. "Check-in" = read the log tail, confirm the process is alive, look for OOM / errors / stalled progress, check `nvidia-smi` for interference from other users, and grab the latest metric values. Don't let a run drift for 20+ minutes without looking — the feedback loop is what catches silent failures (IS-ratio collapse, length drift, OOM) early.

**Always pair a run with a streaming Monitor.** The 5-min ScheduleWakeup is a backup against silent hangs; the primary crash signal is a `Monitor` tool watching the log. Right after `nohup python ... > $LOG &`, fire a `Monitor` grepping for BOTH crash signatures AND progress markers:
  - crash: `Traceback|Error|OOM|Killed|assert|FAILED|exited|terminated|RuntimeError|CUDA error`
  - progress: the stage prints this specific script emits (e.g. `loaded in|reward_fn call|step=|SMOKE DONE`)
Use `persistent: true` for runs longer than 5 min, or set `timeout_ms` to the expected duration. A monitor that greps only success markers is silent on a crashloop — always include the crash alternation.

**Raise-don't-hide rule.** A crash-free run is not a successful run. Every time you report on a run, actively scan its output for *anomalies that a human operator wouldn't spot without you pointing them out*, and surface each one as a problem to investigate — not as a footnote, caveat, or "cosmetic" aside. Examples of things you MUST raise (not explain away):
- Performance well outside the expected envelope (step time, tok/s, load time) relative to similar-sized models/configs.
- Fallback paths engaging silently: FA2→xformers, bnb→CPU offload, vLLM→HF generate, "fast_inference disabled", "kernel not found, using reference impl", "dynamic quant → normal quant".
- Metrics that reconcile to zero (`loss=0`, `grad_norm=0`, `lr=0`, `reward_std=0`, `frac_reward_zero_std=1`) — even when mathematically expected, these mean no learning happened on that step and are a design-smell worth flagging.
- Reporting inconsistencies that hint at misconfiguration: `Trainable parameters = 0 of N`, adapter size vs. target-module count mismatches, dtype mismatches in the banner.
- Deprecation warnings from the exact library versions you monkey-patched (they often foreshadow the next break).

The test: "Would Max notice this from the SMOKE DONE line alone?" If no, raise it explicitly at the top of your report and propose the investigation/fix. Monitoring exists to *surface* problems early, not to confirm that the run technically finished. Don't make Max ask why something looks slow or wrong — if it looks slow or wrong to you, say so up front.

**Disk rule.** `/home` is a shared filesystem — it is *full* from other users' usage and there are single-digit GB free at any given time. Write nothing heavy (model shards, HF cache, wandb logs, rewrite dumps) under `/home`. Heavy artifacts live under **`/data/shil6647/attack-llm-judge/`** (on a 9.1 TB FS with >1 TB free); the subtree is already laid out with `hf_cache/`, `grpo_run/`, `final_models/`, `vllm_cache/`, `tmp/`, `smoke_test/`. Set `HF_HOME=/data/shil6647/attack-llm-judge/hf_cache` and `TRANSFORMERS_CACHE=$HF_HOME` in any launch script. Before every run that will write artifacts, `du -sh /data/shil6647` and confirm headroom >= expected write (typically 20-60 GB for a 14-32B model checkpoint + rollouts + wandb). When a checkpoint is no longer needed, push to HF (`daxmavy/...`) and delete the local copy after verifying HF round-trip.

**Judge inference rule.** Never run judge models via HF `model.generate()` — it's too slow for our per-step / per-eval scoring volumes. Judge inference must go through vLLM. If vLLM doesn't support a given judge model, downgrade to a worse judge the current vLLM does support rather than falling back to HF generate. (Applies to both training-time scoring and held-out eval scoring.)

**Rewriter swap rule.** When debugging a rewriter candidate that is failing in vLLM, GRPO, or both, give yourself **~10 genuine debug attempts** before declaring the candidate dead and switching to the next one on the backup list. A "genuine attempt" is a distinct root-cause hypothesis plus a targeted fix — reinstalling the same version of the same package four times counts as one attempt. Record each attempt (symptom, hypothesis, fix tried, result) in EXPERIMENT_NOTES.md so the next retry doesn't repeat work. Switching rewriters is cheaper than losing the whole night to one stuck candidate.

---
