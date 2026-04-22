# Project instructions
DO NOT EDIT FILES OUTSIDE OF /home/shil6647/attack-llm-judge/* or /data/shil6647/* EVER!!!!!!
DO NOT REMOVE OR DELETE DATA IN THE DATABASE!

You are on a shared environment: don't add additional GPU usage without asking Max. Before launching any job, `nvidia-smi` to confirm the GPUs you plan to use are idle — another user on the box will block CUDA graph capture or OOM you if you pile on.

- Make small, focused git commits for the changes you make. Prefer many small commits over one large one.
- Push all commits to Github remote branch, so that if this server is lost for some reason I can recover the code.
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

**Judge inference rule.** Never run judge models via HF `model.generate()` — it's too slow for our per-step / per-eval scoring volumes. Judge inference must go through vLLM. If vLLM doesn't support a given judge model, downgrade to a worse judge the current vLLM does support rather than falling back to HF generate. (Applies to both training-time scoring and held-out eval scoring.) **Always via the out-of-process HTTP judge server** (`judge.http_client.JudgeHTTP` + `spawn_judge_server`). Do NOT instantiate a judge vLLM engine in the same Python process as the GRPOTrainer/rewriter vLLM — in-process co-location on one A100 is what invalidated the 2026-04-21 run. The judge server owns GPU 0; the rewriter process owns GPU 1.

**Rewriter swap rule.** When debugging a rewriter candidate that is failing in vLLM, GRPO, or both, give yourself **~10 genuine debug attempts** before declaring the candidate dead and switching to the next one on the backup list. A "genuine attempt" is a distinct root-cause hypothesis plus a targeted fix — reinstalling the same version of the same package four times counts as one attempt. Record each attempt (symptom, hypothesis, fix tried, result) in EXPERIMENT_NOTES.md so the next retry doesn't repeat work. Switching rewriters is cheaper than losing the whole night to one stuck candidate.

**Model-ID source-of-truth rule.** The rewriter HF id, the judge panel (`JUDGE_REGISTRY`), and the fold rotation (`FOLDS`) all live in **`config/models.py`** and nowhere else. Never hardcode a model id or a fold rotation inside a script, shell script, or model card — import from `config.models` and call `require_config()` at the top of every `main()`. The 2026-04-21 mission was invalidated because three scripts kept drifted local copies; the HF checkpoint ended up advertising a different judge panel than was actually used at training time. If you need a new rewriter or judge, edit `config/models.py` once and every downstream script picks it up. When reading a model id back out in a log banner, HF commit message, or model-card, always source it from `config.models.REWRITER` / `JUDGE_REGISTRY` so the written record matches the weights that were actually loaded.

**GPU topology rule.** The production pipeline targets **2× A100 80GB**: GPU 0 serves the 2 in-panel judges via vLLM, GPU 1 runs the rewriter (vLLM rollouts + QLoRA GRPOTrainer). Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and pin each process with `CUDA_VISIBLE_DEVICES`. Co-locating all three vLLM engines on a single A100 (the 2026-04-21 pattern) is a known-bad configuration — it forces JudgeVLLM down to `gpu_memory_utilization≈0.28` and corners the rewriter into `≤0.38`, which silently overcommits when the judge panel rotates. If you have only one GPU available (another user on the box, or the host is truly single-A100), stop and flag it to Max rather than falling back to co-location.

**GRPO step budget.** GRPO runs are locked at **400 steps** (MISSION.md §7 + REPLICATION.md §4). Shorter runs (e.g. a 100-step probe) are acceptable for smoke/debug only — never report them as a mission deliverable without Max's explicit sign-off.

---
