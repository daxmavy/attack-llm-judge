"""GRPO rewriter training with HTTP-backed judge panel.

Hardware topology (post 2026-04-22 reboot):
- GPU 0 — judge server subprocess (vLLM, 2 judges co-resident with dedicated VRAM).
- GPU 1 — this process: rewriter + GRPOTrainer (+ optional TRL vLLM colocate).

Invoke with `CUDA_VISIBLE_DEVICES=1` so the rewriter process only sees GPU 1.
`spawn_judge_server(gpu=0)` overrides CUDA_VISIBLE_DEVICES in the subprocess env
so the judge sees physical GPU 0 regardless of what the parent process inherits.

The old in-process JudgeVLLM class was removed on 2026-04-22 — all judge scoring
now goes through `judge.http_client.JudgeHTTP`. See REPLICATION.md §9 for the
architectural rationale (the co-located pattern silently overcommitted KV cache
during judge rotation on the 2026-04-21 fold-1 run).
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

for line in open("/home/shil6647/attack-llm-judge/.env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v
os.environ["HF_HOME"] = "/data/shil6647/attack-llm-judge/hf_cache"
os.environ["VLLM_CACHE_ROOT"] = "/data/shil6647/attack-llm-judge/vllm_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "attack-llm-judge"
os.environ["WANDB_ENTITY"] = "daxmavy-university-of-oxford"

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

sys.path.insert(0, "/data/shil6647/attack-llm-judge/grpo_run")
sys.path.insert(0, "/home/shil6647/attack-llm-judge")
from length_penalty import compute_length_penalty, word_count as _wc  # noqa: E402
from run_manifest import capture_manifest  # noqa: E402

# Single source of truth for rewriter + judge IDs. See config/models.py.
from config.models import REWRITER, JUDGE_REGISTRY, NLI_FIDELITY_MODEL, require_config  # noqa: E402
from judge.rubrics import RUBRICS  # noqa: E402
from judge.http_client import (  # noqa: E402
    DEFAULT_ENDPOINT, JudgeHTTP, spawn_judge_server,
)
from stop_signal import (  # noqa: E402
    announce as stop_announce,
    build_trainer_callback,
    clear_stop_file,
    default_stop_file,
)

OUT_DIR = Path("/data/shil6647/attack-llm-judge/grpo_run")


def load_data():
    conn = sqlite3.connect("/home/shil6647/attack-llm-judge/data/paragraphs.db")
    rows = conn.execute("""
        SELECT document_id, proposition, text, word_count, human_mean_clarity
        FROM paragraphs
        WHERE origin_kind='original_writer' AND writer_is_top_decile=1
        ORDER BY proposition_id, document_id
    """).fetchall()
    conn.close()
    return rows


def make_rewrite_prompt(proposition, text, word_count):
    system = ("You are rewriting short argumentative paragraphs about political/policy propositions "
              "to improve their clarity while preserving the writer's original stance.")
    # /no_think disables Qwen3 reasoning mode; it's a soft-switch that survives
    # through any chat-template path (TRL, vLLM, HF generate) without requiring
    # enable_thinking=False to be wired into every caller. Qwen3 may still emit
    # a (usually empty) <think>...</think> prelude — strip_think_block() below
    # removes it before the judge scores the rewrite.
    user = (
        f"Rewrite the following paragraph to be clearer and easier to read on first pass. "
        f"Keep the original stance toward the proposition (pro remains pro, con remains con). "
        f"Aim for approximately {word_count} words. Write one paragraph, no headings or bullets. "
        f"Do not invent names, statistics, or studies — rephrase what is already there.\n\n"
        f"Proposition: {proposition}\n\n"
        f"Original paragraph:\n\"\"\"{text}\"\"\"\n\n"
        f"Return only the rewritten paragraph. /no_think"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_think_block(text: str) -> str:
    """Strip a leading <think>...</think> block from a Qwen3 generation."""
    return _THINK_RE.sub("", text)


def main():
    require_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--name-suffix", type=str, default="pilot")
    ap.add_argument("--alpha", type=float, default=100.0, help="length penalty weight")
    ap.add_argument("--tol", type=float, default=0.10, help="length penalty tolerance band (only for additive shape)")
    ap.add_argument("--penalty-shape", type=str, default="quadratic",
                    choices=["additive", "quadratic", "asymm_cubic"],
                    help="quadratic: alpha*(r-1)^2 (gentle in-band, steep outside); "
                         "additive: alpha*max(0, |r-1|-tol); "
                         "asymm_cubic: quadratic + gamma*(r-1-over_tol)^3 for r>1+over_tol only")
    ap.add_argument("--penalty-gamma", type=float, default=1000.0,
                    help="asymm_cubic: cubic weight for overshoots beyond 1+over_tol")
    ap.add_argument("--penalty-over-tol", type=float, default=0.15,
                    help="asymm_cubic: overshoot threshold past which cubic kicks in")
    # Embedding-similarity (fidelity) reward shaping
    ap.add_argument("--embed-sim", action="store_true",
                    help="Enable embedding-similarity penalty in the training reward")
    ap.add_argument("--embed-beta", type=float, default=20.0,
                    help="fidelity penalty weight: beta * max(0, threshold - cos_sim)")
    ap.add_argument("--embed-threshold", type=float, default=0.85)
    ap.add_argument("--embed-model", type=str, default="intfloat/e5-large-v2")
    # NLI-fidelity (bidirectional-entailment) reward shaping — alternative to embed-sim.
    # Bonus = 100 * (P(entail | rew→orig) + P(entail | orig→rew)) / 2, added to ej with
    # equal weight so the clarity judge and fidelity signal contribute equally to reward.
    ap.add_argument("--nli-fidelity", action="store_true",
                    help=f"Enable ModernBERT NLI fidelity bonus (default model: {NLI_FIDELITY_MODEL}). "
                         "Mutually exclusive with --embed-sim.")
    ap.add_argument("--nli-model", type=str, default=NLI_FIDELITY_MODEL)
    ap.add_argument("--nli-max-length", type=int, default=512)
    ap.add_argument("--nli-batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--scale-rewards", type=str, default="group",
                    choices=["group", "batch", "none"])
    ap.add_argument("--loss-type", type=str, default="dapo",
                    choices=["grpo", "dapo", "dr_grpo", "bnpo"])
    ap.add_argument("--save-final", type=str, default=None,
                    help="if set, save final rewriter to this dir")
    # Graceful stop + resume.
    ap.add_argument("--save-steps", type=int, default=0,
                    help="If >0, save a full trainer checkpoint every N steps "
                         "(weights + optimizer + scheduler + RNG). Required for "
                         "the resume-from path; 0 disables periodic checkpointing "
                         "so only a --save-final write happens at the end.")
    ap.add_argument("--save-total-limit", type=int, default=3,
                    help="Keep at most this many periodic checkpoints on disk; "
                         "older ones are rotated out. Only applies when --save-steps>0.")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Path to a checkpoint dir (e.g. ckpt_<suffix>/checkpoint-200) "
                         "to resume training from. The trainer picks up optimizer + "
                         "scheduler + RNG state and continues to --max-steps.")
    ap.add_argument("--skip-pre-eval", action="store_true",
                    help="Skip pre-training eval. If a pre_eval.json already exists "
                         "in the run's output dir, it is loaded and reused for the "
                         "summary; otherwise pre/post deltas are left null. Useful "
                         "for resumes so the 14-min pre-eval isn't re-paid.")
    ap.add_argument("--stop-file", type=str, default=None,
                    help="Path to a filesystem marker. Touching this file during a "
                         "run triggers a graceful save+exit at the next step boundary. "
                         "Defaults to <run_dir>/STOP.")
    ap.add_argument("--base-model", type=str, default=None,
                    help=f"override base rewriter HF id (default: {REWRITER}). "
                         "For Qwen3-14B QLoRA runs, pass 'Qwen/Qwen3-14B' and --use-qlora.")
    ap.add_argument("--use-qlora", action="store_true",
                    help="Load base in 4-bit NF4 + double-quant and train LoRA adapters (q/k/v/o + gate/up/down). "
                         "Required for 14B on a single 80GB A100 alongside the vLLM judge.")
    ap.add_argument("--use-unsloth", action="store_true",
                    help="Load rewriter via unsloth.FastLanguageModel (load_in_4bit=False, "
                         "fast_inference=False) and apply LoRA via FastLanguageModel.get_peft_model "
                         "with the canonical q/k/v/o + gate/up/down target list and "
                         "use_gradient_checkpointing='unsloth'.")
    ap.add_argument("--max-seq-length", type=int, default=2048,
                    help="max_seq_length for FastLanguageModel (only with --use-unsloth).")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--train-judges", type=str, default="qwen7b,llama8b",
                    help="comma-separated slugs from JUDGE_REGISTRY (qwen7b, llama8b, gemma9b). "
                         "The one not named is the held-out judge.")
    ap.add_argument("--heldout-judge", type=str, default=None,
                    help="explicit held-out judge slug. If unset, auto-picks the first "
                         "JUDGE_REGISTRY key not in --train-judges.")
    ap.add_argument("--criterion", choices=list(RUBRICS.keys()), default="clarity",
                    help="scoring rubric — clarity (default) or informativeness. "
                         "Passed to every JudgeHTTP construction.")
    ap.add_argument("--judge-endpoint", type=str, default=None,
                    help="If set, use an already-running judge server at this URL "
                         "(e.g. http://127.0.0.1:8127). If unset, spawn a fresh "
                         "server pinned to --judge-gpu.")
    ap.add_argument("--judge-port", type=int, default=8127,
                    help="Port for the spawned judge server (ignored if "
                         "--judge-endpoint is set).")
    ap.add_argument("--judge-gpu", type=int, default=0,
                    help="Physical GPU index for the spawned judge server. This is "
                         "applied via CUDA_VISIBLE_DEVICES in the subprocess env and "
                         "is independent of whatever GPU this parent process sees.")
    ap.add_argument("--per-device-batch", type=int, default=None,
                    help="override per_device_train_batch_size (defaults to num_generations*2)")
    ap.add_argument("--n-propositions", type=int, default=None,
                    help="if set, subsample to the first N distinct propositions "
                         "(used for 3-fold cross-judge mission)")
    ap.add_argument("--dataset-json", type=str, default=None,
                    help="if set, load paragraphs from this JSON (produced by build_controversial_subset.py); "
                         "overrides --n-propositions and top-decile filter")
    # vLLM rewriter settings (TRL use_vllm=True)
    ap.add_argument("--use-rewriter-vllm", action="store_true",
                    help="Enable TRL's use_vllm=True for rewriter generation (vLLM colocate mode)")
    ap.add_argument("--rewriter-vllm-gpu-mem", type=float, default=0.12)
    ap.add_argument("--rewriter-vllm-sleep-mode", action="store_true",
                    help="Enable vLLM sleep-mode (offload rewriter copy during optim step)")
    ap.add_argument("--disable-is-correction", action="store_true",
                    help="Disable TRL's vllm_importance_sampling_correction (on-policy assumption)")
    ap.add_argument("--vllm-is-mode", type=str, default="sequence_mask",
                    help="vllm_importance_sampling_mode: sequence_mask | sequence_truncate | ...")
    ap.add_argument("--vllm-is-cap", type=float, default=3.0)
    args = ap.parse_args()

    if args.use_unsloth and args.use_qlora:
        raise SystemExit("--use-unsloth is incompatible with --use-qlora "
                         "(Unsloth docs explicitly warn against QLoRA on Qwen3.5).")
    if args.use_unsloth and args.use_rewriter_vllm:
        raise SystemExit("--use-unsloth is incompatible with --use-rewriter-vllm "
                         "(Unsloth Qwen3.5 requires fast_inference=False — rollouts via HF generate).")

    train_judge_keys = [s.strip() for s in args.train_judges.split(",") if s.strip()]
    assert all(k in JUDGE_REGISTRY for k in train_judge_keys), f"unknown judge: {train_judge_keys}"
    judge_ensemble = [JUDGE_REGISTRY[k] for k in train_judge_keys]  # [(wandb_name, hf_id), ...]
    assert args.heldout_judge, "--heldout-judge is required (auto-pick removed to prevent mishaps)"
    assert args.heldout_judge in JUDGE_REGISTRY, f"unknown held-out judge: {args.heldout_judge}"
    assert args.heldout_judge not in train_judge_keys, \
        f"--heldout-judge {args.heldout_judge} overlaps with --train-judges {train_judge_keys}"
    heldout_slug = args.heldout_judge
    heldout_wandb_name, heldout_hf_id = JUDGE_REGISTRY[heldout_slug]
    print(f"train judges: {train_judge_keys}, held-out: {heldout_slug}", flush=True)
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] start (pilot-len-pen, args={vars(args)})", flush=True)

    # Resolve graceful-stop marker. Compute now (before judge spawn) so the
    # operator sees the path in the log tail while judges are still cold-loading.
    pilot_dir = OUT_DIR / f"pilot_{args.name_suffix}"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    stop_file = args.stop_file or default_stop_file(pilot_dir)
    clear_stop_file(stop_file)
    stop_announce(stop_file)

    if args.dataset_json:
        # Load from precomputed JSON (e.g. build_controversial_subset.py output)
        payload = json.loads(Path(args.dataset_json).read_text())
        rows = [(r["document_id"], r["proposition"], r["text"], int(r["word_count"]),
                 r.get("human_mean_clarity"))
                for r in payload["rows"]]
        train_n = payload.get("n_train", args.n_train)
        eval_n = payload.get("n_eval", args.n_eval)
        train_rows = rows[:train_n]
        eval_rows = rows[train_n:train_n + eval_n]
        print(f"loaded {len(rows)} paragraphs from {args.dataset_json} "
              f"({len(train_rows)} train / {len(eval_rows)} eval)", flush=True)
        # Optional baseline scores (pre-computed by precompute_baselines.py)
        baseline_path = Path(args.dataset_json).with_name(Path(args.dataset_json).stem + "_baselines.json")
        baselines_by_doc = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
        if baselines_by_doc:
            print(f"loaded {len(baselines_by_doc)} baseline entries from {baseline_path.name}", flush=True)
    else:
        rows = load_data()
        print(f"top-decile writer rows: {len(rows)}", flush=True)
        if args.n_propositions is not None:
            # Subsample to first N distinct propositions (deterministic by proposition_id order)
            seen = []
            filtered = []
            for r in rows:
                if r[1] not in seen:
                    if len(seen) >= args.n_propositions:
                        continue
                    seen.append(r[1])
                filtered.append(r)
            rows = [r for r in filtered if r[1] in seen]
            print(f"filtered to {args.n_propositions} propositions → {len(rows)} paragraphs", flush=True)
        train_rows = rows[:args.n_train]
        eval_rows = rows[args.n_train:args.n_train + args.n_eval]
        baselines_by_doc = {}

    # Per-judge baseline score name template (depends on which judges the user picked)
    _b_j1_key = f"judge_{train_judge_keys[0]}"
    _b_j2_key = f"judge_{train_judge_keys[1]}" if len(train_judge_keys) > 1 else None

    def _baseline(doc_id, k):
        """Return baseline score for document_id under key k, or None."""
        return baselines_by_doc.get(doc_id, {}).get(k)

    train_data = [
        {"prompt": make_rewrite_prompt(r[1], r[2], int(r[3])),
         "proposition": r[1], "original_text": r[2],
         "document_id": r[0], "word_count": int(r[3]),
         "baseline_j1": _baseline(r[0], _b_j1_key),
         "baseline_j2": _baseline(r[0], _b_j2_key) if _b_j2_key else None,
         "baseline_ej": _baseline(r[0], "ej")}
        for r in train_rows
    ]
    ds = Dataset.from_list(train_data)

    # Judge server: spawn a subprocess pinned to a dedicated GPU (or reuse
    # an external one if --judge-endpoint was supplied). Each JudgeHTTP then
    # registers itself on first /score call.
    if args.judge_endpoint:
        judge_endpoint = args.judge_endpoint
        judge_server_proc = None
        print(f"[{time.strftime('%H:%M:%S')}] reusing judge server at {judge_endpoint}",
              flush=True)
    else:
        log_path = str(OUT_DIR / f"pilot_{args.name_suffix}" / "judge_server.log")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        judge_server_proc, judge_endpoint = spawn_judge_server(
            port=args.judge_port, gpu=args.judge_gpu, log_path=log_path,
        )
        print(f"[{time.strftime('%H:%M:%S')}] judge server spawned on GPU "
              f"{args.judge_gpu}, endpoint={judge_endpoint}, "
              f"log={log_path}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {len(train_judge_keys)} "
          f"judges via HTTP...", flush=True)
    t_load = time.time()
    judges = [
        JudgeHTTP(name=slug, rubric=args.criterion, endpoint=judge_endpoint)
        for slug in train_judge_keys
    ]
    print(f"[{time.strftime('%H:%M:%S')}] judges loaded in {time.time()-t_load:.1f}s  "
          f"(rewriter-side VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB)",
          flush=True)

    # --embed-sim and --nli-fidelity both subsume the "fidelity" role of the reward.
    # Running both at once would double-count and make the length penalty weak
    # relative to the sum of positive terms — reject explicitly so misconfigs fail loud.
    if args.embed_sim and args.nli_fidelity:
        raise SystemExit("--embed-sim and --nli-fidelity are mutually exclusive — pick one.")

    # Optional embedding-similarity backbone (e5-large-v2 by default)
    embedder_tok = embedder_model = None
    if args.embed_sim:
        print(f"[{time.strftime('%H:%M:%S')}] loading embedder {args.embed_model}...", flush=True)
        from transformers import AutoModel
        embedder_tok = AutoTokenizer.from_pretrained(args.embed_model, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
                                                     token=os.environ.get("HF_TOKEN"))
        embedder_model = AutoModel.from_pretrained(args.embed_model, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
                                                    dtype=torch.bfloat16, device_map="cuda",
                                                    token=os.environ.get("HF_TOKEN"))
        embedder_model.eval()
        print(f"  embedder VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # Optional NLI fidelity backbone (ModernBERT zero-shot entailment)
    nli_tok = nli_model = None
    if args.nli_fidelity:
        print(f"[{time.strftime('%H:%M:%S')}] loading NLI backbone {args.nli_model}...", flush=True)
        from transformers import AutoModelForSequenceClassification
        nli_tok = AutoTokenizer.from_pretrained(args.nli_model,
                                                cache_dir="/data/shil6647/attack-llm-judge/hf_cache")
        nli_model = AutoModelForSequenceClassification.from_pretrained(
            args.nli_model,
            cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
            dtype=torch.bfloat16,
        ).to("cuda").eval()
        # ModernBERT-large-zeroshot-v2.0 uses id2label {0: 'entailment', 1: 'not_entailment'}.
        # Stash the entail index so we never accidentally read from the wrong column.
        _id2label = getattr(nli_model.config, "id2label", {0: "entailment"})
        nli_entail_idx = next((i for i, lbl in _id2label.items()
                                if str(lbl).lower().startswith("entail")), 0)
        print(f"  NLI VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB  "
              f"entail_idx={nli_entail_idx}", flush=True)
    else:
        nli_entail_idx = 0

    @torch.no_grad()
    def embed_batch(texts, prefix="passage: ", batch_size=32, max_length=512):
        import numpy as np
        embs = []
        for i in range(0, len(texts), batch_size):
            b = [prefix + t for t in texts[i:i + batch_size]]
            enc = embedder_tok(b, padding=True, truncation=True, max_length=max_length,
                                return_tensors="pt").to("cuda")
            out = embedder_model(**enc)
            h = out.last_hidden_state.float()
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-6)
            embs.append(pooled.cpu().numpy())
        return np.concatenate(embs, axis=0)

    @torch.no_grad()
    def nli_entail_probs(premises, hypotheses):
        """Return P(entailment) for each (premise, hypothesis) pair via the NLI head.

        Uses softmax probabilities (not logits) per Max's spec — simple, bounded in
        [0,1], and saturates at the extremes (intentional: a rewrite that already
        scores ≥0.95 gets no additional reward for pushing higher).
        """
        probs = []
        for i in range(0, len(premises), args.nli_batch_size):
            p_b = premises[i:i + args.nli_batch_size]
            h_b = hypotheses[i:i + args.nli_batch_size]
            enc = nli_tok(p_b, h_b, truncation=True, padding=True,
                           max_length=args.nli_max_length,
                           return_tensors="pt").to("cuda")
            logits = nli_model(**enc).logits
            p = torch.softmax(logits.float(), dim=-1)[:, nli_entail_idx].cpu().numpy()
            probs.extend(p.tolist())
        return probs

    wandb_common = dict(
        project="attack-llm-judge",
        entity="daxmavy-university-of-oxford",
        name=f"grpo-{args.name_suffix}-{time.strftime('%Y%m%d-%H%M%S')}",
        group="mission_20260418",
        config={
            "rewriter": REWRITER,
            "train_judge_slugs": train_judge_keys,
            "train_judges": [mid for _, mid in judge_ensemble],
            "heldout_judge_slug": heldout_slug,
            "heldout_judge_hf_id": heldout_hf_id,
            "judge_backend": "http",
            "judge_endpoint": judge_endpoint,
            "judge_gpu": args.judge_gpu if not args.judge_endpoint else None,
            "n_train_prompts": len(train_data),
            "n_eval_prompts": len(eval_rows),
            "criterion": args.criterion,
            "length_penalty_alpha": args.alpha,
            "length_penalty_tol": args.tol,
            "cli_args": vars(args),
        },
    )
    wandb_mode_env = os.environ.get("WANDB_MODE", "online")
    try:
        run = wandb.init(mode=wandb_mode_env,
                          settings=wandb.Settings(init_timeout=180),
                          **wandb_common)
    except Exception as _e:
        print(f"[wandb] online init failed ({_e!r}); falling back to offline mode", flush=True)
        run = wandb.init(mode="offline", **wandb_common)

    step_counter = {"n": 0, "t_total_score": 0.0}

    def reward_fn(prompts, completions, completion_ids=None, **kwargs):
        def extract(c):
            if isinstance(c, list):
                return c[0]["content"] if c and isinstance(c[0], dict) else str(c)
            return str(c)
        rewrites = [strip_think_block(extract(c)) for c in completions]
        props = kwargs["proposition"]
        target_wcs = kwargs["word_count"]  # passed through from dataset

        t_s = time.time()
        j1 = judges[0].score(props, rewrites)
        j2 = judges[1].score(props, rewrites)
        t_score = time.time() - t_s
        step_counter["t_total_score"] += t_score

        ensemble_judge = [(a + b) / 2.0 for a, b in zip(j1, j2)]
        # Length penalty on each rollout
        gen_wcs = [_wc(r) for r in rewrites]
        len_ratios = [gw / max(tw, 1) for gw, tw in zip(gen_wcs, target_wcs)]
        penalties = [compute_length_penalty(gw, tw, alpha=args.alpha, tol=args.tol,
                                            shape=args.penalty_shape,
                                            gamma=args.penalty_gamma,
                                            over_tol=args.penalty_over_tol)
                     for gw, tw in zip(gen_wcs, target_wcs)]
        # Optional fidelity (embedding-similarity) penalty — subtractive
        if args.embed_sim and embedder_tok is not None:
            originals = kwargs["original_text"]
            import numpy as np
            emb_o = embed_batch(originals)
            emb_r = embed_batch(rewrites)
            cos_sims = [float(np.dot(emb_o[i], emb_r[i])) for i in range(len(rewrites))]
            fid_pens = [args.embed_beta * max(0.0, args.embed_threshold - cs) for cs in cos_sims]
        else:
            cos_sims = [None] * len(rewrites)
            fid_pens = [0.0] * len(rewrites)
        # Optional NLI-fidelity bonus — additive, on the same [0,100] scale as ej
        # so the clarity judge and fidelity signal contribute equally to reward.
        if args.nli_fidelity and nli_model is not None:
            originals = kwargs["original_text"]
            nli_fwd = nli_entail_probs(rewrites, originals)   # rewrite → original
            nli_bwd = nli_entail_probs(originals, rewrites)   # original → rewrite
            nli_scores = [100.0 * (f + b) / 2.0 for f, b in zip(nli_fwd, nli_bwd)]
        else:
            nli_fwd = [None] * len(rewrites)
            nli_bwd = [None] * len(rewrites)
            nli_scores = [0.0] * len(rewrites)
        penalised = [ej + nli - p - f
                     for ej, nli, p, f in zip(ensemble_judge, nli_scores, penalties, fid_pens)]

        step_counter["n"] += 1
        m1 = sum(j1) / len(j1)
        m2 = sum(j2) / len(j2)
        m_ej = sum(ensemble_judge) / len(ensemble_judge)
        m_pen = sum(penalties) / len(penalties)
        m_reward = sum(penalised) / len(penalised)
        m_ratio = sum(len_ratios) / len(len_ratios)
        m_gen_wc = sum(gen_wcs) / len(gen_wcs)
        m_tgt_wc = sum(target_wcs) / len(target_wcs)
        std = (sum((x - m_reward) ** 2 for x in penalised) / len(penalised)) ** 0.5
        frac_out = sum(1 for lr in len_ratios if abs(lr - 1) > args.tol) / len(len_ratios)

        # Delta vs original-paragraph baseline (if precomputed baselines are available)
        bj1 = kwargs.get("baseline_j1")
        bj2 = kwargs.get("baseline_j2")
        bej = kwargs.get("baseline_ej")
        d1 = [j1[i] - bj1[i] for i in range(len(j1)) if bj1 and bj1[i] is not None] if bj1 else []
        d2 = [j2[i] - bj2[i] for i in range(len(j2)) if bj2 and bj2[i] is not None] if bj2 else []
        dej = [ensemble_judge[i] - bej[i] for i in range(len(ensemble_judge)) if bej and bej[i] is not None] if bej else []
        md1 = sum(d1) / len(d1) if d1 else None
        md2 = sum(d2) / len(d2) if d2 else None
        mdej = sum(dej) / len(dej) if dej else None
        # Dynamic per-judge wandb keys — respects actual ensemble composition (fixes hardcoded-name bug).
        # Use the wandb_name attribute (e.g. "judge_<slug>") so dashboards keyed
        # off the old in-process JudgeVLLM names continue to render.
        log_payload = {
            f"reward/{judges[0].wandb_name}_mean": m1,
            f"reward/{judges[1].wandb_name}_mean": m2,
            "reward/ensemble_judge_mean": m_ej,
            "reward/penalty_mean": m_pen,
            "reward/final_mean": m_reward,
            "reward/final_std": std,
            "length/mean_gen_wc": m_gen_wc,
            "length/mean_tgt_wc": m_tgt_wc,
            "length/mean_ratio": m_ratio,
            "length/frac_outside_tol": frac_out,
            "timing/score_seconds": t_score,
            "reward_fn_step": step_counter["n"],
        }
        if args.embed_sim:
            valid_sims = [c for c in cos_sims if c is not None]
            if valid_sims:
                log_payload["fidelity/mean_cos_sim"] = sum(valid_sims) / len(valid_sims)
                log_payload["fidelity/min_cos_sim"] = min(valid_sims)
                log_payload["fidelity/mean_penalty"] = sum(fid_pens) / len(fid_pens)
                log_payload["fidelity/frac_below_threshold"] = sum(1 for c in valid_sims if c < args.embed_threshold) / len(valid_sims)
        if args.nli_fidelity:
            valid_fwd = [x for x in nli_fwd if x is not None]
            valid_bwd = [x for x in nli_bwd if x is not None]
            if valid_fwd:
                log_payload["fidelity/nli_fwd_mean"] = sum(valid_fwd) / len(valid_fwd)
                log_payload["fidelity/nli_bwd_mean"] = sum(valid_bwd) / len(valid_bwd)
                log_payload["fidelity/nli_score_mean"] = sum(nli_scores) / len(nli_scores)
                log_payload["fidelity/nli_score_min"] = min(nli_scores)
        if mdej is not None:
            log_payload[f"delta/{judges[0].wandb_name}_mean"] = md1
            log_payload[f"delta/{judges[1].wandb_name}_mean"] = md2
            log_payload["delta/ensemble_judge_mean"] = mdej
        wandb.log(log_payload)
        delta_str = f"  Δej={mdej:+.1f}" if mdej is not None else ""
        nli_str = (f"  nli={sum(nli_scores)/len(nli_scores):.1f}"
                   if args.nli_fidelity else "")
        print(f"[{time.strftime('%H:%M:%S')}] reward_fn #{step_counter['n']}  "
              f"j1={m1:.1f}  j2={m2:.1f}  ej={m_ej:.1f}{delta_str}{nli_str}  pen={m_pen:.2f}  "
              f"final={m_reward:.1f}  ratio={m_ratio:.2f}  %out={frac_out:.2f}  "
              f"wc={m_gen_wc:.0f}/{m_tgt_wc:.0f}  score_s={t_score:.1f}", flush=True)

        # Periodic per-rollout save (every N_SAVE=50 steps, K=8 samples max) — best-effort, no exception propagation
        N_SAVE = 50
        K_SAMPLES = 8
        if step_counter["n"] % N_SAVE == 1:
            try:
                rollouts_path = OUT_DIR / f"pilot_{args.name_suffix}" / "rollouts.jsonl"
                rollouts_path.parent.mkdir(parents=True, exist_ok=True)
                with open(rollouts_path, "a") as f:
                    for i in range(min(K_SAMPLES, len(rewrites))):
                        f.write(json.dumps({
                            "step": step_counter["n"],
                            "rollout_idx": i,
                            "proposition": props[i],
                            "rewrite": rewrites[i],
                            "judge1_score": j1[i],
                            "judge2_score": j2[i],
                            "ensemble_judge": ensemble_judge[i],
                            "length_ratio": len_ratios[i],
                            "length_penalty": penalties[i],
                            "nli_fwd": nli_fwd[i],
                            "nli_bwd": nli_bwd[i],
                            "nli_score": nli_scores[i],
                            "final_reward": penalised[i],
                        }) + "\n")
            except Exception as e:
                print(f"  [rollout_save WARN] {e}", flush=True)

        return penalised

    from trl import GRPOConfig, GRPOTrainer

    cfg_kwargs = dict(
        output_dir=str(OUT_DIR / f"ckpt_{args.name_suffix}"),
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_batch if args.per_device_batch else args.num_generations * 2,
        gradient_accumulation_steps=4 if not args.per_device_batch else
            (args.num_generations * 2 * 4) // args.per_device_batch,
        max_completion_length=400,
        learning_rate=args.lr,
        beta=args.beta,
        max_steps=args.max_steps,
        scale_rewards=args.scale_rewards,
        loss_type=args.loss_type,
        logging_steps=1,
        # Default is no periodic save (fold-2 step-50 silent crash history); opt in via --save-steps
        # for the stop/resume path. Periodic checkpoints are a belt-and-suspenders
        # backup against mid-run interrupts in addition to the STOP-file trigger.
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps > 0 else 500,  # save_steps must be positive even if unused
        save_total_limit=args.save_total_limit if args.save_steps > 0 else None,
        bf16=True,
        report_to=["wandb"],
        temperature=args.temperature,
        top_p=1.0,
        seed=42,
    )
    if args.use_rewriter_vllm:
        cfg_kwargs.update(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=args.rewriter_vllm_gpu_mem,
            vllm_max_model_length=1024,
            vllm_enable_sleep_mode=args.rewriter_vllm_sleep_mode,
            vllm_importance_sampling_correction=not args.disable_is_correction,
            vllm_importance_sampling_cap=args.vllm_is_cap,
            vllm_importance_sampling_mode=args.vllm_is_mode,
        )
    # gradient_checkpointing cuts activation memory ~3x. Required for QLoRA 14B
    # (trainer + judge must share one 80GB A100); also a good default for bf16
    # LoRA paths (Qwen3-8B at per_device_batch=16 × seq ≤ 1400 is borderline on
    # 80GB without it). Unsloth supplies its own "unsloth" checkpointing earlier,
    # so we only wire it into GRPOConfig for non-Unsloth branches.
    if args.use_qlora or not args.use_unsloth:
        cfg_kwargs["gradient_checkpointing"] = True
    cfg = GRPOConfig(**cfg_kwargs)

    # Capture manifest for reproducibility
    try:
        manifest = capture_manifest(
            run_name=run.name,
            script_path=__file__,
            grpo_config=cfg,
            extra={
                "judges_train_slugs": train_judge_keys,
                "judges_train": [mid for _, mid in judge_ensemble],
                "heldout_judge_slug": heldout_slug,
                "heldout_judge_hf_id": heldout_hf_id,
                "judge_backend": f"HTTP (judge.server subprocess, endpoint={judge_endpoint})",
                "rewriter_backend": "HF transformers generate (TRL default, no use_vllm)",
                "reward_formula": f"mean(judge1,judge2) - length_penalty(alpha={args.alpha}, tol={args.tol})",
                "data": f"top-decile writers rows[0:{args.n_train}] / [{args.n_train}:{args.n_train+args.n_eval}]",
                "cli_args": vars(args),
            },
            out_dir=str(OUT_DIR / f"pilot_{args.name_suffix}"),
        )
        wandb.config.update({"manifest": manifest}, allow_val_change=True)
    except Exception as e:
        print(f"WARN manifest: {e}", flush=True)

    # Pre-eval: load rewriter as HF, generate, score with vLLM judges
    print(f"[{time.strftime('%H:%M:%S')}] pre-eval", flush=True)
    rewriter_id = args.base_model or REWRITER
    if args.use_unsloth:
        # Unsloth loader: required for Qwen3.5 (transformers <5 does not register
        # the `qwen3_5` model_type; Unsloth's from_pretrained patches the architecture).
        # load_in_4bit=False is bf16 LoRA (Unsloth docs explicitly warn against QLoRA
        # on Qwen3.5 variants). fast_inference=False because vLLM does not yet
        # support Qwen3.5 — rollouts fall back to HF generate.
        from unsloth import FastLanguageModel
        rw_model, rw_tok = FastLanguageModel.from_pretrained(
            model_name=rewriter_id,
            max_seq_length=args.max_seq_length,
            load_in_4bit=False,
            fast_inference=False,
        )
        if rw_tok.pad_token is None:
            rw_tok.pad_token = rw_tok.eos_token
        rw_tok.padding_side = "left"
        # target_modules list matches Unsloth's canonical Qwen/Llama notebook
        # example. The shortcut string "all-linear" breaks in this PEFT version
        # (0.18.x) — it gets iterated character-wise, yielding target names
        # like 'i','a','l','n','e','r','-'. Explicit list avoids the issue.
        _unsloth_targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]
        rw_model = FastLanguageModel.get_peft_model(
            rw_model,
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=_unsloth_targets,
            lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print(f"[{time.strftime('%H:%M:%S')}] rewriter loaded via Unsloth bf16 LoRA ({rewriter_id}), "
              f"r={args.lora_r} alpha={args.lora_alpha} targets={_unsloth_targets}", flush=True)
    else:
        rw_tok = AutoTokenizer.from_pretrained(rewriter_id, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
                                                token=os.environ.get("HF_TOKEN"))
        if rw_tok.pad_token is None:
            rw_tok.pad_token = rw_tok.eos_token
        rw_tok.padding_side = "left"
    if not args.use_unsloth and args.use_qlora:
        # 4-bit NF4 + double-quant for the 14B base so it fits on one 80GB A100
        # alongside the ~22GB vLLM judge. LoRA adapters (q/k/v/o + gate/up/down)
        # are the only trainable params; everything else is frozen 4-bit. Pattern
        # validated in scripts/preflight_rewriter.py (smoke v6).
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        rw_model = AutoModelForCausalLM.from_pretrained(
            rewriter_id, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
            quantization_config=bnb, device_map={"": 0},
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"[{time.strftime('%H:%M:%S')}] rewriter loaded 4-bit QLoRA ({rewriter_id})", flush=True)
    elif not args.use_unsloth:
        # attn_implementation="sdpa" forces PyTorch's scaled-dot-product path, which
        # computes softmax in fp32 on A100 and is numerically stable in bf16.
        # The alternative ("eager") and xformers both produced NaN gradients on
        # Qwen3-8B in the 2026-04-22 smoke — flash-attn would be fastest but has
        # no torch 2.10+cu128+cp311 wheel and source build fails in this env.
        rw_model = AutoModelForCausalLM.from_pretrained(
            rewriter_id, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
            dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa",
            token=os.environ.get("HF_TOKEN"),
        )

    @torch.no_grad()
    def generate_eval(model_, tok_, eval_rows_, batch=8, max_new=260):
        model_.eval()
        prompts_chat = [make_rewrite_prompt(r[1], r[2], int(r[3])) for r in eval_rows_]
        # enable_thinking=False makes the template pre-fill an empty think
        # block so the model skips reasoning; pairs with /no_think in the user
        # message (defense in depth against Qwen3 thinking-mode defaults).
        _tpl_kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            prompt_strs = [tok_.apply_chat_template(p, enable_thinking=False, **_tpl_kwargs) for p in prompts_chat]
        except TypeError:
            prompt_strs = [tok_.apply_chat_template(p, **_tpl_kwargs) for p in prompts_chat]
        outs = []
        for i in range(0, len(prompt_strs), batch):
            b = prompt_strs[i:i + batch]
            enc = tok_(b, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
            g = model_.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                pad_token_id=tok_.pad_token_id or tok_.eos_token_id)
            gen = g[:, enc["input_ids"].shape[1]:]
            outs.extend(tok_.batch_decode(gen, skip_special_tokens=True))
        return [strip_think_block(o) for o in outs]

    props_eval = [r[1] for r in eval_rows]
    pre_eval_json = pilot_dir / "pre_eval.json"
    if args.skip_pre_eval and pre_eval_json.exists():
        _cached = json.loads(pre_eval_json.read_text())
        pre_rewrites = _cached["pre_rewrites"]
        pre_scores = _cached["pre_scores"]
        print(f"[{time.strftime('%H:%M:%S')}] --skip-pre-eval: loaded cached pre-eval "
              f"from {pre_eval_json}", flush=True)
    elif args.skip_pre_eval:
        # Refuse to skip silently — the use case is resume after a STOP, and
        # at that point the original run should have persisted pre_eval.json.
        # If no cache exists, it means pre-eval was never run; running post-eval
        # alone produces pre/post deltas that are silently None, which would
        # be easy to misread as "model got worse / same".
        raise SystemExit(
            f"--skip-pre-eval requires a cached {pre_eval_json}; run the first "
            f"launch without --skip-pre-eval so it persists, then use "
            f"--skip-pre-eval on subsequent resumes."
        )
    else:
        pre_rewrites = generate_eval(rw_model, rw_tok, eval_rows)
        pre_scores = {j.wandb_name: j.score(props_eval, pre_rewrites) for j in judges}
        pre_eval_json.write_text(json.dumps({
            "pre_rewrites": pre_rewrites,
            "pre_scores": pre_scores,
            "eval_document_ids": [r[0] for r in eval_rows],
        }, indent=2))
        print(f"[{time.strftime('%H:%M:%S')}] pre-eval cached to {pre_eval_json} "
              f"(reusable via --skip-pre-eval on resume)", flush=True)
    if pre_scores is not None:
        for jn, s in pre_scores.items():
            print(f"  pre  {jn} mean={sum(s)/len(s):.2f}", flush=True)

    # Keep rw_model alive — it's passed into GRPOTrainer below so the trainer uses
    # the bf16 weights instead of reloading in FP32.
    gc.collect()
    torch.cuda.empty_cache()

    # Training
    print(f"[{time.strftime('%H:%M:%S')}] building GRPOTrainer...", flush=True)
    # Pass the already-bf16-loaded rw_model instead of the string REWRITER.
    # If GRPOTrainer receives the string it reloads in FP32, doubling save size
    # (6 GB vs 3 GB) and forcing a 2-shard save (5 GB + 1.2 GB). A bf16 model
    # fits in a single <5 GB shard, avoiding partial-write failures on tight disks.
    trainer_kwargs = dict(
        model=rw_model,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=ds,
        processing_class=rw_tok,
    )
    if args.use_unsloth:
        # Adapters already applied via FastLanguageModel.get_peft_model — do NOT
        # pass peft_config to GRPOTrainer or the trainer tries to re-apply LoRA
        # on top of an already-PEFT-wrapped model.
        print(f"[{time.strftime('%H:%M:%S')}] Unsloth adapters already applied — skipping trainer peft_config", flush=True)
    else:
        # Both --use-qlora and vanilla bf16 paths attach LoRA here. QLoRA loaded a
        # 4-bit base; bf16 loaded a full-precision base. In both cases we want
        # adapter-only training via peft_config.
        from peft import LoraConfig
        trainer_kwargs["peft_config"] = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        _mode = "QLoRA 4-bit" if args.use_qlora else "bf16 LoRA"
        print(f"[{time.strftime('%H:%M:%S')}] {_mode} r={args.lora_r} alpha={args.lora_alpha} on q/k/v/o + gate/up/down", flush=True)
    trainer = GRPOTrainer(**trainer_kwargs)
    # Graceful-stop: at each step boundary, if the STOP marker exists, the
    # trainer saves a checkpoint and exits cleanly instead of running to
    # --max-steps. Pair with --save-steps so the stop-save produces a
    # resumable checkpoint.
    trainer.add_callback(build_trainer_callback(str(stop_file)))
    t_train_start = time.time()
    resume_msg = f" (resume from {args.resume_from})" if args.resume_from else ""
    print(f"[{time.strftime('%H:%M:%S')}] starting GRPO training "
          f"({args.max_steps} steps{resume_msg})", flush=True)
    trainer.train(resume_from_checkpoint=args.resume_from)
    train_elapsed = time.time() - t_train_start
    stopped_early = trainer.state.global_step < args.max_steps
    if stopped_early:
        print(f"[{time.strftime('%H:%M:%S')}] training STOPPED EARLY at step "
              f"{trainer.state.global_step}/{args.max_steps} "
              f"(elapsed {train_elapsed/60:.1f} min) — checkpoint written by STOP callback", flush=True)
        # Short-circuit: post-eval + held-out scoring would block both GPUs for
        # ~20 more min, defeating the point of a graceful stop. The STOP-callback
        # already wrote a resumable checkpoint under ckpt_<suffix>/. Tear down
        # and exit. Resume with:
        #   python run_pilot_len_pen.py ... --resume-from <ckpt> --skip-pre-eval
        print(f"[{time.strftime('%H:%M:%S')}] skipping post-eval + held-out on early stop. "
              f"Resume with --resume-from {cfg.output_dir}/checkpoint-{trainer.state.global_step} "
              f"--skip-pre-eval", flush=True)
        try:
            wandb.log({"runtime/stopped_early_step": trainer.state.global_step,
                       "runtime/train_min": train_elapsed / 60})
            wandb.finish()
        except Exception as e:
            print(f"  warn: wandb.finish on early-stop failed: {e!r}", flush=True)
        if judge_server_proc is not None:
            try:
                JudgeHTTP.request_shutdown(judge_endpoint, timeout=10.0)
            except Exception:
                try:
                    judge_server_proc.terminate()
                except Exception:
                    pass
            try:
                judge_server_proc.wait(timeout=60)
            except Exception:
                try:
                    judge_server_proc.kill()
                except Exception:
                    pass
        return
    print(f"[{time.strftime('%H:%M:%S')}] training done in {train_elapsed/60:.1f} min", flush=True)

    # Save model FIRST, before any eval or vLLM teardown.
    # Deep debug: save_model reliably crashes if it runs after vLLM judges tear down
    # (NCCL/distributed state corrupts at EngineCore subprocess shutdown). Running save
    # immediately after training matches the fold1/2/3 pattern that worked reliably.
    # Periodic save_steps=50 in GRPOConfig is the belt-and-suspenders backup.
    if args.save_final:
        try:
            trainer.save_model(args.save_final)
            rw_tok.save_pretrained(args.save_final)
            print(f"[{time.strftime('%H:%M:%S')}] saved final model to {args.save_final}", flush=True)
        except Exception as e:
            print(f"save_model failed (periodic ckpts remain in ckpt_{args.name_suffix}/): {e!r}", flush=True)

    trained_model = trainer.model
    post_rewrites = generate_eval(trained_model, rw_tok, eval_rows)
    post_scores = {j.wandb_name: j.score(props_eval, post_rewrites) for j in judges}
    for jn, s in post_scores.items():
        print(f"  post {jn} mean={sum(s)/len(s):.2f}", flush=True)

    # Save artefacts
    summary = {}
    for jn in pre_scores:
        pm = sum(pre_scores[jn]) / len(pre_scores[jn])
        qm = sum(post_scores[jn]) / len(post_scores[jn])
        summary[jn] = {"pre_mean": pm, "post_mean": qm, "delta": qm - pm}
        wandb.log({
            f"eval/{jn}_pre_mean": pm,
            f"eval/{jn}_post_mean": qm,
            f"eval/{jn}_delta": qm - pm,
        })
        print(f"  {jn}: pre={pm:.2f}  post={qm:.2f}  delta={qm-pm:+.2f}", flush=True)

    # Partial artefact write — capture in-panel data before held-out / save crashes.
    pilot_dir = OUT_DIR / f"pilot_{args.name_suffix}"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    (pilot_dir / "eval_summary_partial.json").write_text(json.dumps({
        "summary": summary,
        "pre_rewrites": pre_rewrites,
        "post_rewrites": post_rewrites,
        "eval_document_ids": [r[0] for r in eval_rows],
        "eval_originals": [r[2] for r in eval_rows],
        "eval_propositions": [r[1] for r in eval_rows],
        "eval_word_counts": [int(r[3]) for r in eval_rows],
    }, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] wrote partial eval_summary", flush=True)

    # Held-out judge eval — unload the in-panel judges server-side first to free
    # the GPU-0 VRAM before loading the held-out engine. Rewriter VRAM on GPU 1
    # is freed via the local deletes + empty_cache below.
    heldout_pre_scores = heldout_post_scores = None
    if heldout_slug is not None:
        print(f"[{time.strftime('%H:%M:%S')}] unloading in-panel judges server-side...",
              flush=True)
        for j in judges:
            try:
                j.unload()
            except Exception as e:
                print(f"  warn: unload({j.name}) failed: {e!r}", flush=True)
        del judges
        del trainer
        del trained_model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[{time.strftime('%H:%M:%S')}] loading held-out {heldout_slug} "
              f"({heldout_hf_id}) via HTTP...", flush=True)
        held = JudgeHTTP(name=heldout_slug, rubric=args.criterion,
                          endpoint=judge_endpoint)
        heldout_pre_scores = held.score(props_eval, pre_rewrites)
        heldout_post_scores = held.score(props_eval, post_rewrites)
        hm_pre = sum(heldout_pre_scores) / len(heldout_pre_scores)
        hm_post = sum(heldout_post_scores) / len(heldout_post_scores)
        summary[heldout_wandb_name] = {
            "pre_mean": hm_pre, "post_mean": hm_post,
            "delta": hm_post - hm_pre, "role": "held_out",
        }
        wandb.log({
            f"eval/{heldout_wandb_name}_pre_mean": hm_pre,
            f"eval/{heldout_wandb_name}_post_mean": hm_post,
            f"eval/{heldout_wandb_name}_delta": hm_post - hm_pre,
        })
        print(f"  [HELD-OUT] {heldout_slug}: pre={hm_pre:.2f}  "
              f"post={hm_post:.2f}  delta={hm_post-hm_pre:+.2f}", flush=True)

    (OUT_DIR / f"pilot_{args.name_suffix}" / "eval_summary.json").write_text(json.dumps({
        "summary": summary,
        "pre_rewrites": pre_rewrites,
        "post_rewrites": post_rewrites,
        "heldout_pre_scores": heldout_pre_scores,
        "heldout_post_scores": heldout_post_scores,
        "eval_document_ids": [r[0] for r in eval_rows],
        "eval_originals": [r[2] for r in eval_rows],
        "eval_propositions": [r[1] for r in eval_rows],
        "eval_word_counts": [int(r[3]) for r in eval_rows],
        "timing": {
            "total_elapsed_s": time.time() - t_start,
            "train_elapsed_s": train_elapsed,
            "total_scoring_s_sum": step_counter["t_total_score"],
            "n_reward_calls": step_counter["n"],
        },
    }, indent=2))

    total_elapsed = time.time() - t_start
    print(f"[{time.strftime('%H:%M:%S')}] DONE total={total_elapsed/60:.1f} min  "
          f"train={train_elapsed/60:.1f} min  n_reward_calls={step_counter['n']}", flush=True)
    wandb.log({"runtime/total_min": total_elapsed / 60,
               "runtime/train_min": train_elapsed / 60,
               "runtime/total_scoring_min": step_counter["t_total_score"] / 60})
    wandb.finish()

    # Judge server teardown — only the spawned variant; an externally-provided
    # endpoint is left running (caller owns its lifecycle).
    if judge_server_proc is not None:
        print(f"[{time.strftime('%H:%M:%S')}] shutting down judge server "
              f"(pid={judge_server_proc.pid})...", flush=True)
        try:
            JudgeHTTP.request_shutdown(judge_endpoint, timeout=10.0)
        except Exception as e:
            print(f"  warn: /shutdown failed: {e!r}; falling back to SIGTERM",
                  flush=True)
            try:
                judge_server_proc.terminate()
            except Exception:
                pass
        try:
            judge_server_proc.wait(timeout=60)
        except Exception:
            try:
                judge_server_proc.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()
