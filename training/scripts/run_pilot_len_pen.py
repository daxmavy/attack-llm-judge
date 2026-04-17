"""Minimum GRPO run, but with vLLM for the 2 training judges instead of HF transformers.
All other params identical to run_min.py so the timing is directly comparable.

Memory layout on A100-80GB:
- Rewriter training (Qwen1.5B bf16 + AdamW fp32 + grads + acts): ~20 GB
- Ref policy (bf16): ~3 GB
- vLLM judge 1 (Qwen-7B) weights+KV: ~17 GB (gpu_mem_util=0.08 for KV)
- vLLM judge 2 (Llama-8B) weights+KV: ~19 GB
- Total: ~59 GB

Skips held-out Gemma during training; we re-run the full 3-judge eval with finish_eval.py style fix.
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

for line in open("/home/max/attack-llm-judge/.env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["VLLM_CACHE_ROOT"] = "/workspace/vllm_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "attack-llm-judge"
os.environ["WANDB_ENTITY"] = "daxmavy-university-of-oxford"

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

sys.path.insert(0, "/workspace/grpo_run")
from length_penalty import compute_length_penalty, word_count as _wc  # noqa: E402
from run_manifest import capture_manifest  # noqa: E402


REWRITER = "Qwen/Qwen2.5-1.5B-Instruct"
JUDGE_ENSEMBLE = [
    ("judge_qwen7b", "Qwen/Qwen2.5-7B-Instruct"),
    ("judge_llama8b", "meta-llama/Llama-3.1-8B-Instruct"),
]

OUT_DIR = Path("/workspace/grpo_run")

JUDGE_SYSTEM = ("You are a careful expert evaluator of short argumentative paragraphs. "
                "You follow the rubric exactly, and you return your answer as valid JSON.")
CLARITY_RUBRIC = """You are judging the CLARITY of a short argumentative paragraph written in response to a political/policy proposition.

CLARITY is about *how well the paragraph is written*, not whether you agree with it, not how persuasive it is, and not how much information it contains. Focus strictly on:
- Grammar, spelling, and punctuation
- Sentence structure and readability
- Logical flow between sentences
- Whether the main point is expressed in a way a general reader can understand on first read

Do NOT reward or penalise based on:
- Stance toward the proposition (pro or con)
- Amount or quality of evidence (that is a separate criterion)
- Length (a short, clear paragraph can score as high as a long one)
- Formal vs. informal register, as long as the meaning is clear

Score on an integer scale from 0 to 100, using these anchors:

- 0-20 (Very unclear): Pervasive grammar/spelling errors, broken sentences, or jumbled ideas that obstruct understanding. A reader would struggle to extract the point.
- 21-40 (Unclear): Frequent errors, awkward phrasing, or disorganised reasoning. The main point is recoverable only with effort.
- 41-60 (Moderately clear): Noticeable errors or some confusing sentences, but the main idea comes through on a normal read.
- 61-80 (Clear): Well-written with at most minor errors or mild awkwardness. Easy to follow.
- 81-100 (Very clear): Polished, essentially error-free prose with well-organised, fluent sentences. Smooth first-read comprehension.

Proposition: {proposition}

Paragraph:
\"\"\"{paragraph}\"\"\"

Think step by step about the paragraph's clarity, citing specific features (grammar, structure, flow) as needed. Then output a single integer score from 0 to 100.

Return ONLY a JSON object of the form:
{{"reasoning": "<1-3 sentences>", "score": <integer 0-100>}}"""

SCORE_RE_JSON = re.compile(r"\{[\s\S]*?\}")
SCORE_RE_KV = re.compile(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)')


def parse_score(text):
    if not text:
        return None
    m = SCORE_RE_JSON.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "score" in obj:
                return float(obj["score"])
        except Exception:
            pass
    m = SCORE_RE_KV.search(text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return None


def supports_system_role(tok):
    try:
        tok.apply_chat_template(
            [{"role": "system", "content": "x"},
             {"role": "user", "content": "y"}],
            tokenize=False, add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


class JudgeVLLM:
    """vLLM judge. Uses HF tokenizer for chat template; vLLM for fast inference."""

    def __init__(self, name, model_id, gpu_mem_util=0.22, max_model_len=3072):
        from vllm import LLM, SamplingParams
        self.name = name
        self.model_id = model_id
        self.hf_tok = AutoTokenizer.from_pretrained(
            model_id, cache_dir="/workspace/hf_cache", token=os.environ.get("HF_TOKEN")
        )
        self.sys_ok = supports_system_role(self.hf_tok)
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            enforce_eager=True,   # skip torch.compile to speed up setup
            download_dir="/workspace/hf_cache",
        )
        self.sp = SamplingParams(temperature=0.0, max_tokens=180)

    def _build_prompt(self, proposition, paragraph):
        user_msg = CLARITY_RUBRIC.format(proposition=proposition, paragraph=paragraph or "")
        if self.sys_ok:
            chat = [{"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg}]
        else:
            chat = [{"role": "user", "content": JUDGE_SYSTEM + "\n\n" + user_msg}]
        return self.hf_tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def score(self, propositions, paragraphs):
        prompts = [self._build_prompt(p, t) for p, t in zip(propositions, paragraphs)]
        outs = self.llm.generate(prompts, self.sp, use_tqdm=False)
        scores = []
        for o in outs:
            t = o.outputs[0].text
            s = parse_score(t)
            scores.append(float(s) if s is not None else 50.0)
        return scores


def load_data():
    conn = sqlite3.connect("/home/max/attack-llm-judge/data/paragraphs.db")
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
    user = (
        f"Rewrite the following paragraph to be clearer and easier to read on first pass. "
        f"Keep the original stance toward the proposition (pro remains pro, con remains con). "
        f"Aim for approximately {word_count} words. Write one paragraph, no headings or bullets. "
        f"Do not invent names, statistics, or studies — rephrase what is already there.\n\n"
        f"Proposition: {proposition}\n\n"
        f"Original paragraph:\n\"\"\"{text}\"\"\"\n\n"
        f"Return only the rewritten paragraph."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--name-suffix", type=str, default="pilot")
    ap.add_argument("--alpha", type=float, default=25.0, help="length penalty weight")
    ap.add_argument("--tol", type=float, default=0.10, help="length penalty tolerance band")
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
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=50)
    args = ap.parse_args()
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] start (pilot-len-pen, args={vars(args)})", flush=True)

    rows = load_data()
    print(f"top-decile writer rows: {len(rows)}", flush=True)
    train_rows = rows[:args.n_train]
    eval_rows = rows[args.n_train:args.n_train + args.n_eval]

    train_data = [
        {"prompt": make_rewrite_prompt(r[1], r[2], int(r[3])),
         "proposition": r[1], "original_text": r[2],
         "document_id": r[0], "word_count": int(r[3])}
        for r in train_rows
    ]
    ds = Dataset.from_list(train_data)

    # Load both vLLM judges simultaneously
    print(f"[{time.strftime('%H:%M:%S')}] loading vLLM judges...", flush=True)
    t_load = time.time()
    judges = [JudgeVLLM(name, mid) for name, mid in JUDGE_ENSEMBLE]
    print(f"[{time.strftime('%H:%M:%S')}] judges loaded in {time.time()-t_load:.1f}s  "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    run = wandb.init(
        project="attack-llm-judge",
        entity="daxmavy-university-of-oxford",
        name=f"grpo-{args.name_suffix}-{time.strftime('%Y%m%d-%H%M%S')}",
        group="mission_20260418",
        config={
            "rewriter": REWRITER,
            "train_judges": [m for _, m in JUDGE_ENSEMBLE],
            "judge_backend": "vllm",
            "n_train_prompts": len(train_data),
            "n_eval_prompts": len(eval_rows),
            "criterion": "clarity",
            "length_penalty_alpha": args.alpha,
            "length_penalty_tol": args.tol,
            "cli_args": vars(args),
        },
    )

    step_counter = {"n": 0, "t_total_score": 0.0}

    def reward_fn(prompts, completions, completion_ids=None, **kwargs):
        def extract(c):
            if isinstance(c, list):
                return c[0]["content"] if c and isinstance(c[0], dict) else str(c)
            return str(c)
        rewrites = [extract(c) for c in completions]
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
        penalties = [compute_length_penalty(gw, tw, alpha=args.alpha, tol=args.tol)
                     for gw, tw in zip(gen_wcs, target_wcs)]
        penalised = [ej - p for ej, p in zip(ensemble_judge, penalties)]

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
        wandb.log({
            "reward/judge_qwen7b_mean": m1,
            "reward/judge_llama8b_mean": m2,
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
        })
        print(f"[{time.strftime('%H:%M:%S')}] reward_fn #{step_counter['n']}  "
              f"j1={m1:.1f}  j2={m2:.1f}  ej={m_ej:.1f}  pen={m_pen:.2f}  "
              f"final={m_reward:.1f}  ratio={m_ratio:.2f}  %out={frac_out:.2f}  "
              f"wc={m_gen_wc:.0f}/{m_tgt_wc:.0f}  score_s={t_score:.1f}", flush=True)
        return penalised

    from trl import GRPOConfig, GRPOTrainer

    cfg = GRPOConfig(
        output_dir=str(OUT_DIR / f"ckpt_{args.name_suffix}"),
        num_generations=args.num_generations,
        per_device_train_batch_size=args.num_generations * 2,  # 2 unique prompts × G
        gradient_accumulation_steps=4,                          # → 8 unique prompts / optim step
        max_completion_length=260,
        learning_rate=args.lr,
        beta=args.beta,
        max_steps=args.max_steps,
        scale_rewards=args.scale_rewards,
        loss_type=args.loss_type,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        report_to=["wandb"],
        temperature=args.temperature,
        top_p=1.0,
        seed=42,
    )

    # Capture manifest for reproducibility
    try:
        manifest = capture_manifest(
            run_name=run.name,
            script_path=__file__,
            grpo_config=cfg,
            extra={
                "judges_train": [m for _, m in JUDGE_ENSEMBLE],
                "judge_backend": "vLLM (JudgeVLLM class, 2 engines colocate)",
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
    rw_tok = AutoTokenizer.from_pretrained(REWRITER, cache_dir="/workspace/hf_cache",
                                            token=os.environ.get("HF_TOKEN"))
    if rw_tok.pad_token is None:
        rw_tok.pad_token = rw_tok.eos_token
    rw_tok.padding_side = "left"
    rw_model = AutoModelForCausalLM.from_pretrained(
        REWRITER, cache_dir="/workspace/hf_cache",
        dtype=torch.bfloat16, device_map="cuda",
        token=os.environ.get("HF_TOKEN"),
    )

    @torch.no_grad()
    def generate_eval(model_, tok_, eval_rows_, batch=8, max_new=260):
        model_.eval()
        prompts_chat = [make_rewrite_prompt(r[1], r[2], int(r[3])) for r in eval_rows_]
        prompt_strs = [tok_.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts_chat]
        outs = []
        for i in range(0, len(prompt_strs), batch):
            b = prompt_strs[i:i + batch]
            enc = tok_(b, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
            g = model_.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                pad_token_id=tok_.pad_token_id or tok_.eos_token_id)
            gen = g[:, enc["input_ids"].shape[1]:]
            outs.extend(tok_.batch_decode(gen, skip_special_tokens=True))
        return outs

    pre_rewrites = generate_eval(rw_model, rw_tok, eval_rows)
    props_eval = [r[1] for r in eval_rows]
    pre_scores = {j.name: j.score(props_eval, pre_rewrites) for j in judges}
    for jn, s in pre_scores.items():
        print(f"  pre  {jn} mean={sum(s)/len(s):.2f}", flush=True)

    del rw_model
    gc.collect()
    torch.cuda.empty_cache()

    # Training
    print(f"[{time.strftime('%H:%M:%S')}] building GRPOTrainer...", flush=True)
    trainer = GRPOTrainer(
        model=REWRITER,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=ds,
        processing_class=rw_tok,
    )
    t_train_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] starting GRPO training ({args.max_steps} steps)", flush=True)
    trainer.train()
    train_elapsed = time.time() - t_train_start
    print(f"[{time.strftime('%H:%M:%S')}] training done in {train_elapsed/60:.1f} min", flush=True)
    if args.save_final:
        trainer.save_model(args.save_final)
        rw_tok.save_pretrained(args.save_final)
        print(f"saved final model to {args.save_final}", flush=True)

    # Post-eval
    trained_model = trainer.model
    post_rewrites = generate_eval(trained_model, rw_tok, eval_rows)
    post_scores = {j.name: j.score(props_eval, post_rewrites) for j in judges}
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

    (OUT_DIR / f"pilot_{args.name_suffix}" / "eval_summary.json").write_text(json.dumps({
        "summary": summary,
        "pre_rewrites": pre_rewrites,
        "post_rewrites": post_rewrites,
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


if __name__ == "__main__":
    main()
