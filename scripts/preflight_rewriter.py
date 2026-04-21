"""Pre-flight compatibility check for candidate rewriter base models.

For each model:
  1. Load in vLLM (no library version changes).
  2. Apply chat template + generate a single short paragraph.
  3. Attempt a minimal GRPOTrainer step with a live judge.
     (Keep vLLM/TRL versions fixed; report failures honestly.)

Outputs a structured JSON report plus a human-readable summary.
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

ENV = "/home/max/attack-llm-judge/.env"
if os.path.exists(ENV):
    for line in open(ENV):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("VLLM_CACHE_ROOT", "/workspace/vllm_cache")

sys.path.insert(0, "/workspace/grpo_run")
sys.path.insert(0, "/home/max/attack-llm-judge")


CANDIDATES = [
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "google/gemma-3-1b-it",
]


def check_vllm_generate(model_id: str):
    """Stage 1+2: load model in vLLM and generate a short paragraph."""
    result = {"stage": "vllm_generate", "ok": False, "error": None, "details": {}}
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            model_id, cache_dir="/workspace/hf_cache",
            token=os.environ.get("HF_TOKEN"),
        )
        result["details"]["tokenizer_loaded"] = True
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        t0 = time.time()
        llm = LLM(
            model=model_id, dtype="bfloat16",
            gpu_memory_utilization=0.18,
            max_model_len=3072, enforce_eager=True,
            download_dir="/workspace/hf_cache",
        )
        result["details"]["load_time_s"] = round(time.time() - t0, 1)

        # Try chat template — mirrors how run_pilot_len_pen.py and run_mission_attacks.py build prompts
        chat = [
            {"role": "system", "content": "You are a careful writing assistant."},
            {"role": "user", "content": "Rewrite in one sentence: AI is changing industries. Return only the rewrite."},
        ]
        try:
            prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            result["details"]["system_role_supported"] = True
        except Exception as e:
            # Retry without system role (Gemma family often needs this)
            result["details"]["system_role_supported"] = False
            result["details"]["system_role_error"] = str(e)[:200]
            chat_nosys = [
                {"role": "user", "content": "You are a careful writing assistant.\n\nRewrite in one sentence: AI is changing industries. Return only the rewrite."},
            ]
            prompt = tok.apply_chat_template(chat_nosys, tokenize=False, add_generation_prompt=True)

        sp = SamplingParams(temperature=0.0, max_tokens=80)
        outs = llm.generate([prompt], sp, use_tqdm=False)
        gen = outs[0].outputs[0].text
        result["details"]["generated_text"] = gen[:400]
        result["details"]["generation_len_chars"] = len(gen)
        result["ok"] = True
        # Cleanup
        del llm
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-3000:]
    return result


def check_grpo_step(model_id: str):
    """Stage 3: single GRPOTrainer step with a live vLLM judge.

    Uses qwen95b as a single judge + Qwen2.5-1.5B-ish config to minimise moving parts.
    """
    result = {"stage": "grpo_step", "ok": False, "error": None, "details": {}}
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import GRPOConfig, GRPOTrainer
        from run_pilot_len_pen import JudgeVLLM, JUDGE_REGISTRY

        tok = AutoTokenizer.from_pretrained(model_id, cache_dir="/workspace/hf_cache",
                                             token=os.environ.get("HF_TOKEN"))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir="/workspace/hf_cache",
            dtype=torch.bfloat16, device_map="cuda",
            token=os.environ.get("HF_TOKEN"),
        )
        result["details"]["hf_load_time_s"] = round(time.time() - t0, 1)
        result["details"]["model_n_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)

        # Judge
        judge_spec = JUDGE_REGISTRY["qwen95b"]
        judge = JudgeVLLM(*judge_spec, rubric="clarity")
        result["details"]["judge_loaded"] = True

        # Minimal reward fn
        def reward_fn(prompts, completions, completion_ids=None, **kwargs):
            def extract(c):
                if isinstance(c, list):
                    return c[0]["content"] if c and isinstance(c[0], dict) else str(c)
                return str(c)
            rewrites = [extract(c) for c in completions]
            props = kwargs["proposition"]
            j = judge.score(props, rewrites)
            return list(j)

        # Tiny dataset: 4 fake examples
        fake_data = [
            {"proposition": f"Proposition {i}",
             "prompt": [{"role": "system", "content": "Rewrite."},
                         {"role": "user", "content": f"Paragraph {i}: AI is important."}]}
            for i in range(4)
        ]
        from datasets import Dataset
        ds = Dataset.from_list(fake_data)

        cfg = GRPOConfig(
            output_dir="/workspace/grpo_run/ckpt_preflight",
            max_steps=1,
            num_generations=4,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=5e-6,
            temperature=1.0,
            save_strategy="no",
            logging_steps=1,
            report_to=[],
        )

        t0 = time.time()
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=cfg,
            train_dataset=ds,
            processing_class=tok,
        )
        trainer.train()
        result["details"]["train_time_s"] = round(time.time() - t0, 1)
        result["ok"] = True

        del trainer, model, judge
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-3000:]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=CANDIDATES)
    ap.add_argument("--skip-grpo", action="store_true", help="only run vLLM generate stage")
    ap.add_argument("--out", default="/home/max/attack-llm-judge/PREFLIGHT.md")
    ap.add_argument("--json-out", default="/workspace/grpo_run/preflight_results.json")
    args = ap.parse_args()

    all_results = {}
    for m in args.models:
        print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] PREFLIGHT: {m}\n{'='*70}", flush=True)
        r = {"model": m}
        r["vllm_generate"] = check_vllm_generate(m)
        print(f"  vllm_generate ok: {r['vllm_generate']['ok']}", flush=True)
        if r["vllm_generate"]["ok"] and not args.skip_grpo:
            r["grpo_step"] = check_grpo_step(m)
            print(f"  grpo_step ok: {r['grpo_step']['ok']}", flush=True)
        all_results[m] = r

    # Write JSON for machine parsing
    Path(args.json_out).write_text(json.dumps(all_results, indent=2))
    print(f"\nJSON report: {args.json_out}", flush=True)

    # Write markdown report
    lines = [f"# Pre-flight report — {time.strftime('%Y-%m-%d %H:%M UTC')}", ""]
    for m, r in all_results.items():
        lines.append(f"## `{m}`")
        vg = r["vllm_generate"]
        gs = r.get("grpo_step", {"ok": None, "error": "skipped"})
        status = "✅ pass" if (vg["ok"] and gs.get("ok")) else ("⚠️  partial" if vg["ok"] else "❌ fail")
        lines.append(f"- **Status:** {status}")
        lines.append(f"- vLLM generate: `ok={vg['ok']}`  details={json.dumps(vg.get('details', {}), separators=(',', ':'))[:500]}")
        if vg.get("error"):
            lines.append(f"  - error: `{vg['error']}`")
        lines.append(f"- GRPO step: `ok={gs.get('ok')}`  details={json.dumps(gs.get('details', {}), separators=(',', ':'))[:500]}")
        if gs.get("error"):
            lines.append(f"  - error: `{gs['error']}`")
        if vg.get("details", {}).get("generated_text"):
            lines.append(f"- Sample generation: `{vg['details']['generated_text'][:200]}`")
        lines.append("")
    Path(args.out).write_text("\n".join(lines))
    print(f"MD report: {args.out}", flush=True)


if __name__ == "__main__":
    main()
