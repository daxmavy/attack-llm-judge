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

ENV = "/home/shil6647/attack-llm-judge/.env"
if os.path.exists(ENV):
    for line in open(ENV):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v
os.environ.setdefault("HF_HOME", "/data/shil6647/attack-llm-judge/hf_cache")
os.environ.setdefault("VLLM_CACHE_ROOT", "/data/shil6647/attack-llm-judge/vllm_cache")

sys.path.insert(0, "/data/shil6647/attack-llm-judge/grpo_run")
sys.path.insert(0, "/home/shil6647/attack-llm-judge")


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
            model_id, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
            token=os.environ.get("HF_TOKEN"),
        )
        result["details"]["tokenizer_loaded"] = True
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        t0 = time.time()
        llm = LLM(
            model=model_id, dtype="bfloat16",
            gpu_memory_utilization=0.55,
            max_model_len=3072, enforce_eager=True,
            download_dir="/data/shil6647/attack-llm-judge/hf_cache",
        )
        result["details"]["load_time_s"] = round(time.time() - t0, 1)

        # Try chat template — mirrors how run_pilot_len_pen.py and run_mission_attacks.py build prompts
        chat = [
            {"role": "system", "content": "You are a careful writing assistant."},
            {"role": "user", "content": "Rewrite in one sentence: AI is changing industries. Return only the rewrite."},
        ]
        # Qwen3 family: disable thinking via chat_template_kwargs. Harmless
        # for other models (unused kwarg dropped by non-Qwen3 templates).
        tpl_kwargs = {"enable_thinking": False}
        try:
            prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, **tpl_kwargs)
            result["details"]["system_role_supported"] = True
        except Exception as e:
            # Retry without system role (Gemma family often needs this)
            result["details"]["system_role_supported"] = False
            result["details"]["system_role_error"] = str(e)[:200]
            chat_nosys = [
                {"role": "user", "content": "You are a careful writing assistant.\n\nRewrite in one sentence: AI is changing industries. Return only the rewrite."},
            ]
            try:
                prompt = tok.apply_chat_template(chat_nosys, tokenize=False, add_generation_prompt=True, **tpl_kwargs)
            except TypeError:
                prompt = tok.apply_chat_template(chat_nosys, tokenize=False, add_generation_prompt=True)

        sp = SamplingParams(temperature=0.0, max_tokens=80)
        outs = llm.generate([prompt], sp, use_tqdm=False)
        gen = outs[0].outputs[0].text
        result["details"]["generated_text"] = gen[:400]
        result["details"]["generation_len_chars"] = len(gen)
        result["details"]["has_think_tokens"] = ("<think>" in gen) or ("</think>" in gen)
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


def check_grpo_step(model_id: str, use_qlora: bool = False, train_device: str = "cuda", max_steps: int = 1):
    """Stage 3: single GRPOTrainer step with a live vLLM judge.

    When `use_qlora`, loads the base in 4-bit (bnb nf4, double-quant) and wraps with
    a LoraConfig targeting all attention+MLP projections. TRL's GRPOTrainer accepts
    `peft_config=...` and handles reference-policy via adapter-disable (no bf16 copy).

    `train_device` lets you pin the trainer model to a specific GPU (e.g. "cuda:1")
    so the vLLM judge can own another GPU when `CUDA_VISIBLE_DEVICES="0,1"`.
    """
    result = {"stage": "grpo_step", "ok": False, "error": None, "details": {"use_qlora": use_qlora}}
    # Parse train_idx BEFORE any heavy import so we can pin
    # ACCELERATE_TORCH_DEVICE before trl/accelerate first touches PartialState
    # (which is a singleton that reads env only once at first instantiation).
    train_idx = 0
    if isinstance(train_device, str) and ":" in train_device:
        train_idx = int(train_device.split(":", 1)[1])
    elif isinstance(train_device, int):
        train_idx = train_device
    os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{train_idx}"
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import GRPOConfig, GRPOTrainer
        from run_pilot_len_pen import JudgeVLLM, JUDGE_REGISTRY

        tok = AutoTokenizer.from_pretrained(model_id, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
                                             token=os.environ.get("HF_TOKEN"))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        # Judge FIRST — vLLM spawns its EngineCore subprocess which picks visible
        # GPU 0 via CUDA_VISIBLE_DEVICES. Loading it before the trainer keeps the
        # two GPUs cleanly split (judge on physical GPU 0, trainer on GPU train_idx).
        judge_spec = JUDGE_REGISTRY["qwen95b"]
        judge = JudgeVLLM(*judge_spec, rubric="clarity")
        result["details"]["judge_loaded"] = True

        # Pin current device so torch.cuda.current_device() matches device_map
        # for the 4-bit/8-bit model load (accelerate validates placement).
        torch.cuda.set_device(train_idx)
        result["details"]["train_device_idx"] = train_idx
        result["details"]["accelerate_torch_device"] = os.environ.get("ACCELERATE_TORCH_DEVICE", "")

        t0 = time.time()
        if use_qlora:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
                quantization_config=bnb, device_map={"": train_idx},
                token=os.environ.get("HF_TOKEN"),
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
                dtype=torch.bfloat16, device_map={"": train_idx},
                token=os.environ.get("HF_TOKEN"),
            )
        result["details"]["hf_load_time_s"] = round(time.time() - t0, 1)
        result["details"]["model_n_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)

        # Belt-and-suspenders: even with ACCELERATE_TORCH_DEVICE pinned, TRL's
        # in-process generate() rollout path has been observed to pass input_ids
        # on cuda:0 to a model whose embed_tokens live on cuda:1 (see
        # EXPERIMENT_NOTES B-13). Install a root-level forward pre-hook that
        # forces every incoming tensor to the model's actual param device.
        _model_device = next(model.parameters()).device
        result["details"]["model_device"] = str(_model_device)

        def _align_inputs_to_model_device(module, args, kwargs):
            new_args = tuple(
                a.to(_model_device, non_blocking=True) if torch.is_tensor(a) and a.device != _model_device else a
                for a in args
            )
            new_kwargs = {
                k: (v.to(_model_device, non_blocking=True) if torch.is_tensor(v) and v.device != _model_device else v)
                for k, v in kwargs.items()
            }
            return new_args, new_kwargs

        model.register_forward_pre_hook(_align_inputs_to_model_device, with_kwargs=True)
        result["details"]["pre_hook_installed"] = True

        peft_config = None
        if use_qlora:
            from peft import LoraConfig
            peft_config = LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            )
            # Keep as list so the JSON report can serialize cleanly (LoraConfig
            # stores target_modules as a set after __post_init__).
            result["details"]["lora_target_modules"] = list(peft_config.target_modules)

        def reward_fn(prompts, completions, completion_ids=None, **kwargs):
            def extract(c):
                if isinstance(c, list):
                    return c[0]["content"] if c and isinstance(c[0], dict) else str(c)
                return str(c)
            rewrites = [extract(c) for c in completions]
            props = kwargs["proposition"]
            j = judge.score(props, rewrites)
            return list(j)

        fake_data = [
            {"proposition": f"Proposition {i}",
             "prompt": [{"role": "system", "content": "Rewrite."},
                         {"role": "user", "content": f"Paragraph {i}: AI is important."}]}
            for i in range(4)
        ]
        from datasets import Dataset
        ds = Dataset.from_list(fake_data)

        cfg = GRPOConfig(
            output_dir="/data/shil6647/attack-llm-judge/grpo_run/ckpt_preflight",
            max_steps=max_steps,
            num_generations=4,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=5e-6,
            temperature=1.0,
            save_strategy="no",
            logging_steps=1,
            report_to=[],
            gradient_checkpointing=use_qlora,
        )

        t0 = time.time()
        trainer_kwargs = dict(
            model=model, reward_funcs=reward_fn, args=cfg,
            train_dataset=ds, processing_class=tok,
        )
        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config
        trainer = GRPOTrainer(**trainer_kwargs)
        trainer.train()
        result["details"]["train_time_s"] = round(time.time() - t0, 1)
        if hasattr(trainer, "state") and getattr(trainer.state, "log_history", None):
            result["details"]["first_step_log"] = trainer.state.log_history[-1]
            # Capture every per-step log entry so timing probes can distinguish
            # cold-start (step 1) from steady-state (step 2+) step times.
            result["details"]["all_step_logs"] = list(trainer.state.log_history)
        result["details"]["mean_step_time_s"] = round((time.time() - t0) / max(1, max_steps), 1)
        if torch.cuda.is_available():
            result["details"]["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
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
    ap.add_argument("--use-qlora", action="store_true", help="load base in 4-bit + LoRA adapters for the GRPO stage")
    ap.add_argument("--train-device", default="cuda:1", help="device for GRPO trainer model (keeps judge on a separate GPU)")
    ap.add_argument("--max-steps", type=int, default=1, help="GRPO training steps (>=2 for a timing probe that surfaces steady-state step time)")
    ap.add_argument("--out", default="/home/shil6647/attack-llm-judge/PREFLIGHT.md")
    ap.add_argument("--json-out", default="/data/shil6647/attack-llm-judge/grpo_run/preflight_results.json")
    args = ap.parse_args()

    all_results = {}
    for m in args.models:
        print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] PREFLIGHT: {m}\n{'='*70}", flush=True)
        r = {"model": m}
        r["vllm_generate"] = check_vllm_generate(m)
        print(f"  vllm_generate ok: {r['vllm_generate']['ok']}", flush=True)
        if r["vllm_generate"]["ok"] and not args.skip_grpo:
            r["grpo_step"] = check_grpo_step(m, use_qlora=args.use_qlora, train_device=args.train_device, max_steps=args.max_steps)
            print(f"  grpo_step ok: {r['grpo_step']['ok']}", flush=True)
            if not r["grpo_step"]["ok"]:
                # Surface the caught exception immediately — otherwise the main loop's
                # JSON/MD writes can fail and hide it from the log entirely.
                print(f"  grpo_step error: {r['grpo_step'].get('error')}", flush=True)
                tb = r["grpo_step"].get("traceback")
                if tb:
                    print(f"  grpo_step traceback:\n{tb}", flush=True)
        if not r["vllm_generate"]["ok"]:
            print(f"  vllm_generate error: {r['vllm_generate'].get('error')}", flush=True)
            tb = r["vllm_generate"].get("traceback")
            if tb:
                print(f"  vllm_generate traceback:\n{tb}", flush=True)
        all_results[m] = r

    # Write markdown report FIRST — a JSON encoder blowup on some nested object
    # shouldn't take the whole report down with it.
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

    # Now the JSON dump. `default=str` is a safety net for anything that slipped
    # through the explicit serialization cleanup (sets, dtypes, tensors, etc.).
    Path(args.json_out).write_text(json.dumps(all_results, indent=2, default=str))
    print(f"JSON report: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
