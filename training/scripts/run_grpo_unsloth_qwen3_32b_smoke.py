"""Smoke test: Unsloth + LoRA + GRPO on Qwen3-32B (pure dense LM).

Originally targeted Qwen3.5-27B, but that model bundles a siglip vision tower
which Unsloth's fast_inference path refuses ("Found architectures: siglip,
qwen3_5"). Switched to Qwen3-32B (pure Qwen3ForCausalLM, no vision tower) —
same size class, better supported by Unsloth's vLLM-colocate fast_inference.

GRPO config ported from Qwen3_5_(4B)_Vision_GRPO.ipynb (GSPO-style sampling).

Architecture:
- GPU 0: Qwen3-32B 4-bit QLoRA + Unsloth fast_inference (in-process vLLM for
  rollouts) + LoRA r=16 + TRL GRPOTrainer.
- GPU 1: gemma-2-9b-it judge as a vLLM OpenAI-compatible server (subprocess).

Purpose: verify the full GRPO pipeline runs on Qwen3-32B with step time in the
10-60s range (vs the 191s/step we saw with HF generate() on Qwen3.5-27B).

Run:
    HF_HOME=/data/shil6647/attack-llm-judge/hf_cache \
    python3 training/scripts/run_grpo_unsloth_qwen3_32b_smoke.py
"""
from __future__ import annotations

import atexit
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/shil6647/attack-llm-judge/hf_cache")
os.environ.setdefault("VLLM_CACHE_ROOT", "/data/shil6647/attack-llm-judge/vllm_cache")
os.environ.setdefault("TMPDIR", "/data/shil6647/attack-llm-judge/tmp")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

env_path = Path("/home/shil6647/attack-llm-judge/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

from unsloth import FastLanguageModel  # must precede torch / trl imports

import torch
from datasets import Dataset
from openai import OpenAI


# Switched from unsloth/Qwen3.5-27B (VL hybrid, rejected by Unsloth fast_inference
# because of the siglip vision tower) to unsloth/Qwen3-32B pure Qwen3ForCausalLM.
# Unsloth's `-unsloth-bnb-4bit` prequantized variant uses Unsloth's mixed-precision
# scheme, designed for the fast_inference vLLM-colocate path.
REWRITER_MODEL = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
JUDGE_MODEL = "google/gemma-2-9b-it"
JUDGE_GPU = "1"
JUDGE_HOST = "127.0.0.1"
JUDGE_PORT = 8910
JUDGE_URL = f"http://{JUDGE_HOST}:{JUDGE_PORT}/v1"
DATASET_JSON = "/home/shil6647/attack-llm-judge/data/controversial_40_3fold.json"
SAVE_DIR = "/data/shil6647/attack-llm-judge/smoke_test/qwen3_32b_lora"
OUT_DIR = "/data/shil6647/attack-llm-judge/smoke_test/qwen3_32b_ckpt"
JUDGE_LOG = "/data/shil6647/attack-llm-judge/tmp/smoke_qwen3_32b_judge_server.log"

CLARITY_RUBRIC = """You are judging the CLARITY of a short argumentative paragraph written in response to a political/policy proposition.

CLARITY is about *how well the paragraph is written*, not whether you agree with it, not how persuasive it is, and not how much information it contains. Focus strictly on:
- Grammar, spelling, and punctuation
- Sentence structure and readability
- Logical flow between sentences
- Whether the main point is expressed in a way a general reader can understand on first read

Score on an integer scale from 0 to 100.

Proposition: {proposition}

Paragraph:
\"\"\"{paragraph}\"\"\"

Return ONLY a JSON object of the form:
{{"reasoning": "<1-3 sentences>", "score": <integer 0-100>}}"""


def parse_score(text: str) -> float:
    if not text:
        return 50.0
    m = re.search(r'"score"\s*:\s*(-?\d+)', text)
    if m:
        return float(m.group(1))
    m = re.search(r'\bscore[^0-9\-]{0,10}(-?\d+)', text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 50.0


def start_judge_server(wait_timeout_s: int = 600) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = JUDGE_GPU
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["HF_HOME"] = "/data/resource/huggingface"
    env["VLLM_CACHE_ROOT"] = os.environ.get("VLLM_CACHE_ROOT",
                                            "/data/shil6647/attack-llm-judge/vllm_cache")
    conda_lib = str(Path(sys.executable).resolve().parent.parent / "lib")
    env["LD_LIBRARY_PATH"] = conda_lib + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", JUDGE_MODEL,
        "--host", JUDGE_HOST,
        "--port", str(JUDGE_PORT),
        "--gpu-memory-utilization", "0.30",
        "--max-model-len", "2048",
        "--dtype", "bfloat16",
        "--enforce-eager",
        "--download-dir", "/data/resource/huggingface/hub",
    ]
    print(f"[{time.strftime('%H:%M:%S')}] launching judge server on GPU {JUDGE_GPU}  "
          f"(log: {JUDGE_LOG})", flush=True)
    Path(JUDGE_LOG).parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(JUDGE_LOG, "wb")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
                             start_new_session=True)

    def _cleanup():
        if proc.poll() is None:
            print(f"[{time.strftime('%H:%M:%S')}] terminating judge server (pid={proc.pid})",
                  flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
    atexit.register(_cleanup)

    deadline = time.time() + wait_timeout_s
    ping = f"http://{JUDGE_HOST}:{JUDGE_PORT}/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"judge server exited early with code {proc.returncode}; "
                               f"see {JUDGE_LOG}")
        try:
            with urllib.request.urlopen(ping, timeout=3) as r:
                if r.status == 200:
                    print(f"[{time.strftime('%H:%M:%S')}] judge server ready (pid={proc.pid})",
                          flush=True)
                    return proc
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    raise RuntimeError(f"judge server did not become ready within {wait_timeout_s}s; "
                       f"see {JUDGE_LOG}")


def build_rewrite_messages(proposition: str, text: str, word_count: int):
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
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] starting Qwen3-32B GRPO smoke", flush=True)

    start_judge_server()
    judge_client = OpenAI(base_url=JUDGE_URL, api_key="EMPTY")

    print(f"[{time.strftime('%H:%M:%S')}] loading rewriter {REWRITER_MODEL}", flush=True)
    t_load = time.time()
    # 4-bit prequantized (Unsloth mixed-precision bnb, ~18 GB) + fast_inference
    # spins up Unsloth's in-process vLLM engine on GPU 0. gpu_memory_utilization=0.4
    # caps vLLM's VRAM at ~32 GB (weights + KV cache); training activations +
    # LoRA grads + optimizer state share the remaining ~48 GB on the A100 80GB.
    # Rollouts flow through vLLM instead of HF generate(), cutting step time
    # ~5-20x per the Qwen3 GRPO notebook.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=REWRITER_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        fast_inference=True,
        max_lora_rank=16,
        gpu_memory_utilization=0.4,
    )
    print(f"[{time.strftime('%H:%M:%S')}] rewriter loaded in {time.time() - t_load:.1f}s  "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # LoRA per Qwen3 GRPO nb's target list (attention + MLP) with
    # rank bumped from 8 (SFT nb) to 16 (GRPO vision nb) since RL needs more
    # capacity to move the policy than supervised fine-tuning.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        # Qwen3 dense uses q_proj/k_proj/v_proj/o_proj for attention and
        # gate_proj/up_proj/down_proj for MLP. out_proj was Qwen3.5-VL-specific
        # (vision-tower output projection) — dropped for pure-LM Qwen3-32B.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Diagnostic: Unsloth's trainer-boot banner reports "Trainable parameters
    # = 0 of 27B" which is cosmetic if get_peft_model really attached LoRA.
    # Print the ground truth so we can tell a real misconfig from a bad count.
    _n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _n_total = sum(p.numel() for p in model.parameters())
    _lora_names = [n for n, p in model.named_parameters()
                   if p.requires_grad and "lora" in n.lower()]
    print(f"[{time.strftime('%H:%M:%S')}] trainable_params={_n_trainable:,} "
          f"total={_n_total:,} lora_modules={len(_lora_names)} "
          f"first_lora={_lora_names[0] if _lora_names else 'NONE'}", flush=True)

    # Tiny dataset: 4 rows. Pre-format the chat template with enable_thinking=False
    # — TRL's GRPOTrainer doesn't expose a hook for chat_template_kwargs, so we
    # hand it a string prompt instead of a conversation list.
    payload = json.loads(Path(DATASET_JSON).read_text())
    rows = payload["rows"][:4]
    print(f"[{time.strftime('%H:%M:%S')}] using {len(rows)} smoke rows", flush=True)

    def format_prompt(r):
        msgs = build_rewrite_messages(r["proposition"], r["text"], int(r["word_count"]))
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    ds = Dataset.from_list([
        {
            "prompt": format_prompt(r),
            "proposition": r["proposition"],
            "word_count": int(r["word_count"]),
        }
        for r in rows
    ])

    def score_with_judge(props, rewrites):
        scores = []
        for p, para in zip(props, rewrites):
            user_msg = CLARITY_RUBRIC.format(proposition=p, paragraph=para or "")
            try:
                resp = judge_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{
                        "role": "user",
                        "content": ("You are a careful expert evaluator. Follow the rubric "
                                    "and return valid JSON.\n\n" + user_msg),
                    }],
                    temperature=0.0,
                    max_tokens=180,
                )
                text = resp.choices[0].message.content or ""
            except Exception as e:
                print(f"  [judge WARN] {e!r}", flush=True)
                text = ""
            scores.append(parse_score(text))
        return scores

    step = {"n": 0}

    def reward_fn(prompts, completions, **kwargs):
        def extract(c):
            if isinstance(c, list) and c and isinstance(c[0], dict):
                return c[0].get("content", "")
            return str(c)
        rewrites = [extract(c) for c in completions]
        scores = score_with_judge(kwargs["proposition"], rewrites)
        step["n"] += 1
        mean = sum(scores) / max(len(scores), 1)
        print(f"[{time.strftime('%H:%M:%S')}] reward_fn call #{step['n']}  "
              f"n_rollouts={len(rewrites)}  mean_score={mean:.2f}  "
              f"sample_rewrite={rewrites[0][:120]!r}", flush=True)
        return scores

    # TRL 0.22.2 + transformers 5.3 compat: transformers' _is_package_available
    # now returns (bool, version) by default, but TRL's import_utils assumes a
    # plain bool — so every _*_available module flag is a truthy tuple, which
    # triggers a spurious `from vllm_ascend import ...` inside grpo_trainer.
    # Correct the flags to bools before the lazy grpo_trainer loader runs.
    import trl.import_utils as _trl_imp
    for _attr in [a for a in dir(_trl_imp) if a.startswith("_") and a.endswith("_available")]:
        _v = getattr(_trl_imp, _attr)
        if isinstance(_v, tuple):
            setattr(_trl_imp, _attr, bool(_v[0]))
    # vllm 0.18.1 renamed GuidedDecodingParams -> StructuredOutputsParams; TRL
    # 0.22.2 still does `from vllm.sampling_params import GuidedDecodingParams`.
    # We need use_vllm=True for fast rollouts, so alias the symbol. The only
    # call-site that would break is the regex-guided decoding path, which we
    # don't use (no guided_decoding_regex set on GRPOConfig).
    import vllm.sampling_params as _vllm_sp
    if not hasattr(_vllm_sp, "GuidedDecodingParams"):
        _vllm_sp.GuidedDecodingParams = _vllm_sp.StructuredOutputsParams

    from trl import GRPOConfig, GRPOTrainer

    # Workaround for TRL 0.22.2 bug: prepare_peft_model (called inside
    # GRPOTrainer.__init__) does `dataclasses.replace(args, gradient_checkpointing=False)`,
    # which re-runs __post_init__ on a config where generation_batch_size and
    # steps_per_generation are both already populated (computed during the
    # first __post_init__). The `both set` branch raises unconditionally.
    # Reset generation_batch_size to None so the second pass recomputes it.
    _orig_grpo_post = GRPOConfig.__post_init__
    def _grpo_post_tolerant(self):
        if self.generation_batch_size is not None and self.steps_per_generation is not None:
            self.generation_batch_size = None
        _orig_grpo_post(self)
    GRPOConfig.__post_init__ = _grpo_post_tolerant

    # GRPOConfig ported from Qwen3_5_(4B)_Vision_GRPO.ipynb with two changes:
    #   max_steps=1  (smoke, not the notebook's 60)
    #   max_completion_length=200  (rewrite task ~90 words, not math proof)
    cfg = GRPOConfig(
        output_dir=OUT_DIR,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        # TRL 0.22.2 requires generation_batch_size % num_generations == 0, so
        # per_device_train_batch_size must be a multiple of num_generations.
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_prompt_length=1024,
        max_completion_length=200,
        max_steps=1,
        save_strategy="no",
        max_grad_norm=0.1,
        bf16=True,
        report_to="none",
        temperature=1.0,
        top_p=1.0,
        seed=3407,
        # GSPO-style sampling (from the Unsloth GRPO nb):
        importance_sampling_level="sequence",
        mask_truncated_completions=False,
        loss_type="dr_grpo",
        # vLLM colocate: rollouts flow through the in-process vLLM engine
        # that FastLanguageModel(fast_inference=True) spun up above, sharing
        # GPU 0 with training. Cuts per-step rollout cost by ~5-20x vs HF
        # generate. vllm_gpu_memory_utilization mirrors the 0.4 we gave the
        # engine at load time so TRL doesn't re-reserve.
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
    )

    # TRL 0.22.2 writes `model.warnings_issued["estimate_tokens"] = True` during
    # trainer construction; transformers 5.x no longer exposes that attribute on
    # PreTrainedModel. Install the dict on the innermost base model so the
    # PEFT __getattr__ chain finds it.
    _base = getattr(getattr(model, "base_model", None), "model", None) or model
    if not hasattr(_base, "warnings_issued"):
        _base.warnings_issued = {}

    print(f"[{time.strftime('%H:%M:%S')}] building GRPOTrainer", flush=True)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"[{time.strftime('%H:%M:%S')}] trainer.train() — max_steps=1", flush=True)
    t_tr = time.time()
    trainer.train()
    print(f"[{time.strftime('%H:%M:%S')}] train finished in {time.time() - t_tr:.1f}s", flush=True)

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] saving LoRA adapter to {SAVE_DIR}", flush=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    saved = list(Path(SAVE_DIR).glob("*"))
    print(f"[{time.strftime('%H:%M:%S')}] saved files:", flush=True)
    for p in saved:
        try:
            sz = p.stat().st_size
        except Exception:
            sz = -1
        print(f"  {p.name}  ({sz} bytes)", flush=True)

    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] SMOKE DONE  total={elapsed:.1f}s  "
          f"reward_fn_calls={step['n']}", flush=True)


if __name__ == "__main__":
    main()
