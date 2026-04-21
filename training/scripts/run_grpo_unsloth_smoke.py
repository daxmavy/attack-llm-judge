"""Smoke test for the Unsloth + QLoRA + GRPO stack on Qwen3-30B-A3B-Instruct-2507.

Runs ONE GRPO step on a 4-paragraph micro-dataset with a single judge
(gemma-2-9b-it) as the reward signal, then saves the LoRA adapter.

Architecture:
- GPU 0: Unsloth FastModel (rewriter, 4-bit QLoRA) + TRL GRPOTrainer + vLLM rollouts
  via UNSLOTH_VLLM_STANDBY=1. Unsloth grabs ~88% of GPU 0.
- GPU 1: gemma-2-9b-it judge as a vLLM OpenAI-compatible server (subprocess).
  Main process talks to it over HTTP. Keeps the judge's memory budget off GPU 0.

Purpose: verify the dependency stack (unsloth FastModel + vLLM fast_inference +
TRL 0.24.x GRPOTrainer + bitsandbytes 4-bit) loads + runs end-to-end before we
invest in the full training run. Not meant to produce a useful checkpoint.

Run:
    HF_HOME=/data/shil6647/attack-llm-judge/hf_cache \
    UNSLOTH_VLLM_STANDBY=1 \
    python3 training/scripts/run_grpo_unsloth_smoke.py
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
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
# Mixed A100/L40S host → force PCI bus order so CUDA device IDs are stable.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
# Training/rewriter process must see only GPU 0. The judge subprocess launched
# below re-scopes its own CUDA_VISIBLE_DEVICES to GPU 1.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Best-effort load of HF_TOKEN from .env
env_path = Path("/home/shil6647/attack-llm-judge/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

from unsloth import FastModel  # must come before torch / trl imports

import torch
from datasets import Dataset
from openai import OpenAI
from transformers import AutoTokenizer


REWRITER_MODEL = "unsloth/Qwen3-30B-A3B-Instruct-2507"
JUDGE_MODEL = "google/gemma-2-9b-it"
JUDGE_GPU = "1"
JUDGE_HOST = "127.0.0.1"
JUDGE_PORT = 8910
JUDGE_URL = f"http://{JUDGE_HOST}:{JUDGE_PORT}/v1"
DATASET_JSON = "/home/shil6647/attack-llm-judge/data/controversial_40_3fold.json"
SAVE_DIR = "/data/shil6647/attack-llm-judge/smoke_test/lora"
OUT_DIR = "/data/shil6647/attack-llm-judge/smoke_test/ckpt"
HF_CACHE = "/data/shil6647/attack-llm-judge/hf_cache"
JUDGE_LOG = "/data/shil6647/attack-llm-judge/tmp/smoke_judge_server.log"

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
    """Launch gemma-2-9b-it as a vLLM OpenAI-compatible server on JUDGE_GPU.

    Returns the Popen handle. Polls /v1/models until the server responds or the
    timeout elapses. Registers an atexit hook to terminate the subprocess.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = JUDGE_GPU
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Judge is in the shared cache; point HF at it so we don't re-download 18 GB.
    env["HF_HOME"] = "/data/resource/huggingface"
    env["VLLM_CACHE_ROOT"] = os.environ.get("VLLM_CACHE_ROOT",
                                            "/data/shil6647/attack-llm-judge/vllm_cache")
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


def make_rewrite_prompt(proposition: str, text: str, word_count: int):
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
    print(f"[{time.strftime('%H:%M:%S')}] starting smoke test", flush=True)

    # Launch the judge vLLM server on GPU 1 first (so we know it's ready before
    # we burn time loading Qwen3-30B on GPU 0).
    start_judge_server()
    judge_client = OpenAI(base_url=JUDGE_URL, api_key="EMPTY")

    # Load the rewriter via Unsloth FastModel (QLoRA 4-bit + vLLM fast_inference)
    print(f"[{time.strftime('%H:%M:%S')}] loading rewriter {REWRITER_MODEL}", flush=True)
    t_load = time.time()
    model, tokenizer = FastModel.from_pretrained(
        model_name=REWRITER_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=32,
        gpu_memory_utilization=0.85,  # judge is on GPU 1 → we can use most of GPU 0
        float8_kv_cache=True,
    )
    print(f"[{time.strftime('%H:%M:%S')}] rewriter loaded in {time.time() - t_load:.1f}s  "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # Attach LoRA
    model = FastModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Tiny dataset: 4 rows
    payload = json.loads(Path(DATASET_JSON).read_text())
    rows = payload["rows"][:4]
    print(f"[{time.strftime('%H:%M:%S')}] using {len(rows)} smoke rows", flush=True)
    ds = Dataset.from_list([
        {
            "prompt": make_rewrite_prompt(r["proposition"], r["text"], int(r["word_count"])),
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

    from trl import GRPOConfig, GRPOTrainer

    cfg = GRPOConfig(
        output_dir=OUT_DIR,
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_completion_length=260,
        learning_rate=5e-6,
        beta=0.01,
        max_steps=1,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        report_to="none",
        temperature=1.0,
        top_p=1.0,
        seed=42,
        loss_type="dr_grpo",
        use_vllm=True,
        vllm_mode="colocate",
        optim="adamw_8bit",
    )

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

    # Save LoRA adapter
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
