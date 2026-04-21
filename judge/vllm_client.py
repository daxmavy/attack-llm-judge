"""Local judge serving via vLLM.

**Mirrors `training/scripts/heldout_only_eval.py::JudgeVLLM` as closely as
possible.** Same config values (dtype=bf16, max_model_len=3072, enforce_eager=True),
same prompt-building flow (HF tokenizer `apply_chat_template` → vLLM `generate`),
same JSON-parse fallback. Kept as a local copy rather than imported from the
training script so the training script stays untouched even if the eval suite
evolves.

The public interface (`call_judge_local`) accepts an already-formatted system+user
prompt pair (built by `judge.rubrics.build_prompt` upstream), so the rubric text
and system prompt are not hardcoded here — this lets the eval suite's existing
prompt pipeline flow through unchanged.

Models are loaded lazily and cached. A single generation lock serialises calls
into `llm.generate` to avoid issues when multiple eval threads share one vLLM
engine. For bulk use, call `call_judge_local_batch` directly — vLLM's continuous
batching is only useful when many prompts are submitted together.

**VRAM caveat:** `gpu_memory_utilization=0.30` matches training's post-hoc eval
setting. Three attack judges at 0.30 each ≈ 0.90 of an 80 GB A100 — do not run
this alongside an active training job. If a training job is live, either lower
`DEFAULT_GPU_MEM_UTIL` (env var `EVAL_VLLM_MEM_UTIL`) or defer the eval run.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from judge.client import CallResult, _extract_json


# Keep these in sync with the training script's env setup. These are
# no-ops if already set. (Matches /home/max/attack-llm-judge/training/
# scripts/heldout_only_eval.py lines 29-31.)
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("VLLM_CACHE_ROOT", "/workspace/vllm_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


HF_CACHE = Path(os.environ["HF_HOME"])

DEFAULT_GPU_MEM_UTIL = float(os.environ.get("EVAL_VLLM_MEM_UTIL", "0.30"))
DEFAULT_MAX_MODEL_LEN = int(os.environ.get("EVAL_VLLM_MAX_MODEL_LEN", "3072"))


# OpenRouter slug -> HF model id. Extend when new local judges arrive.
LOCAL_MODEL_MAP: dict[str, str] = {
    "qwen/qwen-2.5-7b-instruct":        "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it":             "google/gemma-2-9b-it",
}


_LOCK = threading.Lock()
_GEN_LOCK = threading.Lock()
_CACHE: dict[str, "_Judge"] = {}


def _hf_dir(hf_id: str) -> Path:
    return HF_CACHE / ("models--" + hf_id.replace("/", "--"))


def is_local_available(slug: str) -> bool:
    hf_id = LOCAL_MODEL_MAP.get(slug)
    if not hf_id:
        return False
    d = _hf_dir(hf_id)
    return d.exists() and any(d.rglob("*.safetensors"))


def _supports_system_role(tok) -> bool:
    try:
        tok.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            tokenize=False, add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


class _Judge:
    """Direct port of training/scripts/heldout_only_eval.py::JudgeVLLM.

    Differences from the training class:
    - Accepts the full (system, user) prompt pair at call time rather than
      formatting from CLARITY_RUBRIC (so the eval suite's own rubric/prompt
      pipeline flows through unchanged).
    - Adds a batch entry point that maps 1:1 onto vLLM's `llm.generate`.
    """

    def __init__(self, hf_id: str, gpu_mem_util: float = DEFAULT_GPU_MEM_UTIL,
                 max_model_len: int = DEFAULT_MAX_MODEL_LEN):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        self.hf_id = hf_id
        self.tok = AutoTokenizer.from_pretrained(
            hf_id, cache_dir=str(HF_CACHE), token=os.environ.get("HF_TOKEN"),
        )
        self.sys_ok = _supports_system_role(self.tok)
        self.llm = LLM(
            model=hf_id,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            enforce_eager=True,
            download_dir=str(HF_CACHE),
        )
        self._SamplingParams = SamplingParams

    def _build(self, system_prompt: str, user_prompt: str) -> str:
        if self.sys_ok:
            chat = [{"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}]
        else:
            chat = [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]
        return self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def generate(self, pairs: list[tuple[str, str]], max_tokens: int, temperature: float) -> list[str]:
        prompts = [self._build(s, u) for (s, u) in pairs]
        sp = self._SamplingParams(
            temperature=max(float(temperature), 0.0),
            max_tokens=int(max_tokens),
        )
        with _GEN_LOCK:
            outs = self.llm.generate(prompts, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


def _get(hf_id: str) -> _Judge:
    with _LOCK:
        if hf_id in _CACHE:
            return _CACHE[hf_id]
        j = _Judge(hf_id)
        _CACHE[hf_id] = j
        return j


def _parse(raw: str) -> CallResult:
    parsed = _extract_json(raw)
    if parsed is None or "score" not in parsed:
        return CallResult(False, None, None, raw, "could not parse score")
    try:
        score_int = int(round(float(parsed["score"])))
    except Exception:
        return CallResult(False, None, None, raw, f"score not numeric: {parsed.get('score')}")
    score_int = max(0, min(100, score_int))
    return CallResult(True, score_int, str(parsed.get("reasoning", ""))[:600],
                       raw, None, 0, 0)


def call_judge_local(model_id: str, system_prompt: str, user_prompt: str,
                      max_tokens: int = 250, temperature: float = 0.0) -> CallResult:
    hf_id = LOCAL_MODEL_MAP.get(model_id)
    if not hf_id:
        return CallResult(False, None, None, "", f"no local mapping for {model_id}")
    try:
        judge = _get(hf_id)
    except Exception as e:
        return CallResult(False, None, None, "", f"local load failed: {e}")
    try:
        texts = judge.generate([(system_prompt, user_prompt)],
                                max_tokens=max_tokens, temperature=temperature)
        return _parse(texts[0])
    except Exception as e:
        return CallResult(False, None, None, "", f"local gen failed: {e}")


def call_judge_local_batch(model_id: str, pairs: list[tuple[str, str]],
                             max_tokens: int = 250, temperature: float = 0.0) -> list[CallResult]:
    """Batched variant — send N prompts to vLLM in one call. Preserves order."""
    hf_id = LOCAL_MODEL_MAP.get(model_id)
    if not hf_id:
        return [CallResult(False, None, None, "", f"no local mapping for {model_id}")] * len(pairs)
    try:
        judge = _get(hf_id)
    except Exception as e:
        return [CallResult(False, None, None, "", f"local load failed: {e}")] * len(pairs)
    try:
        texts = judge.generate(pairs, max_tokens=max_tokens, temperature=temperature)
        return [_parse(t) for t in texts]
    except Exception as e:
        return [CallResult(False, None, None, "", f"local gen failed: {e}")] * len(pairs)
