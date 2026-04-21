"""Local rewriter serving via vLLM.

**Mirrors `training/scripts/heldout_only_eval.py::JudgeVLLM` config exactly**
(dtype=bf16, max_model_len=3072, enforce_eager=True, download_dir=/data/shil6647/attack-llm-judge/hf_cache).
The class is structurally the same — only the post-generation parse differs:
rewriter calls return raw text (stripped of wrapper junk), judges parse JSON.

Default base is `Qwen/Qwen2.5-1.5B-Instruct`, the same policy base used by the
training agent, so inference-only methods and training-based methods start from
the same distribution.

Singleton cache + generation lock, same pattern as `judge/vllm_client.py`.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from rewriters.rewriter_client import RewriteResult, _strip_wrappers


os.environ.setdefault("HF_HOME", "/data/shil6647/attack-llm-judge/hf_cache")
os.environ.setdefault("VLLM_CACHE_ROOT", "/data/shil6647/attack-llm-judge/vllm_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


HF_CACHE = Path(os.environ["HF_HOME"])

# Rewriter is small (1.5B) so it needs less mem than judges. Default 0.15 so it
# can coexist with 3 judges at 0.25 each on an A100-80GB when eval is run on
# its own (i.e. no training job is active). Override via env.
DEFAULT_GPU_MEM_UTIL = float(os.environ.get("EVAL_REWRITER_MEM_UTIL", "0.15"))
DEFAULT_MAX_MODEL_LEN = int(os.environ.get("EVAL_REWRITER_MAX_MODEL_LEN", "3072"))


LOCAL_MODEL_MAP: dict[str, str] = {
    "qwen/qwen-2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen/qwen-2.5-7b-instruct":   "Qwen/Qwen2.5-7B-Instruct",
    "qwen/qwen-2.5-72b-instruct":  "Qwen/Qwen2.5-72B-Instruct",
}


_LOCK = threading.Lock()
_GEN_LOCK = threading.Lock()
_CACHE: dict[str, "_Rewriter"] = {}


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


class _Rewriter:
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


def _get(hf_id: str) -> _Rewriter:
    with _LOCK:
        if hf_id in _CACHE:
            return _CACHE[hf_id]
        r = _Rewriter(hf_id)
        _CACHE[hf_id] = r
        return r


def call_rewriter_local(system_prompt: str, user_prompt: str,
                         model_id: str, max_tokens: int = 400,
                         temperature: float = 0.7) -> RewriteResult:
    hf_id = LOCAL_MODEL_MAP.get(model_id)
    if not hf_id:
        return RewriteResult(False, None, f"no local mapping for {model_id}")
    try:
        rew = _get(hf_id)
    except Exception as e:
        return RewriteResult(False, None, f"local load failed: {e}")
    try:
        texts = rew.generate([(system_prompt, user_prompt)],
                              max_tokens=max_tokens, temperature=temperature)
        return RewriteResult(True, _strip_wrappers(texts[0]), None)
    except Exception as e:
        return RewriteResult(False, None, f"local gen failed: {e}")


def call_rewriter_local_batch(model_id: str,
                                pairs: list[tuple[str, str]],
                                max_tokens: int = 400,
                                temperature: float = 0.7) -> list[RewriteResult]:
    """Batched variant for BoN and any future bulk rewriter use."""
    hf_id = LOCAL_MODEL_MAP.get(model_id)
    if not hf_id:
        return [RewriteResult(False, None, f"no local mapping for {model_id}")] * len(pairs)
    try:
        rew = _get(hf_id)
    except Exception as e:
        return [RewriteResult(False, None, f"local load failed: {e}")] * len(pairs)
    try:
        texts = rew.generate(pairs, max_tokens=max_tokens, temperature=temperature)
        return [RewriteResult(True, _strip_wrappers(t), None) for t in texts]
    except Exception as e:
        return [RewriteResult(False, None, f"local gen failed: {e}")] * len(pairs)
