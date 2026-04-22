"""OpenRouter chat completion for the rewriter model."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

import requests

from judge.client import OPENROUTER_URL


# Default base: Qwen 2.5 1.5B Instruct — the same checkpoint the RL-training
# agent uses as the policy base (MODELS.md). Aligning the eval-suite rewriter
# with the training-policy base means inference-only methods and training-based
# methods start from the same distribution, so the "what did training add on
# top of prompting?" comparison is fair. Served locally from /workspace/hf_cache
# via rewriters/local_rewriter.py; Qwen-2.5-1.5B is not on OpenRouter, so local
# is the only path. If the weights are missing, we fail rather than silently
# swapping models.
REWRITER_MODEL = "qwen/qwen-2.5-1.5b-instruct"
REWRITER_LABEL = "qwen-2.5-1.5b"


@dataclass
class RewriteResult:
    ok: bool
    text: Optional[str]
    error: Optional[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0


_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _strip_wrappers(text: str) -> str:
    # Strip any leading <think>...</think> block first — Qwen3 emits these even
    # with /no_think in the user message (the block is empty, but still present).
    t = _THINK_BLOCK_RE.sub("", text).strip()
    # Strip accidental leading "Here is the rewritten paragraph:" type lines.
    if "\n" in t:
        first, rest = t.split("\n", 1)
        if any(k in first.lower() for k in ("here is", "here's", "rewritten paragraph", "sure,")):
            t = rest.strip()
    # Strip outer quotes.
    if len(t) >= 2 and t[0] in '"\u201c\u2018' and t[-1] in '"\u201d\u2019':
        t = t[1:-1].strip()
    # Strip triple-quoted blocks.
    if t.startswith('"""') and t.endswith('"""'):
        t = t[3:-3].strip()
    return t


def call_rewriter(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model_id: str = REWRITER_MODEL,
    max_tokens: int = 400,
    temperature: float = 0.7,
    retries: int = 3,
    timeout: int = 90,
    prefer_local: bool = True,
) -> RewriteResult:
    # Route to local vLLM serving when weights are on disk. Default on because
    # the project's default base (Qwen-2.5-1.5B) is only available locally.
    if prefer_local:
        from rewriters.vllm_rewriter import is_local_available, call_rewriter_local
        if is_local_available(model_id):
            return call_rewriter_local(system_prompt, user_prompt, model_id,
                                        max_tokens=max_tokens, temperature=temperature)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            # OpenRouter can return 200 with an `error` object and no `choices`
            # under transient upstream failures (429-equivalent, provider quotas).
            if "choices" not in data or not data["choices"]:
                last_err = f"no choices in response: {str(data)[:300]}"
                time.sleep(2 ** attempt)
                continue
            raw = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage", {}) or {}
            return RewriteResult(
                ok=True,
                text=_strip_wrappers(raw),
                error=None,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
        except (KeyError, TypeError, IndexError) as e:
            last_err = f"bad response shape: {e}"
            time.sleep(2 ** attempt)
    return RewriteResult(False, None, last_err or "unknown error")
