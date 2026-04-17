"""OpenRouter chat completion for the rewriter model."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests

from judge.client import OPENROUTER_URL


# Qwen 2.5 72B Instruct: capable, cheap (~$0.13/$0.40 per 1M in/out on
# OpenRouter), NOT a judge in this project, and NOT one of the models
# that generated the original paul_data paragraphs (which were
# claude-sonnet-4, deepseek-chat-v3-0324, chatgpt-4o-latest). Sits cleanly
# outside both the judge pool and the original-author pool, so it avoids
# self-preference bias with either.
REWRITER_MODEL = "qwen/qwen-2.5-72b-instruct"
REWRITER_LABEL = "qwen-2.5-72b"


@dataclass
class RewriteResult:
    ok: bool
    text: Optional[str]
    error: Optional[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _strip_wrappers(text: str) -> str:
    t = text.strip()
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
) -> RewriteResult:
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
            raw = data["choices"][0]["message"]["content"]
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
    return RewriteResult(False, None, last_err or "unknown error")
