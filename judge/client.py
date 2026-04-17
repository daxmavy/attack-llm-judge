"""OpenRouter client for LLM-as-a-judge scoring."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model ids on OpenRouter. These are the two in-scope judge models per plan.md
# (Llama 3.3 70B and Gemini 2.0 Flash). GPT-4o-mini and Claude Haiku 3.5 are
# held-out and NOT used here.
JUDGE_MODELS = {
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
}


def load_api_key() -> str:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "OPENROUTER_API_KEY_EXPERIMENTS":
                    return v.strip()
    key = os.environ.get("OPENROUTER_API_KEY_EXPERIMENTS")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY_EXPERIMENTS not found in .env or environment")
    return key


@dataclass
class CallResult:
    ok: bool
    score: Optional[int]
    reasoning: Optional[str]
    raw: str
    error: Optional[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0


_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model response."""
    if not text or not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return None
    snippet = match.group(0)
    try:
        return json.loads(snippet)
    except Exception:
        # Try relaxed: find "score": N
        m = re.search(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if m:
            return {"score": float(m.group(1)), "reasoning": text[:400]}
    return None


# Some models (Gemini 2.5 Pro, GPT-5 family) are reasoning models and
# emit hidden reasoning tokens before the answer. If max_tokens is too
# small the judge's actual JSON reply gets truncated to "Here is the JSON
# requested" style garbage. For these we ask for low reasoning effort and
# bump max_tokens so the answer fits.
REASONING_MODELS = {
    "google/gemini-2.5-pro",
    "openai/gpt-5-mini",
    "openai/gpt-5",
    "openai/gpt-5-pro",
    "anthropic/claude-opus-4.7",  # thinking-enabled by default on some routings
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-0528",
}


def call_judge(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int = 500,
    temperature: float = 0.0,
    retries: int = 3,
    timeout: int = 60,
) -> CallResult:
    """One judge call. Returns CallResult with parsed score."""
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
        "response_format": {"type": "json_object"},
    }
    if model_id in REASONING_MODELS:
        payload["reasoning"] = {"effort": "low"}
        payload["max_tokens"] = max(payload["max_tokens"], 500)
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(2 ** attempt)
                continue
            # Some providers reject response_format; retry once without it.
            if resp.status_code == 400 and "response_format" in payload:
                payload.pop("response_format", None)
                time.sleep(0.5)
                continue
            resp.raise_for_status()
            data = resp.json()
            # OpenRouter sometimes returns 200 with an `error` object and no
            # `choices` under provider quota blips; treat as transient.
            if "choices" not in data or not data["choices"]:
                last_err = f"no choices in response: {str(data)[:300]}"
                time.sleep(2 ** attempt)
                continue
            raw = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage", {}) or {}
            parsed = _extract_json(raw)
            if parsed is None or "score" not in parsed:
                return CallResult(
                    ok=False,
                    score=None,
                    reasoning=None,
                    raw=raw,
                    error="could not parse score",
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
            score_val = parsed["score"]
            try:
                score_int = int(round(float(score_val)))
            except Exception:
                return CallResult(False, None, None, raw, f"score not numeric: {score_val}",
                                  usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            score_int = max(0, min(100, score_int))
            return CallResult(
                ok=True,
                score=score_int,
                reasoning=str(parsed.get("reasoning", ""))[:600],
                raw=raw,
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
    return CallResult(False, None, None, "", last_err or "unknown error")
