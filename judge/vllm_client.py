"""Local judge serving — thin HTTP client over `judge.server`.

Historically this module ran vLLM in-process, keeping one `LLM` object per
judge alive inside whichever script imported it. That was the 2026-04-21
co-location pattern that forced judges down to `gpu_memory_utilization≈0.28`
and silently overcommitted the KV cache during rotation.

As of 2026-04-22 this module is a thin client that forwards to the
out-of-process judge server (see `judge.http_client` + `judge.server`). The
public surface is unchanged — `is_local_available`, `call_judge_local`,
`call_judge_local_batch` — so callers (`judge.client` fallback path, the
held-out eval suite) continue to work.

Connection policy
-----------------
- The server endpoint is read from `JUDGE_HTTP_ENDPOINT` env var, falling back
  to `http://127.0.0.1:8127`.
- No automatic spawn here. The calling script is expected to bring up a
  judge server via `judge.http_client.spawn_judge_server` before the first
  call. If the server is not reachable, every call returns a CallResult with
  `ok=False` and an explanatory error string (matching the previous semantics
  when vLLM itself failed to load).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from judge.client import CallResult, _extract_json

sys.path.insert(0, "/home/shil6647/attack-llm-judge")


os.environ.setdefault("HF_HOME", "/data/shil6647/attack-llm-judge/hf_cache")
os.environ.setdefault("VLLM_CACHE_ROOT", "/data/shil6647/attack-llm-judge/vllm_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


HF_CACHE = Path(os.environ["HF_HOME"])


def _load_local_map() -> dict[str, str]:
    """Map lookup names (both HF id and lower-case slug) → HF id.

    Accepts either the canonical HF id or the slug as a lookup key so existing
    callers (which pass the HF id into `call_judge_local`) work unchanged.
    """
    from config.models import JUDGE_REGISTRY
    out: dict[str, str] = {}
    for _slug, (_name, hf_id) in JUDGE_REGISTRY.items():
        out[hf_id] = hf_id
        out[hf_id.lower()] = hf_id
    return out


def _lookup_slug(hf_id: str) -> str | None:
    """Reverse-lookup slug from HF id — the server addresses judges by slug,
    not by HF id."""
    from config.models import JUDGE_REGISTRY
    for slug, (_name, fid) in JUDGE_REGISTRY.items():
        if fid == hf_id:
            return slug
    return None


LOCAL_MODEL_MAP: dict[str, str] = _load_local_map()


def _hf_dir(hf_id: str) -> Path:
    return HF_CACHE / ("models--" + hf_id.replace("/", "--"))


def is_local_available(slug: str) -> bool:
    """True if the judge's weights are on local disk.

    Same semantics as the pre-refactor version — the caller uses this to
    decide whether to route through the local vLLM path or fall back to
    OpenRouter. Network reachability to the judge server is *not* checked
    here; if weights exist but the server is down, the subsequent
    `call_judge_local` returns a non-ok CallResult with a clear error.
    """
    hf_id = LOCAL_MODEL_MAP.get(slug)
    if not hf_id:
        return False
    d = _hf_dir(hf_id)
    return d.exists() and any(d.rglob("*.safetensors"))


def _endpoint() -> str:
    return os.environ.get("JUDGE_HTTP_ENDPOINT", "http://127.0.0.1:8127")


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


def _generate(slug: str, pairs: list[tuple[str, str]], *,
               max_tokens: int, temperature: float) -> list[str]:
    """Forward (system, user) pairs to the server's /generate endpoint."""
    from judge.http_client import JudgeHTTP
    # The HTTP client's generate_raw() is a thin passthrough; using it here
    # keeps one place responsible for the endpoint wire format.
    judge = JudgeHTTP(
        name=slug,
        rubric="clarity",          # unused by /generate, but required by dataclass
        endpoint=_endpoint(),
        auto_load=False,            # server auto-loads on first /generate call
    )
    return judge.generate_raw(pairs, max_tokens=max_tokens, temperature=temperature)


def call_judge_local(model_id: str, system_prompt: str, user_prompt: str,
                      max_tokens: int = 250, temperature: float = 0.0) -> CallResult:
    hf_id = LOCAL_MODEL_MAP.get(model_id)
    if not hf_id:
        return CallResult(False, None, None, "", f"no local mapping for {model_id}")
    slug = _lookup_slug(hf_id)
    if not slug:
        return CallResult(False, None, None, "",
                           f"no registry slug for hf_id={hf_id}")
    try:
        texts = _generate(slug, [(system_prompt, user_prompt)],
                           max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        return CallResult(False, None, None, "", f"judge server call failed: {e}")
    return _parse(texts[0])


def call_judge_local_batch(model_id: str, pairs: list[tuple[str, str]],
                             max_tokens: int = 250, temperature: float = 0.0) -> list[CallResult]:
    """Batched variant — sends N prompts to the server in one call. Preserves order."""
    hf_id = LOCAL_MODEL_MAP.get(model_id)
    if not hf_id:
        return [CallResult(False, None, None, "", f"no local mapping for {model_id}")] * len(pairs)
    slug = _lookup_slug(hf_id)
    if not slug:
        return [CallResult(False, None, None, "",
                             f"no registry slug for hf_id={hf_id}")] * len(pairs)
    try:
        texts = _generate(slug, pairs, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        return [CallResult(False, None, None, "", f"judge server call failed: {e}")] * len(pairs)
    return [_parse(t) for t in texts]
