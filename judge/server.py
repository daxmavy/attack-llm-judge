"""Stdlib HTTP judge server — one process per GPU.

Launched as a subprocess (usually by `judge.http_client.spawn_judge_server`)
with `CUDA_VISIBLE_DEVICES` pinned to a single GPU. Loads vLLM judge engines
lazily on first /load or /score, keeps them cached, swaps rubrics in-place
without reloading the engine.

Endpoints
---------
- POST /load          {"slug", "rubric"?, "max_model_len"?, "gpu_memory_utilization"?}
                      → loads a judge. Idempotent — returns already_loaded if cached.
- POST /score         {"slug", "rubric"?, "items": [{"proposition", "text"}], "include_raw"?}
                      → {"scores": [...], "reasonings": [...], "raw"?: [...]}
                      Auto-loads if rubric is provided and judge isn't cached.
- POST /generate      {"slug", "pairs": [[system, user], ...], "max_tokens"?, "temperature"?}
                      → {"texts": [...]}  — raw generation for callers that build
                      their own (system, user) prompts (e.g. the held-out eval
                      suite in judge/vllm_client.py).
- POST /set_rubric    {"slug", "rubric"} → swap rubric text without reloading.
- POST /unload        {"slug"} → drop the vLLM engine; free VRAM.
- GET  /health        {"ok": true, "loaded": [slugs...], "pid": ...}
- POST /shutdown      {"ok": true} and schedules SIGTERM on self.

All endpoints respond with JSON; errors are HTTP 4xx/5xx with an `error`
field. Single-threaded per-judge access is enforced by `_GEN_LOCK` — vLLM's
`llm.generate` doesn't like concurrent callers.

Intentionally stdlib-only (http.server + json). FastAPI/uvicorn would pull
async surprises into what is fundamentally a one-client-at-a-time workflow.

Launch directly for debugging:
    CUDA_VISIBLE_DEVICES=0 python3 -m judge.server --port 8127
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# Load .env BEFORE importing anything that reads HF_TOKEN.
_ENV_PATH = "/home/shil6647/attack-llm-judge/.env"
if os.path.exists(_ENV_PATH):
    for _ln in open(_ENV_PATH):
        _ln = _ln.strip()
        if "=" in _ln and not _ln.startswith("#"):
            _k, _v = _ln.split("=", 1)
            os.environ.setdefault(_k, _v)

os.environ.setdefault("HF_HOME", "/data/shil6647/attack-llm-judge/hf_cache")
os.environ.setdefault("VLLM_CACHE_ROOT", "/data/shil6647/attack-llm-judge/vllm_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add repo root so config.models + judge.rubrics imports resolve whether
# this is launched via `-m judge.server` or directly as a script.
sys.path.insert(0, "/home/shil6647/attack-llm-judge")

from config.models import JUDGE_REGISTRY, require_config  # noqa: E402
from judge.rubrics import (  # noqa: E402
    JUDGE_SYSTEM,
    RUBRICS,
    parse_score_and_reasoning,
    supports_system_role,
)


def _auto_gpu_mem_util(hf_id: str) -> float:
    """Default vLLM memory share — assumes we own a whole GPU.

    The co-located values (0.22-0.28) were a workaround for the
    2026-04-21 single-GPU pattern. With a dedicated judge GPU we can
    afford roughly double. Callers can still override via the
    `gpu_memory_utilization` field on /load.
    """
    mid = hf_id.lower()
    if "70b" in mid or "49b" in mid:
        return 0.90
    if "27b" in mid or "32b" in mid:
        return 0.70
    if "13b" in mid or "14b" in mid:
        return 0.55
    if "8b" in mid or "9b" in mid or "7b" in mid:
        return 0.45
    return 0.40


class JudgeState:
    """One vLLM judge engine plus rubric/tokenizer state."""

    def __init__(self, slug: str, hf_id: str, tok, llm, sp, sys_ok: bool, rubric: str):
        self.slug = slug
        self.hf_id = hf_id
        self.tok = tok
        self.llm = llm
        self.sp = sp
        self.sys_ok = sys_ok
        self.rubric_name = rubric
        # vLLM's generate is not safe to call concurrently from multiple
        # threads. Serialise here; the HTTP layer is ThreadingHTTPServer.
        self.gen_lock = threading.Lock()

    @classmethod
    def load(cls, slug: str, rubric: str = "clarity", max_model_len: int = 3072,
             gpu_memory_utilization: float | None = None) -> "JudgeState":
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        _wandb_name, hf_id = JUDGE_REGISTRY[slug]
        if gpu_memory_utilization is None:
            gpu_memory_utilization = _auto_gpu_mem_util(hf_id)

        # NOTE: HF discussion #84 on Mistral-Small-3.1-24B recommends
        # fix_mistral_regex=True on Mistral tokenizers, but under
        # transformers 5.5.0 that kwarg triggers AttributeError:
        # 'tokenizers.Tokenizer' object has no attribute 'backend_tokenizer'
        # on RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic. The regex
        # warning is cosmetic — verify_mistral24_scoring.py passed end-to-end
        # without the kwarg, so we leave it off until transformers fixes it.
        tok = AutoTokenizer.from_pretrained(
            hf_id, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
            token=os.environ.get("HF_TOKEN"),
        )
        sys_ok = supports_system_role(tok)
        llm = LLM(
            model=hf_id, dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len, enforce_eager=True,
            download_dir="/data/shil6647/attack-llm-judge/hf_cache",
        )
        sp = SamplingParams(temperature=0.0, max_tokens=180)
        return cls(slug=slug, hf_id=hf_id, tok=tok, llm=llm, sp=sp,
                    sys_ok=sys_ok, rubric=rubric)

    def set_rubric(self, rubric: str) -> None:
        if rubric not in RUBRICS:
            raise ValueError(f"unknown rubric: {rubric}")
        self.rubric_name = rubric

    def _build_prompt(self, proposition: str, paragraph: str) -> str:
        user_msg = RUBRICS[self.rubric_name].format(
            proposition=proposition, paragraph=paragraph or "")
        if self.sys_ok:
            chat = [{"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg}]
        else:
            chat = [{"role": "user", "content": JUDGE_SYSTEM + "\n\n" + user_msg}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        # Qwen3 family defaults to thinking mode which burns the 180-token budget.
        if "qwen3" in self.hf_id.lower():
            kwargs["enable_thinking"] = False
        return self.tok.apply_chat_template(chat, **kwargs)

    def _build_from_pair(self, system_prompt: str, user_prompt: str) -> str:
        if self.sys_ok:
            chat = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        else:
            chat = [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if "qwen3" in self.hf_id.lower():
            kwargs["enable_thinking"] = False
        return self.tok.apply_chat_template(chat, **kwargs)

    def generate_raw(self, pairs: list[tuple[str, str]], *, max_tokens: int,
                      temperature: float) -> list[str]:
        """For legacy callers that build their own (system, user) prompts."""
        from vllm import SamplingParams
        prompts = [self._build_from_pair(s, u) for (s, u) in pairs]
        sp = SamplingParams(temperature=max(float(temperature), 0.0),
                             max_tokens=int(max_tokens))
        with self.gen_lock:
            outs = self.llm.generate(prompts, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def score_batch(self, propositions: list[str], paragraphs: list[str],
                     *, include_raw: bool = False) -> dict:
        prompts = [self._build_prompt(p, t) for p, t in zip(propositions, paragraphs)]
        with self.gen_lock:
            outs = self.llm.generate(prompts, self.sp, use_tqdm=False)
        scores: list[float] = []
        reasonings: list[str] = []
        raws: list[str] = []
        for o in outs:
            raw = o.outputs[0].text if o.outputs else ""
            raws.append(raw)
            s, r = parse_score_and_reasoning(raw)
            # Legacy training reward falls back to 50.0 on unparseable;
            # preserve that semantics so the reward_fn behaviour is unchanged.
            scores.append(float(s) if s is not None else 50.0)
            reasonings.append(r)
        payload: dict = {"scores": scores, "reasonings": reasonings}
        if include_raw:
            payload["raw"] = raws
        return payload

    def teardown(self) -> None:
        try:
            del self.llm
        except Exception:
            pass


_STATE: dict[str, JudgeState] = {}
_LOAD_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    # Silence default per-request access log; keep only our own prints.
    def log_message(self, fmt: str, *args) -> None:
        return

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length", "0") or "0")
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        return json.loads(raw.decode("utf-8"))

    def _reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/health":
                self._reply(200, {"ok": True, "loaded": sorted(_STATE.keys()),
                                   "pid": os.getpid()})
            else:
                self._reply(404, {"ok": False, "error": f"no GET {path}"})
        except Exception as e:  # pragma: no cover — defensive
            self._reply(500, {"ok": False, "error": f"{type(e).__name__}: {e}",
                               "traceback": traceback.format_exc()[-2000:]})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_body()
            if path == "/load":
                self._do_load(body)
            elif path == "/score":
                self._do_score(body)
            elif path == "/generate":
                self._do_generate(body)
            elif path == "/set_rubric":
                self._do_set_rubric(body)
            elif path == "/unload":
                self._do_unload(body)
            elif path == "/shutdown":
                self._reply(200, {"ok": True})
                threading.Thread(
                    target=lambda: (time.sleep(0.1), os.kill(os.getpid(), signal.SIGTERM)),
                    daemon=True,
                ).start()
            else:
                self._reply(404, {"ok": False, "error": f"no POST {path}"})
        except Exception as e:
            self._reply(500, {"ok": False, "error": f"{type(e).__name__}: {e}",
                               "traceback": traceback.format_exc()[-2000:]})

    def _do_load(self, body: dict) -> None:
        slug = body["slug"]
        rubric = body.get("rubric", "clarity")
        if slug not in JUDGE_REGISTRY:
            self._reply(400, {"ok": False, "error": f"unknown judge slug: {slug}"})
            return
        if rubric not in RUBRICS:
            self._reply(400, {"ok": False, "error": f"unknown rubric: {rubric}"})
            return
        with _LOAD_LOCK:
            if slug in _STATE:
                _STATE[slug].set_rubric(rubric)
                self._reply(200, {"ok": True, "already_loaded": True,
                                   "slug": slug, "hf_id": _STATE[slug].hf_id,
                                   "rubric": rubric})
                return
            print(f"[judge-server] loading {slug} (rubric={rubric})", flush=True)
            t0 = time.time()
            _STATE[slug] = JudgeState.load(
                slug, rubric=rubric,
                max_model_len=int(body.get("max_model_len", 3072)),
                gpu_memory_utilization=body.get("gpu_memory_utilization"),
            )
            print(f"[judge-server]   {slug} loaded in {time.time()-t0:.1f}s", flush=True)
        self._reply(200, {"ok": True, "slug": slug, "hf_id": _STATE[slug].hf_id,
                           "rubric": rubric})

    def _do_score(self, body: dict) -> None:
        slug = body["slug"]
        rubric = body.get("rubric")
        items = body.get("items") or []
        include_raw = bool(body.get("include_raw", False))
        if slug not in _STATE:
            if rubric is None:
                self._reply(400, {"ok": False,
                                   "error": f"judge {slug} not loaded and no rubric supplied"})
                return
            with _LOAD_LOCK:
                if slug not in _STATE:
                    if slug not in JUDGE_REGISTRY:
                        self._reply(400, {"ok": False,
                                           "error": f"unknown judge slug: {slug}"})
                        return
                    print(f"[judge-server] auto-loading {slug} on /score", flush=True)
                    t0 = time.time()
                    _STATE[slug] = JudgeState.load(slug, rubric=rubric)
                    print(f"[judge-server]   {slug} loaded in {time.time()-t0:.1f}s",
                          flush=True)
        s = _STATE[slug]
        if rubric and rubric != s.rubric_name:
            s.set_rubric(rubric)
        props = [it["proposition"] for it in items]
        texts = [it["text"] for it in items]
        result = s.score_batch(props, texts, include_raw=include_raw)
        self._reply(200, {"ok": True, "slug": slug, "rubric": s.rubric_name,
                           "n": len(items), **result})

    def _do_generate(self, body: dict) -> None:
        slug = body["slug"]
        pairs = body.get("pairs") or []
        max_tokens = int(body.get("max_tokens", 250))
        temperature = float(body.get("temperature", 0.0))
        if slug not in _STATE:
            # Auto-load with the default rubric — only the model weights are
            # needed for /generate, but /load expects a valid rubric slot.
            with _LOAD_LOCK:
                if slug not in _STATE:
                    if slug not in JUDGE_REGISTRY:
                        self._reply(400, {"ok": False,
                                           "error": f"unknown judge slug: {slug}"})
                        return
                    print(f"[judge-server] auto-loading {slug} on /generate", flush=True)
                    t0 = time.time()
                    _STATE[slug] = JudgeState.load(slug, rubric="clarity")
                    print(f"[judge-server]   {slug} loaded in {time.time()-t0:.1f}s",
                          flush=True)
        s = _STATE[slug]
        norm_pairs = [(str(p[0]), str(p[1])) for p in pairs]
        texts = s.generate_raw(norm_pairs, max_tokens=max_tokens,
                                 temperature=temperature)
        self._reply(200, {"ok": True, "slug": slug, "n": len(pairs),
                           "texts": texts})

    def _do_set_rubric(self, body: dict) -> None:
        slug = body["slug"]
        rubric = body["rubric"]
        if slug not in _STATE:
            self._reply(400, {"ok": False, "error": f"judge {slug} not loaded"})
            return
        if rubric not in RUBRICS:
            self._reply(400, {"ok": False, "error": f"unknown rubric: {rubric}"})
            return
        _STATE[slug].set_rubric(rubric)
        self._reply(200, {"ok": True, "slug": slug, "rubric": rubric})

    def _do_unload(self, body: dict) -> None:
        slug = body["slug"]
        with _LOAD_LOCK:
            s = _STATE.pop(slug, None)
        if s:
            try:
                s.teardown()
            finally:
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        self._reply(200, {"ok": True, "slug": slug, "existed": s is not None})


def _install_signal_handlers(srv: ThreadingHTTPServer) -> None:
    def _handler(signum, _frame):
        print(f"[judge-server] signal {signum} → shutting down", flush=True)
        threading.Thread(target=srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8127)
    args = ap.parse_args()
    require_config()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    _install_signal_handlers(srv)
    print(f"[judge-server] listening on {args.host}:{args.port} pid={os.getpid()} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '?')}",
          flush=True)
    try:
        srv.serve_forever()
    finally:
        print(f"[judge-server] teardown pid={os.getpid()}", flush=True)
        for slug in list(_STATE):
            try:
                _STATE.pop(slug).teardown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
