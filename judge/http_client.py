"""HTTP client for the out-of-process judge server.

`JudgeHTTP` is a drop-in replacement for the in-process `JudgeVLLM` class that
used to live in `training/scripts/run_pilot_len_pen.py`. Same public surface
(`name`, `model_id`, `rubric_name`, `rubric_text`, `.score(props, paragraphs)`),
plus `.score_full()` for ICIR's raw-text needs and `.set_rubric()` for in-place
rubric swaps without reloading the vLLM engine.

The 2026-04-21 mission failed in part because all three vLLM engines (two
judges + rewriter) were co-located on a single A100 — each forced down to
`gpu_memory_utilization ≈ 0.28`, which silently overcommitted during rotation.
Routing judges through an HTTP server on its own GPU fixes that: the judge
process owns GPU 0, the rewriter process owns GPU 1, and neither can corner
the other's KV cache budget.

Typical usage from a rewriter-side script:

    from judge.http_client import JudgeHTTP, spawn_judge_server

    srv_proc, endpoint = spawn_judge_server(port=8127, gpu=0)
    try:
        judges = [
            JudgeHTTP(name=slug, rubric="clarity", endpoint=endpoint)
            for slug in IN_PANEL_SLUGS
        ]
        # ... use judges[i].score(props, paragraphs) ...
    finally:
        # Graceful HTTP shutdown so vLLM releases VRAM cleanly.
        try:
            JudgeHTTP.request_shutdown(endpoint)
        except Exception:
            pass
        srv_proc.wait(timeout=30)

If an external judge server is already running (separate terminal, CI, etc.)
skip `spawn_judge_server` and just pass the endpoint string.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from config.models import JUDGE_REGISTRY
from judge.rubrics import RUBRICS


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8127
DEFAULT_ENDPOINT = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"


class JudgeHTTPError(RuntimeError):
    """Raised when the judge server responds with a non-ok status."""


def _post(endpoint: str, path: str, payload: dict, *, timeout: float = 600.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise JudgeHTTPError(f"POST {path} → {e.code}: {detail[:500]}") from e
    data = json.loads(raw.decode("utf-8"))
    if not data.get("ok", False):
        raise JudgeHTTPError(f"POST {path} not-ok: {data.get('error', data)}")
    return data


def _get(endpoint: str, path: str, *, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(endpoint.rstrip("/") + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@dataclass
class JudgeHTTP:
    """Thin HTTP wrapper exposing the old JudgeVLLM interface.

    Parameters
    ----------
    name:
        Slug in `config.models.JUDGE_REGISTRY`. Used as the DB `judge_slug`.
    rubric:
        Rubric key (`clarity` or `informativeness`). Can be swapped later
        via `set_rubric()` without reloading the engine.
    endpoint:
        Base URL of the judge server (default 127.0.0.1:8127).
    auto_load:
        If True (default), POST /load at construction time so subsequent
        /score calls don't pay the load latency, and so the caller knows
        immediately if the server can't load the model.
    request_timeout:
        Per-request timeout (seconds). Set generously — a 49B judge load
        can take 60-90s and a large batch /score can take minutes.
    """

    name: str
    rubric: str = "clarity"
    endpoint: str = DEFAULT_ENDPOINT
    auto_load: bool = True
    request_timeout: float = 600.0
    max_model_len: int = 3072
    gpu_memory_utilization: float | None = None
    # Populated from JUDGE_REGISTRY at __post_init__ — these mirror the
    # attributes the old in-process JudgeVLLM exposed.
    model_id: str = field(init=False)
    wandb_name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.name not in JUDGE_REGISTRY:
            raise KeyError(
                f"judge slug '{self.name}' not in JUDGE_REGISTRY — "
                f"edit config/models.py"
            )
        self.wandb_name, self.model_id = JUDGE_REGISTRY[self.name]
        if self.rubric not in RUBRICS:
            raise ValueError(f"unknown rubric: {self.rubric}")
        if self.auto_load:
            self._load()

    # ---- public API ---------------------------------------------------------

    @property
    def rubric_name(self) -> str:
        return self.rubric

    @property
    def rubric_text(self) -> str:
        return RUBRICS[self.rubric]

    def score(self, propositions: list[str], paragraphs: list[str]) -> list[float]:
        """Return a list of numeric scores, one per (prop, paragraph) pair.

        Preserves legacy training-reward semantics: unparseable judge output
        falls back to 50.0 (handled server-side).
        """
        data = self._score(propositions, paragraphs, include_raw=False)
        return [float(x) for x in data["scores"]]

    def score_full(
        self,
        propositions: list[str],
        paragraphs: list[str],
        *,
        include_raw: bool = False,
    ) -> dict:
        """Return {scores, reasonings[, raw]} — used by ICIR for feedback."""
        return self._score(propositions, paragraphs, include_raw=include_raw)

    def generate_raw(
        self,
        pairs: list[tuple[str, str]],
        *,
        max_tokens: int = 250,
        temperature: float = 0.0,
    ) -> list[str]:
        """Legacy path for callers that build their own (system, user) prompts
        (e.g. `judge.vllm_client` / held-out eval). Prefer `.score()` when the
        rubric lives in `judge.rubrics` — `generate_raw` skips the server's
        score-parsing pipeline."""
        payload = {
            "slug": self.name,
            "pairs": [list(p) for p in pairs],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        data = _post(
            self.endpoint, "/generate", payload, timeout=self.request_timeout
        )
        return list(data["texts"])

    def set_rubric(self, rubric: str) -> None:
        """Swap the rubric without reloading the vLLM engine."""
        if rubric not in RUBRICS:
            raise ValueError(f"unknown rubric: {rubric}")
        _post(
            self.endpoint,
            "/set_rubric",
            {"slug": self.name, "rubric": rubric},
            timeout=self.request_timeout,
        )
        self.rubric = rubric

    def unload(self) -> None:
        """Ask the server to drop this judge's vLLM engine (frees VRAM)."""
        _post(
            self.endpoint,
            "/unload",
            {"slug": self.name},
            timeout=self.request_timeout,
        )

    # ---- internal -----------------------------------------------------------

    def _load(self) -> None:
        payload: dict[str, Any] = {
            "slug": self.name,
            "rubric": self.rubric,
            "max_model_len": self.max_model_len,
        }
        if self.gpu_memory_utilization is not None:
            payload["gpu_memory_utilization"] = self.gpu_memory_utilization
        _post(self.endpoint, "/load", payload, timeout=self.request_timeout)

    def _score(
        self, propositions: list[str], paragraphs: list[str], *, include_raw: bool
    ) -> dict:
        if len(propositions) != len(paragraphs):
            raise ValueError(
                f"propositions/paragraphs length mismatch: "
                f"{len(propositions)} vs {len(paragraphs)}"
            )
        items = [
            {"proposition": p, "text": t} for p, t in zip(propositions, paragraphs)
        ]
        payload = {
            "slug": self.name,
            "rubric": self.rubric,
            "items": items,
            "include_raw": include_raw,
        }
        data = _post(
            self.endpoint, "/score", payload, timeout=self.request_timeout
        )
        return {
            k: data[k]
            for k in ("scores", "reasonings", "raw")
            if k in data
        }

    # ---- class-level helpers ------------------------------------------------

    @staticmethod
    def health(endpoint: str = DEFAULT_ENDPOINT, *, timeout: float = 5.0) -> dict:
        return _get(endpoint, "/health", timeout=timeout)

    @staticmethod
    def request_shutdown(endpoint: str = DEFAULT_ENDPOINT, *, timeout: float = 10.0) -> None:
        _post(endpoint, "/shutdown", {}, timeout=timeout)


# -----------------------------------------------------------------------------
# Server spawn helper
# -----------------------------------------------------------------------------
def _port_listens(host: str, port: int, *, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def spawn_judge_server(
    *,
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    gpu: int = 0,
    wait_timeout: float = 300.0,
    log_path: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen, str]:
    """Launch `python -m judge.server` as a subprocess, pinned to one GPU.

    Returns (proc, endpoint). Polls /health until it responds or
    `wait_timeout` seconds have elapsed; on timeout the subprocess is
    terminated and a RuntimeError is raised.

    If an existing server is already listening on the port, returns a dummy
    Popen with the existing endpoint — no new process is launched. This lets
    a developer leave a server running in another terminal while iterating
    on rewriter code.
    """
    endpoint = f"http://{host}:{port}"
    if _port_listens(host, port):
        # Already up — don't stomp on it.
        try:
            JudgeHTTP.health(endpoint)
        except Exception as e:
            raise RuntimeError(
                f"port {port} is busy but /health failed: {e} — "
                f"kill the stale process or use a different port"
            ) from e
        print(
            f"[judge-client] reusing existing judge server at {endpoint}",
            flush=True,
        )
        return subprocess.Popen(["true"]), endpoint

    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("HF_HOME", "/data/shil6647/attack-llm-judge/hf_cache")
    env.setdefault("TRANSFORMERS_CACHE", env["HF_HOME"])
    env.setdefault("VLLM_CACHE_ROOT", "/data/shil6647/attack-llm-judge/vllm_cache")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if extra_env:
        env.update(extra_env)

    cmd = [
        sys.executable, "-u", "-m", "judge.server",
        "--host", host, "--port", str(port),
    ]
    stdout = None
    stderr = None
    if log_path:
        # Append so a caller can reuse a log across retries if desired.
        f = open(log_path, "ab", buffering=0)
        stdout = f
        stderr = subprocess.STDOUT

    print(
        f"[judge-client] spawning judge server on GPU {gpu} "
        f"(port {port}, pid-to-be)",
        flush=True,
    )
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        cwd="/home/shil6647/attack-llm-judge",
    )

    # Poll /health. vLLM import alone can take 15-30s before the listener
    # binds; the actual judge load is deferred until /load is called.
    t0 = time.time()
    last_err: Exception | None = None
    while time.time() - t0 < wait_timeout:
        if proc.poll() is not None:
            raise RuntimeError(
                f"judge server exited with code {proc.returncode} "
                f"before /health responded"
                + (f" (log: {log_path})" if log_path else "")
            )
        try:
            JudgeHTTP.health(endpoint, timeout=2.0)
            elapsed = time.time() - t0
            print(
                f"[judge-client] judge server healthy after {elapsed:.1f}s "
                f"(pid={proc.pid})",
                flush=True,
            )
            return proc, endpoint
        except Exception as e:
            last_err = e
            time.sleep(0.5)

    proc.terminate()
    raise RuntimeError(
        f"judge server did not become healthy within {wait_timeout:.0f}s "
        f"(last error: {last_err})"
        + (f" (log: {log_path})" if log_path else "")
    )
