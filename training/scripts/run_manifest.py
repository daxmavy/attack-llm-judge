"""Run manifest helper. Captures everything needed to reproduce a training run.

Usage at top of any training script:
    from run_manifest import capture_manifest
    manifest = capture_manifest(
        run_name="pilot_drgrpo_b0.001",
        script_path=__file__,
        grpo_config=cfg,  # GRPOConfig or dict
        extra={"rewriter": REWRITER, "judges": [...], ...},
        out_dir="/data/shil6647/attack-llm-judge/grpo_run/pilot_...",
    )
    wandb.config.update({"manifest": manifest}, allow_val_change=True)

Captured:
- All GRPOConfig hyperparameters (every field, whether explicitly set or default)
- Script file hash (SHA256)
- Git SHA + dirty flag (if available)
- Python + key package versions
- GPU info
- Data query hash (optional, via `data_query` in extra)
- Timestamp
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path


def _file_sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"ERROR: {e}"


def _git_info(repo_dir="/home/shil6647/attack-llm-judge"):
    try:
        sha = subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", repo_dir, "status", "--porcelain"], stderr=subprocess.DEVNULL).strip())
        branch = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
        return {"sha": sha, "dirty": dirty, "branch": branch, "repo": repo_dir}
    except Exception as e:
        return {"error": str(e), "repo": repo_dir}


def _pkg_versions():
    out = {}
    for pkg in ("torch", "transformers", "trl", "vllm", "accelerate", "datasets", "wandb",
                "numpy", "safetensors", "flash_attn", "xformers"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", "?")
        except Exception:
            out[pkg] = None
    out["python"] = sys.version.split()[0]
    return out


def _gpu_info():
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode().strip()
        return q
    except Exception as e:
        return f"ERROR: {e}"


def _grpo_config_dict(cfg):
    """Extract every field from a GRPOConfig dataclass, including defaults."""
    if cfg is None:
        return {}
    if is_dataclass(cfg):
        # asdict recursively; but some TrainingArguments fields are not JSON-safe
        try:
            d = asdict(cfg)
        except Exception:
            d = {k: getattr(cfg, k, None) for k in dir(cfg)
                 if not k.startswith("_") and not callable(getattr(cfg, k, None))}
    elif isinstance(cfg, dict):
        d = dict(cfg)
    else:
        d = {k: getattr(cfg, k, None) for k in dir(cfg)
             if not k.startswith("_") and not callable(getattr(cfg, k, None))}
    # Drop non-JSON-safe values
    clean = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            clean[k] = v
        except Exception:
            clean[k] = repr(v)[:200]
    return clean


def capture_manifest(run_name, script_path, grpo_config=None, extra=None, out_dir=None):
    """Build a manifest dict, write to disk, return dict (for wandb logging)."""
    manifest = {
        "run_name": run_name,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "script": {
            "path": str(script_path),
            "sha256": _file_sha256(script_path),
            "mtime": datetime.utcfromtimestamp(
                os.path.getmtime(script_path)).isoformat(timespec="seconds") + "Z"
                if os.path.exists(script_path) else None,
        },
        "git": _git_info(),
        "packages": _pkg_versions(),
        "gpu": _gpu_info(),
        "grpo_config": _grpo_config_dict(grpo_config),
        "extra": extra or {},
    }
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


if __name__ == "__main__":
    # Smoke test
    m = capture_manifest("smoke", __file__, grpo_config=None, extra={"test": True})
    print(json.dumps(m, indent=2, default=str))
