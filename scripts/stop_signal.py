"""Graceful-stop primitives for long-running GRPO and attack-batch scripts.

Both the training script (`training/scripts/run_pilot_len_pen.py`) and the
attack-batch scripts (`scripts/run_mission_attacks.py`, `scripts/run_icir.py`)
poll a filesystem marker between natural checkpoint boundaries (a training
step, a per-method vLLM call, a per-fold × criterion ICIR loop). When the
marker is present the job commits its progress and exits 0 so the GPUs are
freed without losing work.

Usage (operator side):

    # while a run is in flight, stop it cleanly:
    touch <stop_file>

The running script prints its stop-file path at startup; the path lives
under the run's output dir (e.g. `/workspace/grpo_run/pilot_<suffix>/STOP`)
so it is per-run and survives restarts. Each script clears any stale STOP
on launch so a marker from a previous aborted run doesn't fire immediately.

Resume contract:
- GRPO: re-run the same command with `--resume-from <ckpt>`; pass
  `--skip-pre-eval` to skip the pre-eval if a `pre_eval.json` was already
  persisted.
- Attack scripts: re-run the same command with `--resume-skip-existing`.
  The DB's INSERT OR REPLACE makes re-runs safe; skip-existing avoids
  redoing completed work.
"""
from __future__ import annotations

import os
import time
from pathlib import Path


def default_stop_file(run_dir: str | os.PathLike) -> str:
    """Canonical stop-file path for a run. Callers should ensure the parent dir exists."""
    return str(Path(run_dir) / "STOP")


def clear_stop_file(stop_file: str | os.PathLike) -> None:
    """Remove any stale STOP file at script start."""
    p = Path(stop_file)
    if p.exists():
        p.unlink()


def check_stop_signal(stop_file: str | os.PathLike) -> bool:
    return Path(stop_file).exists()


def announce(stop_file: str | os.PathLike) -> None:
    """Print the stop-file location so operators can find it from the log tail."""
    print(f"[stop-signal] graceful-stop file: {stop_file}", flush=True)
    print(f"[stop-signal]   `touch {stop_file}` to save + exit at the next boundary.", flush=True)


def build_trainer_callback(stop_file: str):
    """Return an HF TrainerCallback that triggers save + stop on STOP-file presence."""
    from transformers.trainer_callback import TrainerCallback

    class StopSignalCallback(TrainerCallback):
        def __init__(self, path: str):
            self.path = path
            self._fired = False

        def on_step_end(self, args, state, control, **kwargs):
            if self._fired:
                return control
            if Path(self.path).exists():
                self._fired = True
                print(
                    f"[{time.strftime('%H:%M:%S')}] [STOP] detected {self.path} — "
                    f"saving checkpoint and exiting after step {state.global_step}.",
                    flush=True,
                )
                control.should_save = True
                control.should_training_stop = True
            return control

    return StopSignalCallback(stop_file)
