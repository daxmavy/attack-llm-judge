"""Single source of truth for rewriter + judge HF model IDs and fold rotation.

Any script in this repo that needs to know which rewriter or which judges to
run MUST import these names from here, not re-declare them locally. The prior
mission (2026-04-21) failed in part because `run_pilot_len_pen.py`,
`run_mission_attacks.py`, `run_icir.py`, `judge/vllm_client.py`, shell
orchestrators, and HF-push scripts each kept their own drifted copy — the
checkpoint that got pushed to HF advertised a judge panel different from the
one actually used at training time.

## How to update

1. Set `REWRITER` to the canonical HF id of the rewriter base (string).
2. Populate `JUDGE_REGISTRY` with `{slug: (wandb_name, hf_id)}` tuples. Slugs
   are opaque identifiers used in the DB (`attack_judge_scores.judge_slug`)
   and in the rotation table below; keep them stable once a run is committed
   to the DB.
3. Edit `FOLDS` to match the in-panel/held-out rotation you want across the
   three folds. The three keys in `FOLDS[fold]` must all exist in
   `JUDGE_REGISTRY`.
4. Leave `require_config()` alone — every entrypoint calls it to fail fast
   with a clear error if the placeholder values haven't been replaced.

## Placeholder state

`REWRITER` and `JUDGE_REGISTRY` ship as `None`/`{}`. Scripts import them
cleanly (so the module graph is intact) but `require_config()` raises at
`main()` entry until real values are filled in. This is deliberate — after
the 2026-04-21 post-mortem Max is picking the new panel by hand; we do NOT
want stale defaults to silently resurrect the old choice.
"""
from __future__ import annotations

from typing import Tuple


# -----------------------------------------------------------------------------
# Rewriter base model
# -----------------------------------------------------------------------------
# Set this to the HF repo id of the rewriter base (e.g. "Qwen/Qwen3-14B").
# Scripts that need the GRPO policy base / the inference rewriter read this.
#
# 2026-04-22 selection: Qwen3.5-9B dense (Unsloth docs confirm bf16 LoRA fits
# in ~22GB on A100 80GB; no QLoRA because Unsloth explicitly warns against
# 4-bit on Qwen3.5 variants). We use the upstream `Qwen/Qwen3.5-9B` repo
# (weights already cached under /data/shil6647/attack-llm-judge/hf_cache);
# Unsloth's reupload differs only by a ~60-byte chat-template patch, so the
# upstream repo is fine for our GRPO + HF-generate rollout path.
REWRITER: str | None = "Qwen/Qwen3.5-9B"


# -----------------------------------------------------------------------------
# Judge panel
# -----------------------------------------------------------------------------
# Map: slug -> (wandb_name, hf_id). Three entries expected — two of them are
# in-panel training judges per fold, the third is held out (see FOLDS below).
#
# - slug: short identifier used in DB rows + fold rotation. Keep stable.
# - wandb_name: the column name used in wandb reward/delta logs; also written
#   to `attack_judge_scores.judge_slug`. By convention `judge_<slug>`.
# - hf_id: the canonical HF repo id used with vLLM + HF tokenizer.
#
# Example shape once populated:
#     JUDGE_REGISTRY = {
#         "judgeA": ("judge_judgeA", "nvidia/Llama-3.3-Nemotron-Super-49B-v1.5"),
#         "judgeB": ("judge_judgeB", "google/gemma-3-27b-it"),
#         "judgeC": ("judge_judgeC", "meta-llama/Llama-3.1-8B-Instruct"),
#     }
JUDGE_REGISTRY: dict[str, Tuple[str, str]] = {}


# -----------------------------------------------------------------------------
# Fold rotation
# -----------------------------------------------------------------------------
# For each fold, which judges are in-panel (reward signal during GRPO, BoN
# panel scoring, ICIR feedback) vs held out (post-hoc transfer eval).
# Every slug here MUST be a key in JUDGE_REGISTRY.
FOLDS: dict[int, dict[str, object]] = {
    1: {"in_panel": ["judgeA", "judgeB"], "held_out": "judgeC"},
    2: {"in_panel": ["judgeA", "judgeC"], "held_out": "judgeB"},
    3: {"in_panel": ["judgeB", "judgeC"], "held_out": "judgeA"},
}


def require_config() -> None:
    """Fail fast at entrypoint if the placeholders haven't been filled.

    Call this at the top of every `main()` that will actually launch a model
    load. This prevents the 2026-04-21 failure mode where a stale hardcoded
    default silently ran instead of the intended panel.
    """
    errors: list[str] = []
    if not REWRITER:
        errors.append("REWRITER is None — set it to the rewriter HF repo id.")
    if not JUDGE_REGISTRY:
        errors.append("JUDGE_REGISTRY is empty — populate with {slug: (name, hf_id)}.")
    else:
        for fold, spec in FOLDS.items():
            for slug in list(spec["in_panel"]) + [spec["held_out"]]:
                if slug not in JUDGE_REGISTRY:
                    errors.append(
                        f"FOLDS[{fold}] references slug '{slug}' that is not "
                        f"in JUDGE_REGISTRY."
                    )
    if errors:
        banner = "=" * 72
        msg = (
            f"\n{banner}\nconfig/models.py is not fully populated:\n  - "
            + "\n  - ".join(errors)
            + f"\nEdit /home/shil6647/attack-llm-judge/config/models.py and rerun.\n{banner}"
        )
        raise SystemExit(msg)


def rewriter_short_name(model_id: str | None = None) -> str:
    """Stable short tag used in rewrite_id prefixes and HF repo names.

    Derived from the HF id by substring match so one registry update propagates
    to every downstream filename automatically. Falls back to a sanitised form
    of the HF id if no substring matches.
    """
    s = (model_id or REWRITER or "").lower()
    if not s:
        return "unknown"
    # Extend this table as new rewriters are adopted. Order matters — longer
    # / more-specific substrings first.
    table = [
        ("qwen3.5-35b-a3b", "qwen35-35b-a3b"),
        ("qwen3.5-27b", "qwen35-27b"),
        ("qwen3.5-9b", "qwen35-9b"),
        ("qwen3.5-4b", "qwen35-4b"),
        ("qwen3-32b", "qwen3-32b"),
        ("qwen3-14b", "qwen3-14b"),
        ("qwen2.5-1.5b", "qwen25-15b"),
        ("qwen2.5-7b", "qwen25-7b"),
        ("lfm2.5-1.2b", "lfm25-12b"),
        ("gemma-3-1b", "gemma3-1b"),
        ("gemma-3-12b", "gemma3-12b"),
        ("mistral-small-3.2", "mistral-small-32"),
    ]
    for needle, tag in table:
        if needle in s:
            return tag
    # Sanitised fallback.
    return s.replace("/", "-").replace(".", "-").replace("_", "-")
