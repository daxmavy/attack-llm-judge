"""Length-penalty reward shaping. Option 1 tolerance-band additive.

reward = judge_ensemble_mean - alpha * max(0, |len_ratio - 1| - tol)

len_ratio = generated_word_count / original_word_count

Defaults: tol=0.10 (±10% free band), alpha=25 (a 40% deviation = -7.5 reward).
"""
from __future__ import annotations


def compute_length_penalty(generated_wc: int, target_wc: int, alpha: float = 25.0, tol: float = 0.10) -> float:
    if target_wc <= 0:
        return 0.0
    r = generated_wc / max(target_wc, 1)
    deviation = abs(r - 1) - tol
    return alpha * max(0.0, deviation)


def word_count(text: str) -> int:
    """Plain whitespace-splitting word counter (matches paul_data convention)."""
    return len(text.split())


def apply_length_penalty(rewrites, target_wcs, base_rewards, alpha: float = 25.0, tol: float = 0.10):
    """Vectorised: return penalised rewards + per-item penalty list (for logging)."""
    penalties = []
    penalised = []
    for rw, tgt, r in zip(rewrites, target_wcs, base_rewards):
        wc = word_count(rw)
        p = compute_length_penalty(wc, tgt, alpha=alpha, tol=tol)
        penalties.append(p)
        penalised.append(r - p)
    return penalised, penalties
