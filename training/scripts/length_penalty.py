"""Length-penalty reward shaping.

Two shapes available, selected by `shape`:

  - "additive" (tolerance-band): penalty = α × max(0, |r-1| - tol).
    Zero inside the band, linear outside. Original design; risks small
    in-band drift because no gradient toward ratio=1 inside the band.

  - "quadratic" (default): penalty = α × (r - 1)².
    Gentle in-band (small quadratic nudges toward r=1), steep outside.
    Matches "gentle within 10%, punish hard outside" intuition:
      α=100: r=0.95 → 0.25; r=0.9 → 1.0; r=1.3 → 9; r=1.5 → 25.

`r = generated_wc / max(target_wc, 1)`.
"""
from __future__ import annotations


def compute_length_penalty(
    generated_wc: int,
    target_wc: int,
    alpha: float = 100.0,
    tol: float = 0.10,
    shape: str = "quadratic",
) -> float:
    if target_wc <= 0:
        return 0.0
    r = generated_wc / max(target_wc, 1)
    if shape == "additive":
        deviation = abs(r - 1) - tol
        return alpha * max(0.0, deviation)
    elif shape == "quadratic":
        return alpha * (r - 1.0) ** 2
    else:
        raise ValueError(f"unknown penalty shape: {shape}")


def word_count(text: str) -> int:
    return len(text.split())


def apply_length_penalty(
    rewrites, target_wcs, base_rewards,
    alpha: float = 100.0, tol: float = 0.10, shape: str = "quadratic",
):
    penalties = []
    penalised = []
    for rw, tgt, r in zip(rewrites, target_wcs, base_rewards):
        wc = word_count(rw)
        p = compute_length_penalty(wc, tgt, alpha=alpha, tol=tol, shape=shape)
        penalties.append(p)
        penalised.append(r - p)
    return penalised, penalties
