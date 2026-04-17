"""Orchestrators for the non-trivial rewriter methods (ICIR, BoN, scaffolded).

Simple methods (naive, lit_informed, naive_tight, lit_informed_tight,
injection_leadin, rules_explicit, rubric_aware) are one call + at most
one length retry; the existing `do_rewrites` loop in `run_rewrite.py`
handles them.

The three orchestrators here return the same contract: a dict with
`text`, `ok`, `error`, token counts, a list of intermediate artefacts
(for audit), and method-specific metadata.
"""
from __future__ import annotations

import concurrent.futures as cf
from typing import Callable, Iterable

from judge.client import call_judge
from judge.rubrics import SYSTEM_PROMPT as JUDGE_SYSTEM, build_prompt
from rewriters.parsing import extract_json
from rewriters.panel_scorer import score_with_panel
from rewriters.rewrite_prompts import (
    SYSTEM_PROMPT as REWRITER_SYSTEM,
    build_retry_prompt,
    build_rewrite_prompt,
    length_bounds,
)
from rewriters.rewriter_client import REWRITER_MODEL, call_rewriter


TIGHT_TOL = 0.10


def _in_range(text: str, lo: int, hi: int) -> bool:
    return lo <= len(text.split()) <= hi


def _one_shot_with_retry(method: str, proposition: str, paragraph: str, api_key: str,
                         temperature: float, **kwargs) -> tuple:
    """Runs one rewrite + at most one length-retry. Returns (final RewriteResult, retry_flag)."""
    user = build_rewrite_prompt(method, proposition, paragraph, **kwargs)
    r = call_rewriter(REWRITER_SYSTEM, user, api_key, model_id=REWRITER_MODEL,
                      max_tokens=400, temperature=temperature)
    if not r.ok or not r.text:
        return r, False
    _, lo, hi = length_bounds(paragraph, TIGHT_TOL)
    if _in_range(r.text, lo, hi):
        return r, False
    # one retry
    retry_user = build_retry_prompt(method, proposition, paragraph, r.text)
    r2 = call_rewriter(REWRITER_SYSTEM, retry_user, api_key, model_id=REWRITER_MODEL,
                       max_tokens=400, temperature=max(0.3, temperature * 0.7))
    if r2.ok and r2.text:
        def _miss(s):
            w = len(s.split())
            return max(0, lo - w) + max(0, w - hi)
        if _miss(r2.text) <= _miss(r.text):
            return r2, True
    return r, True


def run_simple(method: str, proposition: str, paragraph: str, api_key: str,
                temperature: float = 0.7, leadin_variant: int | None = None) -> dict:
    r, retried = _one_shot_with_retry(method, proposition, paragraph, api_key,
                                       temperature, leadin_variant=leadin_variant)
    return {
        "method": method,
        "text": r.text,
        "ok": r.ok,
        "error": r.error,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "retried": retried,
        "calls": 1 + int(retried),
    }


def run_injection_leadin(proposition: str, paragraph: str, api_key: str,
                           rotation_index: int) -> dict:
    """Cycle A/B/C lead-in variants round-robin."""
    return run_simple("injection_leadin", proposition, paragraph, api_key,
                      temperature=0.4, leadin_variant=rotation_index)


def run_scaffolded(proposition: str, paragraph: str, api_key: str) -> dict:
    """JSON-structured plan/draft/critique/final. Parse robustly; one-shot retry if needed."""
    user = build_rewrite_prompt("scaffolded_cot_distill", proposition, paragraph)
    r = call_rewriter(REWRITER_SYSTEM, user, api_key, model_id=REWRITER_MODEL,
                      max_tokens=800, temperature=0.7)
    parsed = extract_json(r.text) if r.text else None
    final_text = parsed.get("final") if parsed else None
    _, lo, hi = length_bounds(paragraph, TIGHT_TOL)
    retried = False
    if not final_text or not _in_range(final_text, lo, hi):
        # Retry with a plain tight rewrite as the fallback "final".
        r2, _ = _one_shot_with_retry("lit_informed_tight", proposition, paragraph, api_key,
                                      temperature=0.5)
        if r2.ok and r2.text:
            final_text = r2.text
            retried = True
    return {
        "method": "scaffolded_cot_distill",
        "text": final_text,
        "ok": bool(final_text),
        "error": None if final_text else "scaffold parse/retry failed",
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "retried": retried,
        "calls": 1 + int(retried),
        "artefacts": parsed,
    }


def run_icir_single(proposition: str, paragraph: str, api_key: str,
                     feedback_judge_slug: str = "qwen-2.5-7b",
                     n_max: int = 4, improve_threshold: float = 2.0,
                     stop_score: float = 90.0) -> dict:
    """
    Iteration 0: seed via lit_informed_tight.
    Iterations 1..n_max: query the feedback judge (clarity only for now),
    then call the rewriter with the iterative-refinement prompt. Stop
    early if score plateaus or passes stop_score.
    """
    from eval_suite.panels import by_slug
    judge = by_slug(feedback_judge_slug)

    # seed
    seed, retried0 = _one_shot_with_retry("icir_single", proposition, paragraph, api_key,
                                           temperature=0.5)
    if not seed.ok or not seed.text:
        return {"method": "icir_single", "text": None, "ok": False,
                "error": seed.error, "prompt_tokens": seed.prompt_tokens,
                "completion_tokens": seed.completion_tokens, "retried": retried0,
                "calls": 1 + int(retried0), "iterations": 0}

    rewrites = [seed.text]
    scores = []
    reasonings = []
    total_p = seed.prompt_tokens or 0
    total_c = seed.completion_tokens or 0
    calls = 1 + int(retried0)

    prev_score = None
    best_idx = 0
    best_score = -1.0

    for t in range(n_max + 1):  # +1 so we also score the last rewrite
        # Score current rewrite
        prompt = build_prompt("clarity", proposition, rewrites[-1])
        jr = call_judge(judge.model_id, JUDGE_SYSTEM, prompt, api_key,
                        max_tokens=250, temperature=0.0)
        calls += 1
        total_p += jr.prompt_tokens or 0
        total_c += jr.completion_tokens or 0
        s = float(jr.score) if (jr.ok and jr.score is not None) else 0.0
        scores.append(s)
        reasonings.append(jr.reasoning or "")
        if s > best_score:
            best_score, best_idx = s, len(rewrites) - 1
        # stopping
        if s >= stop_score:
            break
        if prev_score is not None and (s - prev_score) < improve_threshold:
            break
        if t == n_max:
            break
        # propose next
        ctx = {"prev_rewrite": rewrites[-1], "prev_clarity": int(round(s)),
               "judge_reasoning": reasonings[-1]}
        r_new, retried = _one_shot_with_retry(
            "icir_single", proposition, paragraph, api_key, temperature=0.5,
            icir_context=ctx,
        )
        calls += 1 + int(retried)
        total_p += r_new.prompt_tokens or 0
        total_c += r_new.completion_tokens or 0
        if not r_new.ok or not r_new.text:
            break
        rewrites.append(r_new.text)
        prev_score = s

    best = rewrites[best_idx]
    return {
        "method": "icir_single",
        "text": best,
        "ok": best is not None,
        "error": None,
        "prompt_tokens": total_p,
        "completion_tokens": total_c,
        "retried": False,
        "calls": calls,
        "iterations": len(rewrites),
        "per_iter_scores": scores,
        "feedback_judge": feedback_judge_slug,
    }


def run_bon_panel(proposition: str, paragraph: str, api_key: str,
                    n_samples: int = 8, temperature: float = 1.0) -> dict:
    """Sample n_samples candidates, score each with attack panel, pick argmax."""
    # sample in parallel
    def _sample(_):
        r, retried = _one_shot_with_retry("bon_panel", proposition, paragraph, api_key,
                                           temperature=temperature)
        return r, retried
    with cf.ThreadPoolExecutor(max_workers=min(n_samples, 8)) as ex:
        results = list(ex.map(_sample, range(n_samples)))
    candidates = []
    total_p = 0
    total_c = 0
    calls = 0
    for r, retried in results:
        total_p += r.prompt_tokens or 0
        total_c += r.completion_tokens or 0
        calls += 1 + int(retried)
        if r.ok and r.text:
            candidates.append(r.text)
    _, lo, hi = length_bounds(paragraph, TIGHT_TOL)
    candidates = [c for c in candidates if _in_range(c, lo, hi) and "\n\n" not in c.strip()]
    if not candidates:
        return {"method": "bon_panel", "text": None, "ok": False,
                "error": "no in-range candidates", "prompt_tokens": total_p,
                "completion_tokens": total_c, "retried": False, "calls": calls,
                "n_candidates": 0, "per_candidate_scores": []}

    # score each via attack panel
    per_scores = []
    for cand in candidates:
        scores = score_with_panel(cand, proposition, api_key,
                                   panel_name="attack", criteria=("clarity",))
        per_scores.append(scores)
        calls += 5  # 5 attack judges * 1 criterion

    # aggregate: mean_over_judges(score) (only clarity for now)
    def _agg(score_dict):
        vals = [score_dict[j]["clarity"]["score"] for j in score_dict
                if score_dict[j]["clarity"]["score"] is not None]
        if not vals:
            return -1.0, 0.0
        import statistics as st
        return float(sum(vals) / len(vals)), (float(st.pstdev(vals)) if len(vals) > 1 else 0.0)
    agg = [_agg(s) for s in per_scores]
    # pick argmax of mean; tie-break by lowest variance; then fewest words.
    idx = sorted(range(len(candidates)), key=lambda i: (-agg[i][0], agg[i][1],
                                                         len(candidates[i].split())))[0]
    return {
        "method": "bon_panel",
        "text": candidates[idx],
        "ok": True,
        "error": None,
        "prompt_tokens": total_p,  # judge tokens not tracked here; budget dwarfs rewriter
        "completion_tokens": total_c,
        "retried": False,
        "calls": calls,
        "n_candidates": len(candidates),
        "chosen_mean_score": agg[idx][0],
        "chosen_score_std": agg[idx][1],
    }
