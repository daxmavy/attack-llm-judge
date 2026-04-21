"""Project full-corpus cost for the evaluation suite.

Accepts a target (how many writer paragraphs × how many methods). Prints
a breakdown of:
- rewriter cost (Qwen 2.5 72B via OpenRouter)
- attack-panel judging cost (5 judges × all paragraphs × N criteria)
- gold-panel judging cost (5 judges × all paragraphs × N criteria)
- total

Used to decide whether to run the gold eval on the full corpus, a
subsample, or with a cheaper gold panel. Run after the smoke test so
the per-call token averages are calibrated from actual runs.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import pandas as pd

from eval_suite.panels import ATTACK_PANEL, GOLD_PANEL
from eval_suite.schema import DEFAULT_DB_PATH, connect


@dataclass
class CostBreakdown:
    rewriter_usd: float
    attack_usd: float
    gold_usd: float
    total_usd: float


# Method -> expected rewriter calls per paragraph (from orchestrators.METHOD_SPEC).
# Kept in sync manually.
METHOD_CALLS_PP = {
    "naive": 1.3,
    "lit_informed": 1.3,
    "naive_tight": 1.3,
    "lit_informed_tight": 1.0,
    "rules_explicit": 1.3,
    "rubric_aware": 1.3,
    "scaffolded_cot_distill": 1.3,
    "icir_single": 13.0,
    "bon_panel": 10.0,
}

QWEN_IN_PER_1M = 0.13
QWEN_OUT_PER_1M = 0.40

# Per-call token averages (can be overridden). These match what we saw
# in the earlier baseline runs.
AVG_IN_REWRITER = 900
AVG_OUT_REWRITER = 200
AVG_IN_JUDGE = 700
AVG_OUT_JUDGE = 60


def project(n_paragraphs: int,
             methods: list[str],
             n_criteria: int = 1,
             include_originals: bool = True) -> dict:
    total_rewriter_calls = sum(METHOD_CALLS_PP[m] * n_paragraphs for m in methods)
    rewriter_cost = (total_rewriter_calls * AVG_IN_REWRITER / 1e6 * QWEN_IN_PER_1M
                     + total_rewriter_calls * AVG_OUT_REWRITER / 1e6 * QWEN_OUT_PER_1M)

    n_rewrites = len(methods) * n_paragraphs
    n_to_judge = n_rewrites + (n_paragraphs if include_originals else 0)

    def panel_cost(panel):
        per_judge_per_call = lambda j: (AVG_IN_JUDGE / 1e6 * j.prompt_in_per_1m_usd
                                        + AVG_OUT_JUDGE / 1e6 * j.prompt_out_per_1m_usd)
        total = 0.0
        for j in panel:
            total += per_judge_per_call(j) * n_to_judge * n_criteria
        return total

    attack_cost = panel_cost(ATTACK_PANEL)
    gold_cost = panel_cost(GOLD_PANEL)
    # BoN adds attack-panel judging inside its orchestrator (N_survivors × 5 × 1).
    # Approximate: if bon_panel is in methods, add ~40 panel calls per source.
    if "bon_panel" in methods:
        attack_cost += (40 * n_paragraphs) * (AVG_IN_JUDGE / 1e6 * 0.05 + AVG_OUT_JUDGE / 1e6 * 0.08)

    total = rewriter_cost + attack_cost + gold_cost
    return {
        "n_paragraphs": n_paragraphs,
        "methods": methods,
        "n_criteria": n_criteria,
        "include_originals": include_originals,
        "rewriter_calls_estimated": int(total_rewriter_calls),
        "rewriter_usd": rewriter_cost,
        "attack_usd": attack_cost,
        "gold_usd": gold_cost,
        "total_usd": total,
    }


def auto_project(db_path=DEFAULT_DB_PATH) -> dict:
    con = connect(db_path)
    try:
        n_writers = int(con.execute(
            "SELECT COUNT(*) FROM paragraphs WHERE origin_kind='original_writer'"
        ).fetchone()[0])
        n_all_originals = int(con.execute(
            "SELECT COUNT(*) FROM paragraphs WHERE origin_kind LIKE 'original_%'"
        ).fetchone()[0])
    finally:
        con.close()
    methods = list(METHOD_CALLS_PP.keys())
    out = {
        "full_corpus_writers_only": project(n_writers, methods, n_criteria=1, include_originals=True),
        "full_corpus_all_originals_and_rewrites": project(
            n_writers, methods, n_criteria=1, include_originals=False,
        ),
        "if_we_ALSO_judge_the_non_writer_originals": project(
            n_all_originals, methods=[], n_criteria=1, include_originals=True,
        ),
    }
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None, help="Override paragraph count")
    p.add_argument("--methods", default=",".join(METHOD_CALLS_PP.keys()))
    p.add_argument("--n-criteria", type=int, default=1)
    p.add_argument("--no-originals", action="store_true")
    args = p.parse_args()
    if args.n is None:
        print(json.dumps(auto_project(), indent=2))
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        print(json.dumps(project(args.n, methods, args.n_criteria, not args.no_originals),
                          indent=2))
