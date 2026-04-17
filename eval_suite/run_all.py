"""Master driver: rewrite + judge + metric on the stratified sample.

For a given sample tag and criterion:
1. Generate rewrites for every method in METHOD_SPEC on the sampled writers.
2. Attack-panel judging on (sampled writers + their rewrites).
3. Gold-panel judging on same set.
4. Word-count, embedding, hallucinated-specifics metrics (free).
5. Agreement-score, clarity-regressor predictions (DeBERTa, GPU).
6. Binoculars (if available and --include-detector).

Per-step cost printed so we can halt early if something's going wrong.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from eval_suite.judge_runner import run_panel
from eval_suite.metrics import embedding, hallucinated_specifics, word_count
from eval_suite.metrics import deberta_regressor
from eval_suite.rewrite_runner import METHOD_SPEC, run_method
from eval_suite.sampling import get_sample_ids
from eval_suite.schema import DEFAULT_DB_PATH, connect


def scope_where(sample_tag: str) -> str:
    """SQL WHERE clause that selects (a) the sampled writer originals AND
    (b) their rewrites. Used by the judge runner."""
    return (
        "(origin_kind='original_writer' AND document_id IN (SELECT document_id FROM sampled_writers WHERE tag='"
        + sample_tag + "'))"
        " OR (origin_kind='rewrite' AND base_document_id IN (SELECT document_id FROM sampled_writers WHERE tag='"
        + sample_tag + "'))"
    )


def run(sample_tag: str, criterion: str = "clarity",
         methods: list[str] | None = None,
         skip_rewrites: bool = False,
         skip_attack: bool = False,
         skip_gold: bool = False,
         skip_metrics: bool = False,
         skip_regressors: bool = False,
         max_workers: int = 6,
         db_path: Path = DEFAULT_DB_PATH) -> dict:
    methods = methods or list(METHOD_SPEC.keys())
    ids = get_sample_ids(sample_tag, db_path)
    print(f"[run_all] sample_tag={sample_tag} size={len(ids)} methods={methods}")
    t_start = time.time()
    report: dict = {"sample_tag": sample_tag, "criterion": criterion}

    if not skip_rewrites:
        report["rewrites"] = {}
        for m in methods:
            t0 = time.time()
            stats = run_method(m, origin_kind="original_writer",
                                sample_tag=sample_tag, max_workers=max_workers)
            stats["seconds"] = round(time.time() - t0, 1)
            report["rewrites"][m] = stats
            print(f"[rewrite:{m}] done={stats.get('n_done')} "
                  f"cost=${stats.get('cost_rewriter_usd', 0):.4f} "
                  f"secs={stats['seconds']}")

    where = scope_where(sample_tag)
    if not skip_attack:
        t0 = time.time()
        rp = run_panel(criterion, "attack", where=where, max_workers=max_workers)
        rp["seconds"] = round(time.time() - t0, 1)
        report["attack_judging"] = rp
        print(f"[attack] {rp}")
    if not skip_gold:
        t0 = time.time()
        rp = run_panel(criterion, "gold", where=where, max_workers=max_workers)
        rp["seconds"] = round(time.time() - t0, 1)
        report["gold_judging"] = rp
        print(f"[gold] {rp}")

    if not skip_metrics:
        report["metrics"] = {}
        report["metrics"]["word_count"] = word_count.score_all(db_path)
        print(f"[metric] word_count rows={report['metrics']['word_count']}")
        try:
            report["metrics"]["embed_sim"] = embedding.score_rewrites(db_path=db_path)
        except Exception as e:
            report["metrics"]["embed_sim_error"] = str(e)
            print(f"[metric] embed_sim skipped: {e}")
        try:
            report["metrics"]["hallucinated_specifics"] = hallucinated_specifics.score_rewrites(db_path=db_path)
        except Exception as e:
            report["metrics"]["hallucinated_specifics_error"] = str(e)
            print(f"[metric] hallucinated_specifics skipped: {e}")
        print(f"[metrics] {report['metrics']}")

    if not skip_regressors:
        # Scope to sample scope only (originals + rewrites in the subsample).
        scope_sql = f"""document_id IN (
            SELECT document_id FROM paragraphs
            WHERE (origin_kind='original_writer' AND document_id IN
                   (SELECT document_id FROM sampled_writers WHERE tag='{sample_tag}'))
               OR (origin_kind='rewrite' AND base_document_id IN
                   (SELECT document_id FROM sampled_writers WHERE tag='{sample_tag}'))
        )"""
        try:
            n_agree = deberta_regressor.score_agreement(db_path=db_path, where=scope_sql)
        except Exception as e:
            n_agree = -1
            print(f"[regressor] agreement failed: {e}")
        try:
            n_clarity = deberta_regressor.score_clarity(db_path=db_path, where=scope_sql)
        except Exception as e:
            n_clarity = -1
            print(f"[regressor] clarity failed: {e}")
        report["regressors"] = {"agreement_rows": n_agree, "clarity_rows": n_clarity}

    report["total_seconds"] = round(time.time() - t_start, 1)
    out_path = Path(db_path).parent / f"run_report_{sample_tag}_{int(t_start)}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[run_all] wrote {out_path}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample-tag", required=True)
    p.add_argument("--criterion", default="clarity")
    p.add_argument("--methods", default=None)
    p.add_argument("--max-workers", type=int, default=6)
    p.add_argument("--skip-rewrites", action="store_true")
    p.add_argument("--skip-attack", action="store_true")
    p.add_argument("--skip-gold", action="store_true")
    p.add_argument("--skip-metrics", action="store_true")
    p.add_argument("--skip-regressors", action="store_true")
    args = p.parse_args()
    methods = args.methods.split(",") if args.methods else None
    print(json.dumps(run(
        sample_tag=args.sample_tag,
        criterion=args.criterion,
        methods=methods,
        skip_rewrites=args.skip_rewrites,
        skip_attack=args.skip_attack,
        skip_gold=args.skip_gold,
        skip_metrics=args.skip_metrics,
        skip_regressors=args.skip_regressors,
        max_workers=args.max_workers,
    ), indent=2))
