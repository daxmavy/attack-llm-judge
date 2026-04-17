"""Score a single rewrite against a panel of LLM judges, used by `bon_panel` and `icir_single`.

Returns a nested dict {judge_slug: {criterion: score_or_none}} plus
reasoning strings. Temperature is fixed at 0 so scoring is deterministic.
"""
from __future__ import annotations

import concurrent.futures as cf
from typing import Iterable

from eval_suite.panels import JudgeConfig, panel as get_panel
from judge.client import call_judge
from judge.rubrics import SYSTEM_PROMPT as JUDGE_SYSTEM, build_prompt


def score_with_panel(
    rewrite_text: str,
    proposition: str,
    api_key: str,
    judges: list[JudgeConfig] | None = None,
    panel_name: str = "attack",
    criteria: Iterable[str] = ("clarity",),
    max_workers: int = 5,
) -> dict:
    if judges is None:
        judges = get_panel(panel_name)

    tasks = []
    for j in judges:
        for c in criteria:
            tasks.append((j, c, build_prompt(c, proposition, rewrite_text)))

    def _go(task):
        j, crit, prompt = task
        r = call_judge(j.model_id, JUDGE_SYSTEM, prompt, api_key,
                       max_tokens=250, temperature=0.0)
        return (j.slug, crit, r)

    out: dict[str, dict[str, dict]] = {j.slug: {c: {"score": None, "reasoning": None}
                                                for c in criteria}
                                         for j in judges}
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for slug, crit, res in ex.map(_go, tasks):
            out[slug][crit] = {
                "score": res.score if res.ok else None,
                "reasoning": res.reasoning,
                "ok": res.ok,
            }
    return out
