"""Score attack_rewrites via OpenRouter API.

Used for high-quality / out-of-sample judges that aren't run locally via vLLM.
Async-parallel for throughput. Idempotent — skips rows already scored
(rewrite_id, judge_slug, criterion).

Usage:
  python3 scripts/score_via_openrouter.py \\
    --judges deepseek/deepseek-v4-flash minimax/minimax-m2.7 \\
    --methods naive grpo_nli_400step \\
    --rewriters Qwen/Qwen2.5-1.5B-Instruct LiquidAI/LFM2.5-1.2B-Instruct google/gemma-3-1b-it \\
    --criterion clarity

Score-row shape mirrors local-vLLM scoring:
  judge_slug = 'judge_' + last-segment of model id, dashes/dots → underscore.
  criterion = the rubric used (independent of the rewrite's own criterion).
  score = 0-100 float parsed from the JSON judge response.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import time

sys.path.insert(0, "/workspace/grpo_run")
from run_pilot_len_pen import JUDGE_SYSTEM, RUBRICS, parse_score

DB = "/home/max/attack-llm-judge/data/paragraphs.db"

# Default model_id → judge_slug mapping (matches the convention used elsewhere).
def slug_for(model_id: str) -> str:
    last = model_id.split("/")[-1]
    return "judge_" + re.sub(r"[^a-z0-9]", "_", last.lower())


async def score_one(client, model_id, system, user, semaphore, max_retries=3):
    """One API call. Returns parsed score float or None on persistent failure."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=1500,  # accommodates reasoning models (MiniMax M2.7) which fill .reasoning before .content
                    extra_headers={"HTTP-Referer": "https://github.com/daxmavy/attack-llm-judge",
                                    "X-Title": "attack-llm-judge"},
                )
                text = resp.choices[0].message.content or ""
                score = parse_score(text)
                return score, text
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  [warn] {model_id} failed after {max_retries} retries: {e!r}", flush=True)
                    return None, None
                await asyncio.sleep(2 ** attempt)
        return None, None


async def main_async(args):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY_EXPERIMENTS"],
    )

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Build the universe of (rewrite_id, proposition, text) tuples to score.
    placeholders_methods = ",".join("?" * len(args.methods))
    placeholders_rewriters = ",".join("?" * len(args.rewriters))
    rows = cur.execute(f"""
        SELECT r.rewrite_id, p.proposition, r.text
        FROM attack_rewrites r
        JOIN paragraphs p ON r.source_doc_id = p.document_id
        WHERE r.method IN ({placeholders_methods})
          AND r.rewriter_model IN ({placeholders_rewriters})
    """, list(args.methods) + list(args.rewriters)).fetchall()
    print(f"[{time.strftime('%H:%M:%S')}] universe: {len(rows)} rewrites", flush=True)

    rubric_template = RUBRICS[args.criterion]
    sem = asyncio.Semaphore(args.concurrency)

    for model_id in args.judges:
        slug = slug_for(model_id)
        print(f"\n[{time.strftime('%H:%M:%S')}] === {model_id} → {slug} × {args.criterion} ===", flush=True)

        # Skip rows already scored
        scored = {r[0] for r in cur.execute(
            "SELECT rewrite_id FROM attack_judge_scores WHERE judge_slug=? AND criterion=?",
            (slug, args.criterion)).fetchall()}
        todo = [r for r in rows if r[0] not in scored]
        print(f"  {len(scored)} already scored, {len(todo)} remaining", flush=True)
        if not todo:
            continue

        t0 = time.time()
        # Stream results in chunks for incremental DB commits + progress
        CHUNK = max(1, args.concurrency * 4)
        n_done = 0
        n_failed = 0
        for chunk_start in range(0, len(todo), CHUNK):
            chunk = todo[chunk_start:chunk_start + CHUNK]
            tasks = []
            for rid, prop, text in chunk:
                user_msg = rubric_template.format(proposition=prop, paragraph=text)
                tasks.append(score_one(client, model_id, JUDGE_SYSTEM, user_msg, sem))
            results = await asyncio.gather(*tasks)

            inserts = []
            for (rid, _, _), (score, raw) in zip(chunk, results):
                if score is None:
                    n_failed += 1
                    continue
                inserts.append((rid, slug, args.criterion, float(score),
                                (raw[:2000] if raw else None)))
            cur.executemany("""INSERT OR IGNORE INTO attack_judge_scores
                               (rewrite_id, judge_slug, criterion, score, reasoning)
                               VALUES (?, ?, ?, ?, ?)""", inserts)
            conn.commit()
            n_done += len(chunk)
            rate = n_done / max(1e-6, time.time() - t0)
            eta = (len(todo) - n_done) / max(1e-6, rate) / 60
            print(f"  {n_done}/{len(todo)}  rate={rate:.1f}/s  failed={n_failed}  eta={eta:.1f}min",
                  flush=True)

        print(f"[{time.strftime('%H:%M:%S')}] {model_id} done in {(time.time()-t0)/60:.1f} min  "
              f"({n_failed} failed)", flush=True)

    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", required=True,
                    help="OpenRouter model IDs, e.g. deepseek/deepseek-v4-flash")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--rewriters", nargs="+", required=True)
    ap.add_argument("--criterion", required=True, choices=list(RUBRICS.keys()))
    ap.add_argument("--concurrency", type=int, default=40,
                    help="parallel in-flight requests per judge")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
