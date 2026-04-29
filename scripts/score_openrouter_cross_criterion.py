"""Score every non-criterion-agnostic attack rewrite at its UNMATCHED
criterion (i.e. the criterion opposite to the one it was tagged with) on the
3-judge OR panel. Each rewrite has its native (own-crit) score already; this
fills in the cross-crit cell so we can decompose attack effects on a criterion
the rewriter did NOT target.

bon_candidate is excluded (per project convention — never OR-scored).
Criterion-agnostic methods (naive, original, original_ai) are excluded — they
already have both criteria scored.

Idempotent: skips (rewrite_id, judge_slug, opposite_criterion) tuples that
already exist in attack_judge_scores.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

ENV_FILE = "/home/max/attack-llm-judge/.env"
if os.path.exists(ENV_FILE):
    for line in open(ENV_FILE):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

sys.path.insert(0, "/workspace/grpo_run")
sys.path.insert(0, "/home/max/attack-llm-judge")

DB = "/home/max/attack-llm-judge/data/paragraphs.db"
STOP_FILE = Path("/tmp/openrouter_xcrit.STOP")
PROGRESS_FILE = Path("/tmp/openrouter_xcrit.progress.json")

# Re-use the canonical rubric + parsing
from run_pilot_len_pen import JUDGE_SYSTEM, RUBRICS, parse_score

CRITERION_AGNOSTIC = {"naive", "original", "original_ai"}

_stop_flag = {"reason": None}


def install_signal_handlers():
    def _h(sig, _f): _stop_flag["reason"] = f"signal_{sig}"
    signal.signal(signal.SIGTERM, _h); signal.signal(signal.SIGINT, _h)


def stop_requested():
    return _stop_flag["reason"] or ("stop_file" if STOP_FILE.exists() else None)


class QuotaError(Exception): pass


def slug_for(model_id: str) -> str:
    import re
    return "judge_" + re.sub(r"[^a-z0-9]", "_", model_id.split("/")[-1].lower())


def fetch_universe(conn):
    """Build (rewrite_id, proposition, text, opposite_criterion) tuples for every
    non-bon_candidate, non-criterion-agnostic rewrite."""
    rows = conn.execute("""
        SELECT r.rewrite_id, p.proposition, r.text, r.criterion
        FROM attack_rewrites r
        JOIN paragraphs p ON r.source_doc_id = p.document_id
        WHERE r.method != 'bon_candidate'
          AND r.method NOT IN ('naive', 'original', 'original_ai')
    """).fetchall()
    out = []
    for rid, prop, text, own_crit in rows:
        opposite = "informativeness" if own_crit == "clarity" else "clarity"
        out.append((rid, prop, text, opposite))
    return out


async def score_one(client, model_id, system_msg, user_msg, sem):
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=1500,
                    extra_body={"reasoning": {"enabled": False}},
                    extra_headers={"HTTP-Referer": "https://github.com/daxmavy/attack-llm-judge",
                                   "X-Title": "attack-llm-judge"},
                )
                text = resp.choices[0].message.content or ""
                if not text:
                    text = getattr(resp.choices[0].message, "reasoning", "") or ""
                return parse_score(text), text
            except Exception as e:
                msg = str(e)
                if any(k in msg for k in ("Key limit","Insufficient credits","402","Payment Required")):
                    raise QuotaError(msg)
                if "403" in msg and "exceeded" in msg.lower():
                    raise QuotaError(msg)
                if attempt == 2:
                    return None, None
                await asyncio.sleep(2 ** attempt)


async def score_judge(client, model_id, universe, conn_lock, conn, args, state):
    slug = slug_for(model_id)
    state[slug] = {"scored": 0, "todo": 0, "failed": 0, "rate_per_sec": 0.0,
                   "model": model_id, "started": time.time(), "status": "loading"}

    async with conn_lock:
        already = {(r[0], r[1]) for r in conn.execute(
            "SELECT rewrite_id, criterion FROM attack_judge_scores WHERE judge_slug=?",
            (slug,)).fetchall()}
    todo = [u for u in universe if (u[0], u[3]) not in already]
    state[slug]["todo"] = len(todo)
    state[slug]["status"] = "running" if todo else "done"
    print(f"[{time.strftime('%H:%M:%S')}] {model_id}: {len(todo)}/{len(universe)} cross-crit cells todo",
          flush=True)
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)
    n_done = n_failed = 0
    t0 = time.time()
    CHUNK = max(40, args.concurrency * 4)

    for chunk_start in range(0, len(todo), CHUNK):
        if stop_requested():
            print(f"[{time.strftime('%H:%M:%S')}] {slug} STOP requested; saving + exiting", flush=True)
            return
        chunk = todo[chunk_start:chunk_start + CHUNK]
        tasks = []
        for rid, prop, text, opp_crit in chunk:
            user_msg = RUBRICS[opp_crit].format(proposition=prop, paragraph=text)
            tasks.append(score_one(client, model_id, JUDGE_SYSTEM, user_msg, sem))
        try:
            results = await asyncio.gather(*tasks)
        except QuotaError as e:
            print(f"[{time.strftime('%H:%M:%S')}] {slug} QUOTA: {e}", flush=True)
            _stop_flag["reason"] = "quota"
            return

        inserts = []
        for (rid, _, _, opp_crit), (score, raw) in zip(chunk, results):
            if score is None:
                n_failed += 1
                continue
            inserts.append((rid, slug, opp_crit, float(score), (raw[:2000] if raw else None)))
        async with conn_lock:
            conn.executemany("""INSERT OR IGNORE INTO attack_judge_scores
                                (rewrite_id, judge_slug, criterion, score, reasoning)
                                VALUES (?, ?, ?, ?, ?)""", inserts)
            conn.commit()
        n_done += len(chunk)
        elapsed = time.time() - t0
        rate = n_done / max(1e-6, elapsed)
        eta_min = (len(todo) - n_done) / max(1e-6, rate) / 60
        state[slug].update({"scored": n_done, "failed": n_failed,
                            "rate_per_sec": round(rate, 2),
                            "eta_minutes": round(eta_min, 1),
                            "elapsed_minutes": round(elapsed/60, 2)})
        if n_done % 1000 == 0 or (n_done < 1000 and n_done % 200 == 0):
            print(f"[{time.strftime('%H:%M:%S')}] {slug} {n_done}/{len(todo)} "
                  f"rate={rate:.1f}/s failed={n_failed} eta={eta_min:.1f}min", flush=True)
    state[slug]["status"] = "done"
    print(f"[{time.strftime('%H:%M:%S')}] {slug} DONE in {(time.time()-t0)/60:.1f}min "
          f"({n_done} scored, {n_failed} failed)", flush=True)


async def progress_writer(state, stop_event, interval=30):
    while not stop_event.is_set():
        try:
            PROGRESS_FILE.write_text(json.dumps({
                "wall_time": time.time(), "judges": state,
            }, indent=2))
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def main_async(args):
    install_signal_handlers()
    if STOP_FILE.exists():
        STOP_FILE.unlink()

    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY_EXPERIMENTS"],
    )

    conn = sqlite3.connect(DB, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn_lock = asyncio.Lock()

    universe = fetch_universe(conn)
    print(f"[{time.strftime('%H:%M:%S')}] cross-crit universe: {len(universe)} (rewrite, opp_crit) cells",
          flush=True)

    state = {}
    stop_event = asyncio.Event()
    prog_task = asyncio.create_task(progress_writer(state, stop_event))

    judge_tasks = [
        asyncio.create_task(score_judge(client, mid, universe, conn_lock, conn, args, state))
        for mid in args.judges
    ]
    try:
        await asyncio.gather(*judge_tasks)
    finally:
        stop_event.set()
        await prog_task
        try:
            PROGRESS_FILE.write_text(json.dumps({
                "wall_time": time.time(), "judges": state, "final": True,
            }, indent=2))
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+",
                    default=["qwen/qwen3.5-flash-02-23",
                             "google/gemini-2.5-flash-lite",
                             "xiaomi/mimo-v2-flash"])
    ap.add_argument("--concurrency", type=int, default=30)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
