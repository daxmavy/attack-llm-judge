"""Parallel-judge OpenRouter scoring panel.

Runs N OpenRouter judge models concurrently against the full attack-rewrite
universe (all attack methods + originals + AI-originals). Each judge scores
each row once per criterion (clarity, informativeness). Idempotent: rows
already scored by (judge_slug × criterion) are skipped.

Designed to coexist with the local GPU GRPO chain: OpenRouter is API-only,
no GPU contention. The script also picks up rows that appear in the DB AFTER
script start — but only on a re-run; within a single run the universe is
snapshot at startup. Re-launch any time to backfill new rows.

Graceful stop:
  touch /tmp/openrouter_panel.STOP    → finishes the current chunk per judge,
                                         commits, exits 0
SIGTERM is also handled the same way.

Quota / payment errors (402, 403 "Key limit exceeded", insufficient credits):
the script saves what's been completed and exits 1 with a clear marker line.

Progress monitoring:
  /tmp/openrouter_panel.progress.json — updated every 30s with per-judge
    state (scored, todo, rate, eta_minutes).
  stdout — milestone log lines every 100 scorings per judge for the first
    1000, then every 1000.

Usage:
  python3 scripts/score_openrouter_panel.py \\
    --judges qwen/qwen3-235b-a22b-2507 google/gemini-2.5-flash-lite xiaomi/mimo-v2-flash \\
    --concurrency 30
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import re
import signal
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/grpo_run")
from run_pilot_len_pen import JUDGE_SYSTEM, RUBRICS, parse_score

DB = "/home/max/attack-llm-judge/data/paragraphs.db"
STOP_FILE = Path("/tmp/openrouter_panel.STOP")
PROGRESS_FILE = Path("/tmp/openrouter_panel.progress.json")

# Methods that should be scored — uses criterion of the row, except for the
# criterion-agnostic ones (naive, original, original_ai) which are scored at
# both criteria from a single DB row.
CRITERION_AGNOSTIC = {"naive", "original", "original_ai"}
ATTACK_METHODS = [
    "naive", "lit_informed_tight", "rubric_aware", "icir",
    "bon_panel", "bon_panel_single", "bon_panel_nli", "bon_panel_single_nli",
    "bon_panel_highnli", "bon_panel_mean3",
    "grpo_400step", "grpo_nli_400step", "grpo_nli_single",
    "lit_informed_tight_strictlen_opus47",
    "original", "original_ai",
]

REWRITERS_3 = (
    # name kept for backwards-compat; tuple now covers tiny tier + small tier
    "Qwen/Qwen2.5-1.5B-Instruct",
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "google/gemma-3-1b-it",
    "Qwen/Qwen3-8B",  # imported from singlejudge-big HF dataset
)


def slug_for(model_id: str) -> str:
    last = model_id.split("/")[-1]
    return "judge_" + re.sub(r"[^a-z0-9]", "_", last.lower())


# ─── stop conditions ─────────────────────────────────────────────────────
_stop_flag = {"reason": None}


def install_signal_handlers():
    def _handler(signum, frame):
        _stop_flag["reason"] = f"signal_{signum}"
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def stop_requested() -> str | None:
    if _stop_flag["reason"]:
        return _stop_flag["reason"]
    if STOP_FILE.exists():
        return "stop_file"
    return None


# ─── universe ────────────────────────────────────────────────────────────
def fetch_universe(conn, methods, rewriters):
    """Return list of (rewrite_id, proposition, text, criterion_to_score) tuples.

    Each row appears once per (rewrite × criterion-to-score). For criterion-agnostic
    methods, that means two rows per rewrite_id (clarity + informativeness).
    """
    placeholders_methods = ",".join("?" * len(methods))
    if rewriters is None:
        # for original/original_ai which span 3-rewriter rows + others; allow NULL rewriter_model
        rewriter_clause = ""
        params = list(methods)
    else:
        placeholders_rewriters = ",".join("?" * len(rewriters))
        rewriter_clause = f" AND r.rewriter_model IN ({placeholders_rewriters})"
        params = list(methods) + list(rewriters)
    rows = conn.execute(f"""
        SELECT r.rewrite_id, p.proposition, r.text, r.method, r.criterion
        FROM attack_rewrites r
        JOIN paragraphs p ON r.source_doc_id = p.document_id
        WHERE r.method IN ({placeholders_methods}){rewriter_clause}
    """, params).fetchall()
    universe = []
    for rid, prop, text, method, cri in rows:
        if method in CRITERION_AGNOSTIC:
            # score at both criteria
            for ck in ("clarity", "informativeness"):
                universe.append((rid, prop, text, ck))
        else:
            universe.append((rid, prop, text, cri))
    return universe


# ─── single API call ─────────────────────────────────────────────────────
class QuotaError(Exception):
    pass


async def score_one(client, model_id, system, user, semaphore, max_retries=3):
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
                    max_tokens=1500,
                    extra_body={"reasoning": {"enabled": False}},
                    extra_headers={"HTTP-Referer": "https://github.com/daxmavy/attack-llm-judge",
                                    "X-Title": "attack-llm-judge"},
                )
                text = resp.choices[0].message.content or ""
                if not text:
                    # some reasoning models still emit only .reasoning despite enabled=False;
                    # in that case fall back to whatever's there
                    text = getattr(resp.choices[0].message, "reasoning", "") or ""
                score = parse_score(text)
                return score, text
            except Exception as e:
                msg = str(e)
                if "Key limit" in msg or "Insufficient credits" in msg or "402" in msg or "Payment Required" in msg:
                    raise QuotaError(msg)
                if "403" in msg and "exceeded" in msg.lower():
                    raise QuotaError(msg)
                if attempt == max_retries - 1:
                    return None, None
                await asyncio.sleep(2 ** attempt)
        return None, None


# ─── per-judge scoring task ──────────────────────────────────────────────
async def score_judge(client, model_id, universe, conn_lock, conn, args, state):
    slug = slug_for(model_id)
    state[slug] = {"scored": 0, "todo": 0, "failed": 0, "rate_per_sec": 0.0,
                   "eta_minutes": None, "model": model_id, "started": time.time(),
                   "status": "loading"}

    # Pre-skip: figure out which (rewrite_id, criterion) combos are already scored
    async with conn_lock:
        already = {(r[0], r[1]) for r in conn.execute(
            "SELECT rewrite_id, criterion FROM attack_judge_scores WHERE judge_slug=?",
            (slug,)).fetchall()}
    todo = [u for u in universe if (u[0], u[3]) not in already]
    state[slug]["todo"] = len(todo)
    state[slug]["status"] = "running" if todo else "done_already"
    print(f"[{time.strftime('%H:%M:%S')}] {model_id} ({slug}): {len(already)} already scored, {len(todo)} remaining",
          flush=True)
    if not todo:
        state[slug]["status"] = "done"
        return

    sem = asyncio.Semaphore(args.concurrency)
    n_done = 0
    n_failed = 0
    t0 = time.time()
    eta_calibrated = False

    CHUNK = max(40, args.concurrency * 4)
    for chunk_start in range(0, len(todo), CHUNK):
        # graceful stop check at chunk boundary
        reason = stop_requested()
        if reason:
            print(f"[{time.strftime('%H:%M:%S')}] {slug} STOP requested ({reason}); committing + exiting",
                  flush=True)
            state[slug]["status"] = f"stopped_{reason}"
            return

        chunk = todo[chunk_start:chunk_start + CHUNK]
        tasks = []
        rubric = None  # criterion varies per row, build per-row prompt
        for rid, prop, text, ck in chunk:
            user_msg = RUBRICS[ck].format(proposition=prop, paragraph=text)
            tasks.append(score_one(client, model_id, JUDGE_SYSTEM, user_msg, sem))
        try:
            results = await asyncio.gather(*tasks)
        except QuotaError as e:
            print(f"[{time.strftime('%H:%M:%S')}] {slug} QUOTA EXCEEDED — {e}", flush=True)
            print(f"[{time.strftime('%H:%M:%S')}] {slug} marker: OPENROUTER_QUOTA_EXIT", flush=True)
            _stop_flag["reason"] = "quota"
            state[slug]["status"] = "quota_exit"
            return

        inserts = []
        for (rid, _, _, ck), (score, raw) in zip(chunk, results):
            if score is None:
                n_failed += 1
                continue
            inserts.append((rid, slug, ck, float(score), (raw[:2000] if raw else None)))
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

        # milestone logging
        if not eta_calibrated and n_done >= 100:
            eta_calibrated = True
            print(f"[{time.strftime('%H:%M:%S')}] {slug} ETA-calibration: "
                  f"{n_done}/{len(todo)} after {elapsed:.0f}s, "
                  f"rate={rate:.1f}/s, est_total={eta_min:.1f}min",
                  flush=True)
        if n_done % 1000 == 0 or (n_done < 1000 and n_done % 100 == 0):
            print(f"[{time.strftime('%H:%M:%S')}] {slug} {n_done}/{len(todo)} "
                  f"rate={rate:.1f}/s failed={n_failed} eta={eta_min:.1f}min",
                  flush=True)

    state[slug]["status"] = "done"
    print(f"[{time.strftime('%H:%M:%S')}] {slug} DONE in {(time.time()-t0)/60:.1f}min "
          f"({n_done} scored, {n_failed} failed)", flush=True)


# ─── progress writer ─────────────────────────────────────────────────────
async def progress_writer(state, stop_event, interval=30):
    while not stop_event.is_set():
        try:
            PROGRESS_FILE.write_text(json.dumps({
                "wall_time": time.time(),
                "judges": {k: v for k, v in state.items()},
            }, indent=2))
        except Exception as e:
            print(f"  [progress_writer] {e!r}", flush=True)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


# ─── main ────────────────────────────────────────────────────────────────
async def main_async(args):
    install_signal_handlers()
    if STOP_FILE.exists():
        STOP_FILE.unlink()
    print(f"[stop-signal] graceful-stop file: {STOP_FILE}", flush=True)
    print(f"[stop-signal]   `touch {STOP_FILE}` to save + exit at the next chunk boundary.", flush=True)

    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY_EXPERIMENTS"],
    )

    conn = sqlite3.connect(DB, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn_lock = asyncio.Lock()

    # Build universe: 3-rewriter attack methods + originals (no rewriter filter)
    # + OpenRouter rewriter methods (anthropic/claude-opus-4-7 etc, no rewriter filter)
    print(f"[{time.strftime('%H:%M:%S')}] building universe...", flush=True)
    or_rewriter_methods = {"lit_informed_tight_strictlen_opus47"}
    methods_3rew = [m for m in ATTACK_METHODS
                    if m not in ("original", "original_ai") and m not in or_rewriter_methods]
    universe_3rew = fetch_universe(conn, methods_3rew, REWRITERS_3)
    methods_orig = ["original", "original_ai"]
    universe_orig = fetch_universe(conn, methods_orig, None)
    methods_or_rew = [m for m in ATTACK_METHODS if m in or_rewriter_methods]
    universe_or_rew = fetch_universe(conn, methods_or_rew, None) if methods_or_rew else []
    universe = universe_3rew + universe_orig + universe_or_rew
    print(f"  attack-method calls (3 rewriters): {len(universe_3rew)}", flush=True)
    print(f"  original / original_ai calls:      {len(universe_orig)}", flush=True)
    print(f"  OpenRouter-rewriter calls:         {len(universe_or_rew)}", flush=True)
    print(f"  TOTAL universe per judge:          {len(universe)}", flush=True)

    state = {}
    stop_event = asyncio.Event()
    progress_task = asyncio.create_task(progress_writer(state, stop_event))

    judge_tasks = [
        asyncio.create_task(score_judge(client, mid, universe, conn_lock, conn, args, state))
        for mid in args.judges
    ]
    try:
        await asyncio.gather(*judge_tasks)
    finally:
        stop_event.set()
        await progress_task
        # final progress dump
        try:
            PROGRESS_FILE.write_text(json.dumps({
                "wall_time": time.time(), "judges": state, "final": True,
            }, indent=2))
        except Exception:
            pass

    if _stop_flag["reason"] == "quota":
        print(f"[{time.strftime('%H:%M:%S')}] EXIT 1 — quota exceeded; saved partial progress", flush=True)
        sys.exit(1)
    if _stop_flag["reason"]:
        print(f"[{time.strftime('%H:%M:%S')}] EXIT 0 — stopped via {_stop_flag['reason']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", required=True,
                    help="OpenRouter model IDs")
    ap.add_argument("--concurrency", type=int, default=30,
                    help="parallel in-flight requests per judge")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
