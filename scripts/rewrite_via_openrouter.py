"""Rewrite eval-set paragraphs via OpenRouter (Anthropic Opus 4.7 etc).

Method: lit_informed_tight (per-criterion). Inserts into attack_rewrites with
method='lit_informed_tight_<model_short>' (so existing 'lit_informed_tight'
qwen rows are not overwritten and are still queryable).

Probe mode (--probe N):
  - Generate N rewrites for sanity check, write JSONL to PROBE_OUT, no DB insert.
  - Default N=10; intended for inspecting length tolerance before the full run.

Full mode:
  - 714 eval × 2 criteria = 1428 calls.
  - Idempotent: skips (source_doc_id, criterion) pairs already in DB.

Reasoning is disabled via extra_body={"reasoning":{"enabled": False}} for parity
with the judge panel.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
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

sys.path.insert(0, "/home/max/attack-llm-judge")
from rewriters.rewrite_prompts import (
    SYSTEM_PROMPT,
    LIT_INFORMED_TIGHT_USER_TEMPLATE,
    LIT_INFORMED_TIGHT_INFORMATIVENESS_TEMPLATE,
    LIT_INFORMED_TIGHT_STRICTLEN_USER_TEMPLATE,
    LIT_INFORMED_TIGHT_STRICTLEN_INFORMATIVENESS_TEMPLATE,
    length_bounds,
)

DB = "/home/max/attack-llm-judge/data/paragraphs.db"
DATASET = "/workspace/grpo_run/controversial_40_3fold.json"
STOP_FILE = Path("/tmp/openrouter_rewrite.STOP")
PROGRESS_FILE = Path("/tmp/openrouter_rewrite.progress.json")
PROBE_OUT = Path("/tmp/openrouter_rewrite.probe.jsonl")

MODEL_SHORT = {
    "anthropic/claude-opus-4-7": "opus47",
    "anthropic/claude-sonnet-4-6": "sonnet46",
    "anthropic/claude-haiku-4-5": "haiku45",
}

_stop_flag = {"reason": None}


def install_signal_handlers():
    def _handler(sig, _frame):
        _stop_flag["reason"] = f"signal_{sig}"
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def stop_requested():
    if _stop_flag["reason"]:
        return _stop_flag["reason"]
    if STOP_FILE.exists():
        return "stop_file"
    return None


class QuotaError(Exception):
    pass


def load_eval_rows():
    d = json.loads(Path(DATASET).read_text())
    return [r for r in d["rows"] if r["split"] == "eval"]


def build_user_prompt(row, criterion, strict_length=False):
    orig, lo, hi = length_bounds(row["text"], tolerance=0.10)
    if strict_length:
        tmpl = (LIT_INFORMED_TIGHT_STRICTLEN_INFORMATIVENESS_TEMPLATE
                if criterion == "informativeness"
                else LIT_INFORMED_TIGHT_STRICTLEN_USER_TEMPLATE)
    elif criterion == "informativeness":
        tmpl = LIT_INFORMED_TIGHT_INFORMATIVENESS_TEMPLATE
    else:
        tmpl = LIT_INFORMED_TIGHT_USER_TEMPLATE
    return tmpl.format(
        proposition=row["proposition"],
        paragraph=row["text"],
        orig_words=orig, min_words=lo, max_words=hi,
    )


def make_rewrite_id(doc_id, method, model_short, criterion):
    h = hashlib.sha1(f"{doc_id}|{method}|{model_short}|{criterion}".encode()).hexdigest()[:12]
    return f"orRW_{model_short}_{criterion[:3]}_{doc_id[:20]}_{h}"


async def call_one(client, model_id, system, user, semaphore, max_retries=4):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.7,
                    max_tokens=800,
                    extra_body={"reasoning": {"enabled": False}},
                    extra_headers={"HTTP-Referer": "https://github.com/daxmavy/attack-llm-judge",
                                   "X-Title": "attack-llm-judge"},
                )
                text = resp.choices[0].message.content or ""
                if not text:
                    text = getattr(resp.choices[0].message, "reasoning", "") or ""
                return text.strip(), {
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None) if resp.usage else None,
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None) if resp.usage else None,
                }
            except Exception as e:
                msg = str(e)
                if "Key limit" in msg or "Insufficient credits" in msg or "402" in msg or "Payment Required" in msg:
                    raise QuotaError(msg)
                if "403" in msg and "exceeded" in msg.lower():
                    raise QuotaError(msg)
                if attempt == max_retries - 1:
                    return None, {"error": msg[:300]}
                await asyncio.sleep(2 ** attempt)
        return None, {"error": "max_retries"}


async def progress_writer(state, stop_event, interval=15):
    while not stop_event.is_set():
        try:
            PROGRESS_FILE.write_text(json.dumps({
                "wall_time": time.time(),
                "state": state,
            }, indent=2))
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def run_probe(client, model_id, n, criterion, strict_length=False):
    rows = load_eval_rows()[:n]
    sem = asyncio.Semaphore(min(8, n))
    tasks = []
    for r in rows:
        user = build_user_prompt(r, criterion, strict_length=strict_length)
        tasks.append(call_one(client, model_id, SYSTEM_PROMPT, user, sem))
    t0 = time.time()
    results = await asyncio.gather(*tasks)
    dt = time.time() - t0

    out_recs = []
    in_range = 0
    word_diffs = []
    for r, (text, meta) in zip(rows, results):
        if text is None:
            print(f"  [FAIL] {r['document_id']} -> {meta}", flush=True)
            continue
        orig, lo, hi = length_bounds(r["text"], tolerance=0.10)
        rwc = len(text.split())
        ok = lo <= rwc <= hi
        in_range += 1 if ok else 0
        word_diffs.append(rwc - orig)
        out_recs.append({
            "document_id": r["document_id"],
            "criterion": criterion,
            "orig_words": orig, "rewrite_words": rwc,
            "lo": lo, "hi": hi, "in_range": ok,
            "rewrite": text, "meta": meta,
        })
    PROBE_OUT.write_text("\n".join(json.dumps(o) for o in out_recs))
    tag = "strict" if strict_length else "default"
    print(f"\n=== PROBE SUMMARY ({model_id}, {criterion}, {tag}) ===", flush=True)
    print(f"  generated {len(out_recs)}/{len(rows)} in {dt:.1f}s ({len(out_recs)/dt:.2f}/s)", flush=True)
    print(f"  in-range [{(1-0.10)*100:.0f}%-{(1+0.10)*100:.0f}% of orig]: {in_range}/{len(out_recs)}", flush=True)
    if word_diffs:
        avg = sum(word_diffs) / len(word_diffs)
        print(f"  avg(rewrite_words - orig_words): {avg:+.1f}  range: [{min(word_diffs)},{max(word_diffs)}]", flush=True)
    print(f"  full output: {PROBE_OUT}", flush=True)
    return in_range / max(1, len(out_recs))


async def run_full(client, model_id, criteria, args):
    model_short = MODEL_SHORT.get(model_id, model_id.split("/")[-1].replace("-", "").replace(".", "")[:10])
    if args.strict_length:
        method_tag = f"lit_informed_tight_strictlen_{model_short}"
    else:
        method_tag = f"lit_informed_tight_{model_short}"
    rows = load_eval_rows()
    print(f"[{time.strftime('%H:%M:%S')}] eval rows: {len(rows)}; criteria: {criteria}; model: {model_id}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] method tag: {method_tag}", flush=True)

    conn = sqlite3.connect(DB, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn_lock = asyncio.Lock()

    # Build universe of (row, criterion) pairs not yet in DB
    universe = []
    for r in rows:
        for cri in criteria:
            rid = make_rewrite_id(r["document_id"], method_tag, model_short, cri)
            existing = conn.execute(
                "SELECT 1 FROM attack_rewrites WHERE rewrite_id=?", (rid,)
            ).fetchone()
            if existing:
                continue
            universe.append((r, cri, rid))
    print(f"[{time.strftime('%H:%M:%S')}] todo: {len(universe)} (skipped {len(rows)*len(criteria) - len(universe)} already-done)", flush=True)
    if not universe:
        return

    state = {"todo": len(universe), "done": 0, "failed": 0, "rate_per_sec": 0.0,
             "started": time.time(), "model": model_id, "method": method_tag,
             "in_range": 0, "out_of_range": 0}
    stop_event = asyncio.Event()
    prog_task = asyncio.create_task(progress_writer(state, stop_event))

    sem = asyncio.Semaphore(args.concurrency)
    n_done = 0
    n_failed = 0
    n_in_range = 0
    n_out_of_range = 0
    t0 = time.time()
    CHUNK = max(20, args.concurrency * 4)

    try:
        for chunk_start in range(0, len(universe), CHUNK):
            reason = stop_requested()
            if reason:
                print(f"[{time.strftime('%H:%M:%S')}] STOP requested ({reason}); committing + exiting", flush=True)
                state["status"] = f"stopped_{reason}"
                break
            chunk = universe[chunk_start:chunk_start + CHUNK]
            tasks = []
            for r, cri, rid in chunk:
                user = build_user_prompt(r, cri, strict_length=args.strict_length)
                tasks.append(call_one(client, model_id, SYSTEM_PROMPT, user, sem))
            try:
                results = await asyncio.gather(*tasks)
            except QuotaError as e:
                print(f"[{time.strftime('%H:%M:%S')}] QUOTA EXCEEDED — {e}", flush=True)
                state["status"] = "quota_exit"
                _stop_flag["reason"] = "quota"
                break

            inserts = []
            for (r, cri, rid), (text, meta) in zip(chunk, results):
                if text is None:
                    n_failed += 1
                    continue
                orig, lo, hi = length_bounds(r["text"], tolerance=0.10)
                rwc = len(text.split())
                if lo <= rwc <= hi:
                    n_in_range += 1
                else:
                    n_out_of_range += 1
                config = {"method": method_tag, "criterion": cri, "rewriter": model_id,
                          "temperature": 0.7, "max_tokens": 800,
                          "reasoning_disabled": True, "tolerance": 0.10}
                inserts.append((rid, r["document_id"], method_tag, cri,
                                json.dumps(config, separators=(",", ":")),
                                model_id,
                                text, rwc,
                                json.dumps({**(meta or {}), "orig_words": orig,
                                            "lo": lo, "hi": hi, "in_range": lo <= rwc <= hi},
                                           separators=(",", ":"))))
            async with conn_lock:
                conn.executemany("""
                    INSERT OR IGNORE INTO attack_rewrites
                    (rewrite_id, source_doc_id, method, fold, criterion,
                     config_json, rewriter_model, judge_panel_json,
                     text, word_count, run_metadata_json)
                    VALUES (?, ?, ?, NULL, ?, ?, ?, NULL, ?, ?, ?)
                """, inserts)
                conn.commit()
            n_done += len(chunk)
            elapsed = time.time() - t0
            rate = n_done / max(1e-6, elapsed)
            state.update({"done": n_done, "failed": n_failed,
                          "rate_per_sec": round(rate, 2),
                          "elapsed_minutes": round(elapsed/60, 2),
                          "eta_minutes": round((len(universe) - n_done)/max(1e-6, rate)/60, 1),
                          "in_range": n_in_range, "out_of_range": n_out_of_range})

            if n_done % 100 == 0 or n_done == len(universe):
                pct_in = 100.0 * n_in_range / max(1, n_in_range + n_out_of_range)
                print(f"[{time.strftime('%H:%M:%S')}] {n_done}/{len(universe)} "
                      f"rate={rate:.2f}/s failed={n_failed} in_range={pct_in:.1f}% "
                      f"eta={state['eta_minutes']:.1f}min", flush=True)
    finally:
        stop_event.set()
        await prog_task
        try:
            PROGRESS_FILE.write_text(json.dumps({
                "wall_time": time.time(), "state": state, "final": True,
            }, indent=2))
        except Exception:
            pass

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE: {n_done} rewrites, {n_failed} failed, "
          f"in_range={n_in_range}/{n_in_range+n_out_of_range}", flush=True)


async def main_async(args):
    install_signal_handlers()
    if STOP_FILE.exists():
        STOP_FILE.unlink()
    print(f"[stop-signal] graceful-stop: touch {STOP_FILE}", flush=True)

    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY_EXPERIMENTS"],
    )

    if args.probe is not None:
        await run_probe(client, args.model, args.probe, args.criterion,
                        strict_length=args.strict_length)
        return

    criteria = ["clarity", "informativeness"] if args.criterion == "both" else [args.criterion]
    await run_full(client, args.model, criteria, args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-opus-4-7",
                    help="OpenRouter model slug")
    ap.add_argument("--criterion", choices=["clarity", "informativeness", "both"], default="both")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--probe", type=int, default=None,
                    help="probe mode: generate N rewrites for one criterion, write JSONL, no DB insert")
    ap.add_argument("--strict-length", action="store_true",
                    help="use the strict-length variant of the prompt (hardened MUST-be-in-range constraint)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
