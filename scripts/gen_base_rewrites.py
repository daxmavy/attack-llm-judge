"""Generate base-model (no-GRPO) rewrites via a local vLLM OpenAI server.

Samples N paragraphs from the training slice the rewriter saw
(`origin_kind='original_writer' AND writer_is_top_decile=1`), applies the
same /no_think prompt the GRPO pipeline uses, calls the vLLM chat-completions
endpoint, and saves (original, rewrite, proposition, ...) pairs to JSONL.

Used as the base-model baseline for the ModernBERT NLI distribution probe:
compare against the GRPO-rollout distribution to see how much the fidelity
term (embed-sim or NLI) needs to push the rewriter toward preserving the
original's stance.
"""
import argparse
import json
import random
import re
import sqlite3
import time
from pathlib import Path

import requests

DB = "/home/shil6647/attack-llm-judge/data/paragraphs.db"


_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_think_block(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def make_rewrite_prompt(proposition, text, word_count):
    system = ("You are rewriting short argumentative paragraphs about political/policy propositions "
              "to improve their clarity while preserving the writer's original stance.")
    user = (
        f"Rewrite the following paragraph to be clearer and easier to read on first pass. "
        f"Keep the original stance toward the proposition (pro remains pro, con remains con). "
        f"Aim for approximately {word_count} words. Write one paragraph, no headings or bullets. "
        f"Do not invent names, statistics, or studies — rephrase what is already there.\n\n"
        f"Proposition: {proposition}\n\n"
        f"Original paragraph:\n\"\"\"{text}\"\"\"\n\n"
        f"Return only the rewritten paragraph. /no_think"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8200/v1/chat/completions")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--out", default="/data/shil6647/attack-llm-judge/tmp/base_rewrites/base_rewrites.jsonl")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT document_id, proposition_id, proposition, text, word_count
        FROM paragraphs
        WHERE origin_kind='original_writer' AND writer_is_top_decile=1
        ORDER BY proposition_id, document_id
    """).fetchall()
    conn.close()
    print(f"[{time.strftime('%H:%M:%S')}] {len(rows)} training paragraphs in DB; sampling {args.n}",
          flush=True)
    rng.shuffle(rows)
    sample = rows[: args.n]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Single synchronous loop is fine — vLLM batches server-side. For better
    # throughput we fire via a ThreadPoolExecutor.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def call_one(row):
        doc_id, prop_id, prop, text, wc = row
        msgs = make_rewrite_prompt(prop, text, int(wc))
        body = {
            "model": args.model,
            "messages": msgs,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            # Qwen3-specific: /no_think in the user text plus enable_thinking=False
            # via chat_template_kwargs fully disables reasoning-mode output.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.time()
        resp = requests.post(args.endpoint, json=body, timeout=180)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        rewrite = strip_think_block(raw)
        return {
            "document_id": doc_id,
            "proposition_id": prop_id,
            "proposition": prop,
            "original": text,
            "word_count": int(wc),
            "rewrite_raw": raw,
            "rewrite": rewrite,
            "gen_seconds": round(time.time() - t0, 2),
        }

    t_start = time.time()
    results = []
    with out_path.open("w") as f, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(call_one, r): r for r in sample}
        for i, fut in enumerate(as_completed(futs)):
            try:
                rec = fut.result()
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] err {type(e).__name__}: {e}", flush=True)
                continue
            results.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 25 == 0 or (i + 1) == len(sample):
                elapsed = time.time() - t_start
                print(f"[{time.strftime('%H:%M:%S')}] done {i+1}/{len(sample)} "
                      f"({elapsed:.0f}s, {(i+1)/elapsed:.2f}/s)", flush=True)

    # quick stats
    import numpy as np
    rew_chars = np.array([len(r["rewrite"]) for r in results])
    raw_chars = np.array([len(r["rewrite_raw"]) for r in results])
    had_think = int(np.sum(rew_chars < raw_chars - 5))
    empty = int(np.sum(rew_chars < 20))
    print(f"\n=== wrote {out_path} — N={len(results)} ===")
    print(f"  raw   mean chars = {raw_chars.mean():.0f}")
    print(f"  clean mean chars = {rew_chars.mean():.0f}")
    print(f"  had <think>      = {had_think}/{len(results)}")
    print(f"  <20 chars (drop) = {empty}/{len(results)}")


if __name__ == "__main__":
    main()
