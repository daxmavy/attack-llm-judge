"""Mission runner: feedback-free attacks + BoN generation/scoring + DB persistence.

Subcommands:
    feedback_free       generate naive / lit_informed_tight / rubric_aware on eval set (fold-independent)
    bon_generate        generate K=16 candidates per eval paragraph (fold-independent)
    bon_score FOLD      score stored bon_candidate rewrites with FOLD's 2 in-panel judges; pick argmax → bon_panel/foldN

All output goes to paragraphs.db tables `attack_rewrites` and `attack_judge_scores`.
Uses the out-of-process judge server for all scoring (judge.http_client.JudgeHTTP)
so BoN panel scoring stays consistent with the RL training threat model.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# .env first (HF_TOKEN etc)
ENV_FILE = "/home/shil6647/attack-llm-judge/.env"
if os.path.exists(ENV_FILE):
    for line in open(ENV_FILE):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v
os.environ.setdefault("HF_HOME", "/data/shil6647/attack-llm-judge/hf_cache")
os.environ.setdefault("VLLM_CACHE_ROOT", "/data/shil6647/attack-llm-judge/vllm_cache")

sys.path.insert(0, "/data/shil6647/attack-llm-judge/grpo_run")
sys.path.insert(0, "/home/shil6647/attack-llm-judge")

import sqlite3

from config.models import REWRITER, JUDGE_REGISTRY, FOLDS, rewriter_short_name, require_config
from judge.http_client import JudgeHTTP, spawn_judge_server
from stop_signal import announce as stop_announce, check_stop_signal, clear_stop_file

DB = "/home/shil6647/attack-llm-judge/data/paragraphs.db"
DATASET = "/data/shil6647/attack-llm-judge/grpo_run/controversial_40_3fold.json"
# Default STOP-marker location for graceful cancellation of attack-batch runs.
# Each subcommand can override via --stop-file.
DEFAULT_STOP_DIR = "/data/shil6647/attack-llm-judge/grpo_run/stop_markers"


def _default_stop_file(subcmd: str, criterion: str) -> str:
    return f"{DEFAULT_STOP_DIR}/mission_attacks_{subcmd}_{criterion}.STOP"


def _init_stop_marker(stop_file: str) -> str:
    """Clear a stale marker, ensure the dir exists, and announce the path."""
    Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
    clear_stop_file(stop_file)
    stop_announce(stop_file)
    return stop_file


def _rewriter_short_name(model_id: str) -> str:
    """Thin shim around config.models.rewriter_short_name for back-compat."""
    return rewriter_short_name(model_id)


def _rewriter_vllm_mem_util(model_id: str) -> float:
    """Auto-pick gpu_memory_utilization for rewriter vLLM based on model size.

    Qwen2.5-1.5B (~3 GB bf16) → 0.15 budget easily fits weights+KV.
    Qwen3-14B (~28 GB bf16) → 0.70 budget = 56 GB / 80 GB A100 covers
    weights + overhead + KV headroom for the K=16 BoN rollout
    (0.55 → 44 GB left only ~16 GB for KV after weights + CUDA buffers,
    which caused vLLM "No available memory for cache blocks" at
    max_model_len=3072 in the first attempt).
    """
    s = model_id.lower()
    if "14b" in s or "13b" in s:
        return 0.70
    if "7b" in s or "8b" in s:
        return 0.35
    return 0.20


def _supports_system_role(tok):
    try:
        tok.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            tokenize=False, add_generation_prompt=True)
        return True
    except Exception:
        return False


def load_eval_paragraphs():
    d = json.loads(Path(DATASET).read_text())
    return [r for r in d["rows"] if r["split"] == "eval"]


def make_rewrite_id(doc_id, method, config_hash, fold=None, idx=None):
    parts = [doc_id, method, str(config_hash), str(fold), str(idx)]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def insert_rewrite(conn, **kw):
    cols = ["rewrite_id", "source_doc_id", "method", "fold", "config_json",
            "rewriter_model", "judge_panel_json", "text", "word_count", "run_metadata_json"]
    conn.execute(
        f"INSERT OR REPLACE INTO attack_rewrites ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        [kw.get(c) for c in cols]
    )


def insert_judge_score(conn, **kw):
    cols = ["rewrite_id", "judge_slug", "criterion", "score", "reasoning"]
    conn.execute(
        f"INSERT OR REPLACE INTO attack_judge_scores ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        [kw.get(c) for c in cols]
    )


# ---------- resume / skip-existing helpers ----------
def _done_doc_ids(conn, method, criterion, rewriter):
    """Set of source_doc_ids that already have a (method, criterion, rewriter) row."""
    return set(r[0] for r in conn.execute(
        "SELECT source_doc_id FROM attack_rewrites "
        "WHERE method=? AND criterion=? AND rewriter_model=?",
        (method, criterion, rewriter)
    ).fetchall())


def _bon_candidate_counts(conn, criterion, rewriter):
    """doc_id -> count of bon_candidate rows for (criterion, rewriter)."""
    return dict(conn.execute(
        "SELECT source_doc_id, COUNT(*) FROM attack_rewrites "
        "WHERE method='bon_candidate' AND criterion=? AND rewriter_model=? "
        "GROUP BY source_doc_id",
        (criterion, rewriter)
    ).fetchall())


# ---------- feedback-free ----------
def cmd_feedback_free(args):
    from vllm import LLM, SamplingParams
    from rewriters.rewrite_prompts import build_rewrite_prompt, SYSTEM_PROMPT
    from transformers import AutoTokenizer

    rewriter = args.rewriter or REWRITER
    short = _rewriter_short_name(rewriter)
    eval_rows = load_eval_paragraphs()
    print(f"[{time.strftime('%H:%M:%S')}] loaded {len(eval_rows)} eval paragraphs", flush=True)

    stop_file = _init_stop_marker(args.stop_file or _default_stop_file("feedback_free", args.criterion))
    methods = args.methods or ["naive", "lit_informed_tight", "rubric_aware"]
    conn = sqlite3.connect(DB)

    # Skip methods that already have a complete set of rewrites (pre-load bail-out
    # so we don't cold-start the 30-60s vLLM engine for no work).
    if args.resume_skip_existing:
        pending_methods = []
        for m in methods:
            eff_cri = "clarity" if m == "naive" else args.criterion
            done = _done_doc_ids(conn, m, eff_cri, rewriter)
            if len(done) >= len(eval_rows):
                print(f"[{time.strftime('%H:%M:%S')}] skip-existing: '{m}' already has "
                      f"{len(done)}/{len(eval_rows)} rows for criterion={eff_cri}", flush=True)
            else:
                pending_methods.append(m)
        methods = pending_methods
        if not methods:
            print(f"[{time.strftime('%H:%M:%S')}] nothing to do — all methods complete", flush=True)
            conn.close()
            return

    if check_stop_signal(stop_file):
        print(f"[{time.strftime('%H:%M:%S')}] STOP detected before rewriter load — exiting", flush=True)
        conn.close()
        return

    tok = AutoTokenizer.from_pretrained(rewriter, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
                                          token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[{time.strftime('%H:%M:%S')}] loading vLLM rewriter {rewriter}", flush=True)
    llm = LLM(model=rewriter, dtype="bfloat16",
              gpu_memory_utilization=_rewriter_vllm_mem_util(rewriter),
              max_model_len=3072, enforce_eager=True, download_dir="/data/shil6647/attack-llm-judge/hf_cache")

    sys_ok = _supports_system_role(tok)
    sp = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=400, n=1)
    config = {"temperature": 0.7, "max_tokens": 400, "criterion": args.criterion, "rewriter": rewriter}
    config_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]

    for method in methods:
        # naive is criterion-agnostic; skip if already have a clarity version (reuse same rewrite)
        if method == "naive" and args.criterion == "informativeness":
            existing = conn.execute(
                "SELECT COUNT(*) FROM attack_rewrites WHERE method='naive' AND criterion='clarity' AND rewriter_model=?",
                (rewriter,)
            ).fetchone()[0]
            if existing >= len(eval_rows):
                print(f"[{time.strftime('%H:%M:%S')}] skipping naive for informativeness: reuse clarity rewrites ({rewriter})", flush=True)
                continue

        # Per-method skip-existing: drop eval rows that already have a row for
        # this (method, criterion, rewriter). If the method has any pending
        # rows we still fire the vLLM call but only for the missing ones.
        rows_for_method = eval_rows
        if args.resume_skip_existing:
            eff_cri = "clarity" if method == "naive" else args.criterion
            done = _done_doc_ids(conn, method, eff_cri, rewriter)
            rows_for_method = [r for r in eval_rows if r["document_id"] not in done]
            if not rows_for_method:
                print(f"[{time.strftime('%H:%M:%S')}] skip-existing: '{method}' fully done "
                      f"(criterion={eff_cri}), moving on", flush=True)
                continue
            if len(rows_for_method) < len(eval_rows):
                print(f"[{time.strftime('%H:%M:%S')}] skip-existing: '{method}' resuming "
                      f"{len(rows_for_method)}/{len(eval_rows)} missing rows", flush=True)

        print(f"[{time.strftime('%H:%M:%S')}] generating {method} (criterion={args.criterion}, rewriter={short})...", flush=True)
        prompts_text = []
        for r in rows_for_method:
            user = build_rewrite_prompt(method, r["proposition"], r["text"], criterion=args.criterion)
            if sys_ok:
                chat = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user}]
            else:
                chat = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user}]
            prompts_text.append(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
        t0 = time.time()
        outs = llm.generate(prompts_text, sp, use_tqdm=True)
        t_gen = time.time() - t0
        print(f"  generated {len(outs)} in {t_gen:.1f}s ({len(outs)/t_gen:.1f}/s)", flush=True)

        for r, o in zip(rows_for_method, outs):
            text = o.outputs[0].text.strip()
            rid = make_rewrite_id(r["document_id"], method, f"{short}_{config_hash}")
            conn.execute("""
                INSERT OR REPLACE INTO attack_rewrites
                (rewrite_id, source_doc_id, method, fold, criterion, config_json,
                 rewriter_model, judge_panel_json, text, word_count, run_metadata_json)
                VALUES (?, ?, ?, NULL, ?, ?, ?, NULL, ?, ?, ?)
            """, (rid, r["document_id"], method, args.criterion,
                   json.dumps({**config, "method": method, "config_hash": config_hash}),
                   rewriter, text, len(text.split()),
                   json.dumps({"completion_tokens": len(o.outputs[0].token_ids)})))
        conn.commit()
        print(f"  saved {len(outs)} rows to attack_rewrites", flush=True)

        # Graceful-stop check between methods. The current method's writes are
        # already committed; exiting here leaves a clean state for resume.
        if check_stop_signal(stop_file):
            print(f"[{time.strftime('%H:%M:%S')}] STOP detected after '{method}' — "
                  f"exiting. Re-run with --resume-skip-existing to continue.", flush=True)
            break

    conn.close()
    del llm
    gc.collect()
    print(f"[{time.strftime('%H:%M:%S')}] feedback_free done", flush=True)


# ---------- BoN generate (once, fold-independent) ----------
def cmd_bon_generate(args):
    from vllm import LLM, SamplingParams
    from rewriters.rewrite_prompts import build_rewrite_prompt, SYSTEM_PROMPT
    from transformers import AutoTokenizer

    rewriter = args.rewriter or REWRITER
    short = _rewriter_short_name(rewriter)
    eval_rows = load_eval_paragraphs()
    K = args.k

    stop_file = _init_stop_marker(args.stop_file or _default_stop_file("bon_generate", args.criterion))

    # Skip-existing: skip doc_ids that already have K candidates for
    # (criterion, rewriter). If nothing is pending, bail before vLLM load.
    pending_rows = eval_rows
    if args.resume_skip_existing:
        conn_check = sqlite3.connect(DB)
        counts = _bon_candidate_counts(conn_check, args.criterion, rewriter)
        conn_check.close()
        pending_rows = [r for r in eval_rows if counts.get(r["document_id"], 0) < K]
        if not pending_rows:
            print(f"[{time.strftime('%H:%M:%S')}] skip-existing: all {len(eval_rows)} "
                  f"eval rows already have ≥{K} candidates", flush=True)
            return
        if len(pending_rows) < len(eval_rows):
            print(f"[{time.strftime('%H:%M:%S')}] skip-existing: resuming on "
                  f"{len(pending_rows)}/{len(eval_rows)} rows (rest already have ≥{K})", flush=True)

    if check_stop_signal(stop_file):
        print(f"[{time.strftime('%H:%M:%S')}] STOP detected before rewriter load — exiting", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(rewriter, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
                                          token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[{time.strftime('%H:%M:%S')}] loading vLLM rewriter for BoN K={K} ({rewriter})", flush=True)
    llm = LLM(model=rewriter, dtype="bfloat16",
              gpu_memory_utilization=_rewriter_vllm_mem_util(rewriter),
              max_model_len=3072, enforce_eager=True, download_dir="/data/shil6647/attack-llm-judge/hf_cache")

    sys_ok = _supports_system_role(tok)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=400, n=K)
    config = {"K": K, "temperature": 1.0, "max_tokens": 400, "method": "bon_candidate",
              "criterion": args.criterion, "rewriter": rewriter}
    config_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]

    # Chunked generation: a single all-rows vLLM call finishes in ~5-10 min and
    # cannot be interrupted cleanly. Splitting into chunks gives STOP-between-
    # chunk boundaries at the cost of a tiny bit of batching efficiency; each
    # chunk's writes are committed before the next chunk starts so a STOP
    # during chunk N preserves chunks 0..N-1.
    chunk_size = max(1, args.bon_chunk_size)
    conn = sqlite3.connect(DB)
    n_total = 0
    for ci in range(0, len(pending_rows), chunk_size):
        chunk = pending_rows[ci:ci + chunk_size]
        print(f"[{time.strftime('%H:%M:%S')}] chunk {ci//chunk_size + 1}/"
              f"{(len(pending_rows) + chunk_size - 1)//chunk_size}: "
              f"{len(chunk)} rows × K={K} (criterion={args.criterion})", flush=True)
        prompts_text = []
        for r in chunk:
            user = build_rewrite_prompt("bon_panel", r["proposition"], r["text"], criterion=args.criterion)
            if sys_ok:
                chat = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user}]
            else:
                chat = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user}]
            prompts_text.append(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

        t0 = time.time()
        outs = llm.generate(prompts_text, sp, use_tqdm=False)
        t_gen = time.time() - t0
        chunk_total = sum(len(o.outputs) for o in outs)
        n_total += chunk_total
        print(f"  generated {chunk_total} candidates in {t_gen:.1f}s "
              f"({chunk_total/t_gen:.1f}/s)", flush=True)

        for r, o in zip(chunk, outs):
            for k_idx, cand in enumerate(o.outputs):
                text = cand.text.strip()
                rid = make_rewrite_id(r["document_id"], "bon_candidate", f"{short}_{config_hash}", idx=k_idx)
                conn.execute("""
                    INSERT OR REPLACE INTO attack_rewrites
                    (rewrite_id, source_doc_id, method, fold, criterion, config_json,
                     rewriter_model, judge_panel_json, text, word_count, run_metadata_json)
                    VALUES (?, ?, 'bon_candidate', NULL, ?, ?, ?, NULL, ?, ?, ?)
                """, (rid, r["document_id"], args.criterion,
                       json.dumps({**config, "candidate_idx": k_idx, "config_hash": config_hash}),
                       rewriter, text, len(text.split()),
                       json.dumps({"completion_tokens": len(cand.token_ids)})))
        conn.commit()

        if check_stop_signal(stop_file):
            print(f"[{time.strftime('%H:%M:%S')}] STOP detected after chunk {ci//chunk_size + 1} — "
                  f"exiting. Re-run with --resume-skip-existing to continue.", flush=True)
            break

    conn.close()
    del llm
    gc.collect()
    print(f"[{time.strftime('%H:%M:%S')}] bon_generate done; saved {n_total} candidates", flush=True)


# ---------- BoN score per fold ----------
def cmd_bon_score(args):
    """For the given fold, score stored bon_candidate rows with that fold's 2 in-panel judges,
    pick argmax per source paragraph, save as method='bon_panel' fold=FOLD."""
    fold = args.fold
    fold_spec = FOLDS[fold]
    in_panel_keys = fold_spec["in_panel"]
    print(f"[{time.strftime('%H:%M:%S')}] BoN scoring fold {fold}: in_panel={in_panel_keys}", flush=True)

    rewriter = args.rewriter or REWRITER
    short = _rewriter_short_name(rewriter)

    stop_file = _init_stop_marker(
        args.stop_file or _default_stop_file(f"bon_score_fold{fold}", args.criterion))

    # Load the candidates for this criterion AND rewriter
    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT r.rewrite_id, r.source_doc_id, r.text, p.proposition, r.config_json
        FROM attack_rewrites r
        JOIN paragraphs p ON r.source_doc_id = p.document_id
        WHERE r.method='bon_candidate' AND r.criterion=? AND r.rewriter_model=?
    """, (args.criterion, rewriter)).fetchall()
    print(f"  loaded {len(rows)} candidates (criterion={args.criterion}, rewriter={rewriter}) from DB", flush=True)
    if not rows:
        print(f"  no candidates for criterion={args.criterion} × rewriter={rewriter}! run bon_generate first", flush=True)
        return

    # Skip-existing: if all eval-set docs already have a bon_panel row for this
    # (fold, criterion, rewriter), there's nothing to do. Panel scoring takes
    # ~5 min × 2 judges = 10 min and loads two judge engines, so bail early.
    if args.resume_skip_existing:
        distinct_docs = len(set(r[1] for r in rows))
        done_bon = conn.execute(
            "SELECT COUNT(DISTINCT source_doc_id) FROM attack_rewrites "
            "WHERE method='bon_panel' AND fold=? AND criterion=? AND rewriter_model=?",
            (fold, args.criterion, rewriter)
        ).fetchone()[0]
        if done_bon >= distinct_docs:
            print(f"  skip-existing: fold {fold} already has bon_panel for "
                  f"{done_bon}/{distinct_docs} docs — nothing to do", flush=True)
            conn.close()
            return

    if check_stop_signal(stop_file):
        print(f"  STOP detected before judge spawn — exiting", flush=True)
        conn.close()
        return

    # Judge server — spawn or reuse.
    if args.judge_endpoint:
        judge_endpoint = args.judge_endpoint
        judge_server_proc = None
        print(f"  reusing judge server at {judge_endpoint}", flush=True)
    else:
        log_path = "/data/shil6647/attack-llm-judge/grpo_run/bon_judge_server.log"
        judge_server_proc, judge_endpoint = spawn_judge_server(
            port=args.judge_port, gpu=args.judge_gpu, log_path=log_path,
        )

    judges = [
        JudgeHTTP(name=slug, rubric=args.criterion, endpoint=judge_endpoint)
        for slug in in_panel_keys
    ]

    props = [r[3] for r in rows]
    texts = [r[2] for r in rows]
    score_per_judge = {}
    try:
        for j in judges:
            t0 = time.time()
            s = j.score(props, texts)
            print(f"  {j.wandb_name} scored {len(s)} in {(time.time()-t0)/60:.1f} min",
                  flush=True)
            score_per_judge[j.wandb_name] = s
    finally:
        # Always unload server-side to free VRAM, even if scoring failed.
        for j in judges:
            try:
                j.unload()
            except Exception:
                pass

    # Save per-judge scores to DB (use current criterion)
    for i, r in enumerate(rows):
        for jn, scores in score_per_judge.items():
            insert_judge_score(conn, rewrite_id=r[0], judge_slug=jn,
                                criterion=args.criterion, score=scores[i], reasoning=None)
    conn.commit()

    # Aggregate: per source_doc_id, mean across judges, pick argmax candidate
    by_doc = {}
    for i, r in enumerate(rows):
        doc_id = r[1]
        ej = sum(score_per_judge[jn][i] for jn in score_per_judge) / len(score_per_judge)
        by_doc.setdefault(doc_id, []).append((r[0], ej, r[2], i))

    saved = 0
    for doc_id, cands in by_doc.items():
        cands.sort(key=lambda x: x[1], reverse=True)
        top_rid, top_ej, top_text, top_i = cands[0]
        config = {"K": len(cands), "fold": fold, "in_panel": in_panel_keys,
                  "selected_candidate_rid": top_rid, "ej_score": top_ej,
                  "criterion": args.criterion}
        rid_winner = make_rewrite_id(doc_id, "bon_panel", f"{short}_fold{fold}_{args.criterion}")
        conn.execute("""
            INSERT OR REPLACE INTO attack_rewrites
            (rewrite_id, source_doc_id, method, fold, criterion, config_json,
             rewriter_model, judge_panel_json, text, word_count, run_metadata_json)
            VALUES (?, ?, 'bon_panel', ?, ?, ?, ?, ?, ?, ?, ?)
        """, (rid_winner, doc_id, fold, args.criterion,
               json.dumps(config), rewriter, json.dumps(in_panel_keys),
               top_text, len(top_text.split()),
               json.dumps({"selected_from_candidate_id": top_rid})))
        saved += 1
    conn.commit()
    conn.close()

    if judge_server_proc is not None:
        try:
            JudgeHTTP.request_shutdown(judge_endpoint, timeout=10.0)
        except Exception:
            try:
                judge_server_proc.terminate()
            except Exception:
                pass
        try:
            judge_server_proc.wait(timeout=60)
        except Exception:
            try:
                judge_server_proc.kill()
            except Exception:
                pass
    print(f"[{time.strftime('%H:%M:%S')}] bon_score fold {fold} done; saved {saved} winners", flush=True)


def main():
    require_config()
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    common_criterion = lambda p: p.add_argument(
        "--criterion", choices=["clarity", "informativeness"], default="clarity")
    common_rewriter = lambda p: p.add_argument(
        "--rewriter", type=str, default=None,
        help=f"override rewriter base model (default: {REWRITER})")
    common_stop = lambda p: p.add_argument(
        "--stop-file", type=str, default=None,
        help="Graceful-stop marker path. Touching this file triggers a clean "
             "exit at the next natural boundary (per method / per chunk / "
             f"before judge load). Defaults to a per-subcommand path under {DEFAULT_STOP_DIR}.")
    common_resume = lambda p: p.add_argument(
        "--resume-skip-existing", action="store_true",
        help="Skip rewrites already present in attack_rewrites for the target "
             "(method, criterion, rewriter[, fold]). Re-run a previous command "
             "with this flag to resume after a STOP without redoing work.")

    p_ff = sp.add_parser("feedback_free")
    p_ff.add_argument("--methods", nargs="+", default=None)
    common_criterion(p_ff)
    common_rewriter(p_ff)
    common_stop(p_ff)
    common_resume(p_ff)
    p_ff.set_defaults(func=cmd_feedback_free)

    p_bg = sp.add_parser("bon_generate")
    p_bg.add_argument("--k", type=int, default=16)
    p_bg.add_argument("--bon-chunk-size", type=int, default=8,
                      help="paragraphs per vLLM chunk — trades a small batching "
                           "efficiency hit for a STOP-check boundary every ~30s.")
    common_criterion(p_bg)
    common_rewriter(p_bg)
    common_stop(p_bg)
    common_resume(p_bg)
    p_bg.set_defaults(func=cmd_bon_generate)

    p_bs = sp.add_parser("bon_score")
    p_bs.add_argument("fold", type=int, choices=[1, 2, 3])
    common_criterion(p_bs)
    common_rewriter(p_bs)
    common_stop(p_bs)
    common_resume(p_bs)
    p_bs.add_argument("--judge-endpoint", type=str, default=None,
                       help="Reuse an already-running judge server.")
    p_bs.add_argument("--judge-port", type=int, default=8127)
    p_bs.add_argument("--judge-gpu", type=int, default=0,
                       help="Physical GPU for the spawned judge server.")
    p_bs.set_defaults(func=cmd_bon_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
