"""Display 5 top stance-flip GRPO rollouts with their presumed source paragraph.

For each high-stance-gap rollout we find the training paragraph (same proposition)
with the highest fwd entailment — that's the rewriter's presumed source — and print
the (original, rewrite) pair side-by-side with scores.
"""
import json
import sqlite3
import textwrap
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
CACHE = "/data/shil6647/attack-llm-judge/hf_cache"
DB = "/home/shil6647/attack-llm-judge/data/paragraphs.db"
ENT_IDX = 0


def batched_entail(tok, model, premises, hypotheses, batch=16, max_length=512):
    probs = []
    for i in range(0, len(premises), batch):
        enc = tok(premises[i:i+batch], hypotheses[i:i+batch],
                  truncation=True, padding=True, max_length=max_length,
                  return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits
        p = torch.softmax(logits.float(), dim=-1)[:, ENT_IDX].cpu().numpy()
        probs.extend(p.tolist())
    return probs


def main():
    summary = json.load(open("/home/shil6647/attack-llm-judge/notebooks/rollouts_nli.json"))["rows"]
    # Top-5 by stance_gap
    top = sorted(summary, key=lambda r: -r["stance_gap"])[:5]

    # Load all rollouts so we can recover the full rewrite (summary truncated proposition)
    rollouts = []
    for rp in [
        "/data/shil6647/attack-llm-judge/grpo_run/pilot_qwen3_8b_fold1_clarity/rollouts.jsonl",
        "/data/shil6647/attack-llm-judge/grpo_run/pilot_qwen3_8b_fold1_clarity/rollouts.no-embed-sim.jsonl",
        "/data/shil6647/attack-llm-judge/grpo_run/pilot_qwen3_8b_fold1_clarity/rollouts.oom-dp.jsonl",
    ]:
        p = Path(rp)
        if not p.exists():
            continue
        tag = p.stem
        for line in p.open():
            r = json.loads(line)
            r["_run"] = tag
            rollouts.append(r)

    # Match summary rows back to full rollouts via (step, rollout_idx, ensemble_judge, length_ratio)
    def match(s):
        for r in rollouts:
            if (r["step"] == s["step"] and r["rollout_idx"] == s["rollout_idx"]
                    and abs(r["ensemble_judge"] - s["ensemble_judge"]) < 1e-6):
                return r
        return None

    full = [(s, match(s)) for s in top]

    # Load training paragraphs grouped by proposition so we can pick source candidates
    conn = sqlite3.connect(DB)
    db_rows = conn.execute("""
        SELECT proposition, text FROM paragraphs
        WHERE origin_kind='original_writer' AND writer_is_top_decile=1
    """).fetchall()
    conn.close()
    from collections import defaultdict
    by_prop = defaultdict(list)
    for prop, text in db_rows:
        by_prop[prop].append(text)

    # Score rewrite→candidate for each top row to pick the highest-matching original
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, cache_dir=CACHE, dtype=torch.bfloat16
    ).to("cuda").eval()

    for i, (s, r) in enumerate(full, 1):
        prop = r["proposition"]
        cands = by_prop[prop]
        rew = [r["rewrite"]] * len(cands)
        fwd = batched_entail(tok, model, rew, cands)
        best_idx = int(np.argmax(fwd))
        original = cands[best_idx]

        # Re-score this specific pair to be explicit
        single_fwd = batched_entail(tok, model, [r["rewrite"]], [original])[0]
        single_bwd = batched_entail(tok, model, [original], [r["rewrite"]])[0]
        r2p = batched_entail(tok, model, [r["rewrite"]], [prop])[0]
        o2p = batched_entail(tok, model, [original], [prop])[0]

        print("="*90)
        print(f"EXAMPLE {i}  |  step={r['step']}  rollout_idx={r['rollout_idx']}  ej={r['ensemble_judge']:.1f}  "
              f"len_ratio={r['length_ratio']:.2f}  run={r['_run']}")
        print(f"  fwd(rew→orig)={single_fwd:.3f}  bwd(orig→rew)={single_bwd:.3f}")
        print(f"  rewrite→prop={r2p:.3f}   orig→prop={o2p:.3f}   |gap|={abs(r2p-o2p):.3f}")
        print(f"\nPROPOSITION: {prop}")
        print(f"\nORIGINAL (presumed source, picked by max rewrite→orig over {len(cands)} train cands):")
        print(textwrap.fill(original, width=90, initial_indent='  ', subsequent_indent='  '))
        print(f"\nGRPO REWRITE:")
        print(textwrap.fill(r["rewrite"], width=90, initial_indent='  ', subsequent_indent='  '))
        print()


if __name__ == "__main__":
    main()
