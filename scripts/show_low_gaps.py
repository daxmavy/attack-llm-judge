"""Show low-stance-gap GRPO rollouts — the rewrites that preserved the original register.

Pairs with show_stance_flips.py: this one picks the rollouts where stance_gap is
near 0, picks the presumed source (max fwd over same-proposition training cands),
and prints pairs for inspection.
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

    # Load all rollouts for full text
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

    def match(s):
        for r in rollouts:
            if (r["step"] == s["step"] and r["rollout_idx"] == s["rollout_idx"]
                    and abs(r["ensemble_judge"] - s["ensemble_judge"]) < 1e-6):
                return r
        return None

    # Pick 5 low-gap rollouts spread across training progress (step>=51 to avoid pre-training trivial matches),
    # and require fwd >= 0.7 so we're looking at "stayed on topic AND preserved register".
    candidates = [s for s in summary
                  if s["step"] >= 51 and s["max_fwd"] >= 0.7 and s["stance_gap"] < 0.1]
    candidates.sort(key=lambda s: s["stance_gap"])
    # Diversify across steps
    picked, seen_steps = [], {}
    for s in candidates:
        if seen_steps.get(s["step"], 0) < 2:
            picked.append(s)
            seen_steps[s["step"]] = seen_steps.get(s["step"], 0) + 1
        if len(picked) >= 5:
            break
    if len(picked) < 5:
        picked = candidates[:5]

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

    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, cache_dir=CACHE, dtype=torch.bfloat16
    ).to("cuda").eval()

    for i, s in enumerate(picked, 1):
        r = match(s)
        if r is None:
            continue
        prop = r["proposition"]
        cands = by_prop[prop]
        rew = [r["rewrite"]] * len(cands)
        fwd = batched_entail(tok, model, rew, cands)
        best_idx = int(np.argmax(fwd))
        original = cands[best_idx]

        single_fwd = batched_entail(tok, model, [r["rewrite"]], [original])[0]
        single_bwd = batched_entail(tok, model, [original], [r["rewrite"]])[0]
        r2p = batched_entail(tok, model, [r["rewrite"]], [prop])[0]
        o2p = batched_entail(tok, model, [original], [prop])[0]

        print("="*90)
        print(f"LOW-GAP {i}  |  step={r['step']}  rollout_idx={r['rollout_idx']}  ej={r['ensemble_judge']:.1f}  "
              f"len_ratio={r['length_ratio']:.2f}  run={r['_run']}")
        print(f"  fwd(rew→orig)={single_fwd:.3f}  bwd(orig→rew)={single_bwd:.3f}")
        print(f"  rewrite→prop={r2p:.3f}   orig→prop={o2p:.3f}   |gap|={abs(r2p-o2p):.3f}")
        print(f"\nPROPOSITION: {prop}")
        print(f"\nORIGINAL:")
        print(textwrap.fill(original, width=90, initial_indent='  ', subsequent_indent='  '))
        print(f"\nGRPO REWRITE:")
        print(textwrap.fill(r["rewrite"], width=90, initial_indent='  ', subsequent_indent='  '))
        print()


if __name__ == "__main__":
    main()
