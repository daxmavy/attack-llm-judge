"""Pick the 25 most-controversial propositions by stdev(human_agreement_score),
then within each, sample ~7 writer paragraphs stratified by clarity quintile.

Outputs JSON: /workspace/grpo_run/controversial_25_dataset.json with fields
  {
    "propositions": [(proposition_id, proposition, stdev_agreement), ...],
    "all_rows": [(document_id, proposition, text, word_count, human_mean_clarity), ...],
    "n_train": N,
    "n_eval": N,
    "train_doc_ids": [...],
    "eval_doc_ids": [...],
  }

The `all_rows` order is: sorted by (proposition_id, document_id) deterministically.
Training consumers must split at n_train for train / [n_train:n_train+n_eval] for eval,
same pattern as the existing pipeline.
"""
from __future__ import annotations
import json
import random
import sqlite3
import statistics
from pathlib import Path


DB = "/home/max/attack-llm-judge/data/paragraphs.db"
OUT = Path("/workspace/grpo_run/controversial_25_dataset.json")

N_PROPOSITIONS = 25
TARGET_TRAIN = 130
TARGET_EVAL = 35
TARGET_TOTAL = TARGET_TRAIN + TARGET_EVAL  # 165 paragraphs
N_QUINTILES = 5
SEED = 17


def main():
    conn = sqlite3.connect(DB)

    # Step 1: rank propositions by stdev of human_agreement_score across writers
    cur = conn.execute("""
        SELECT proposition_id, proposition, document_id, text, word_count,
               human_mean_clarity, human_agreement_score
        FROM paragraphs
        WHERE origin_kind='original_writer'
          AND human_agreement_score IS NOT NULL
    """)
    rows = cur.fetchall()
    print(f"total writer rows: {len(rows)}")

    by_prop = {}
    for r in rows:
        by_prop.setdefault(r[0], {"proposition": r[1], "rows": []})["rows"].append(r)

    prop_stats = []
    for pid, d in by_prop.items():
        agrees = [r[6] for r in d["rows"] if r[6] is not None]
        if len(agrees) < 5:
            continue
        sd = statistics.pstdev(agrees)
        mn = statistics.mean(agrees)
        prop_stats.append((pid, d["proposition"], sd, mn, len(d["rows"])))

    prop_stats.sort(key=lambda t: -t[2])  # highest stdev first
    top25 = prop_stats[:N_PROPOSITIONS]
    print(f"top-{N_PROPOSITIONS} by stdev(agreement_score):")
    for i, (pid, prop, sd, mn, n) in enumerate(top25):
        print(f"  {i+1:2d}. pid={pid:3d} sd={sd:.3f} mean={mn:.3f} n={n:3d}  {prop[:70]}")

    # Step 2: assemble candidate rows from those 25 propositions
    selected_pids = {t[0] for t in top25}
    cand_rows = [r for r in rows if r[0] in selected_pids]
    print(f"candidate writer rows in 25 propositions: {len(cand_rows)}")

    # Step 3: global clarity quintiles (computed ON THE CANDIDATE SET so bins reflect the population we care about)
    clarities = sorted(r[5] for r in cand_rows if r[5] is not None)
    cut_idx = [int(len(clarities) * q / N_QUINTILES) for q in range(1, N_QUINTILES)]
    cuts = [clarities[i] for i in cut_idx]

    def quintile_of(c):
        for k, edge in enumerate(cuts):
            if c < edge:
                return k
        return N_QUINTILES - 1

    # Step 4: uniform stratified sample — for each proposition, pick ~7 paragraphs
    # balanced across clarity quintiles as best we can.
    per_prop_target = round(TARGET_TOTAL / N_PROPOSITIONS)  # 165/25 = 6.6 → 7
    print(f"per-proposition target: {per_prop_target} paragraphs")

    rng = random.Random(SEED)
    picked = []
    for pid, _, _, _, _ in top25:
        prop_rows = [r for r in cand_rows if r[0] == pid and r[5] is not None]
        buckets = [[] for _ in range(N_QUINTILES)]
        for r in prop_rows:
            buckets[quintile_of(r[5])].append(r)

        # Shuffle each bucket (seeded) and round-robin pick up to per_prop_target
        for b in buckets:
            rng.shuffle(b)
        out = []
        for bi in range(per_prop_target):
            for q in range(N_QUINTILES):
                if buckets[q]:
                    out.append(buckets[q].pop(0))
                    if len(out) == per_prop_target:
                        break
            if len(out) == per_prop_target:
                break
        picked.extend(out)

    print(f"picked {len(picked)} total paragraphs (target {TARGET_TOTAL})")

    # Trim or re-extend to exactly TARGET_TOTAL
    if len(picked) > TARGET_TOTAL:
        picked = picked[:TARGET_TOTAL]
    elif len(picked) < TARGET_TOTAL:
        # Fill with remaining candidates not already picked
        taken = {r[2] for r in picked}
        extras = [r for r in cand_rows if r[2] not in taken and r[5] is not None]
        rng.shuffle(extras)
        picked.extend(extras[: TARGET_TOTAL - len(picked)])
    assert len(picked) == TARGET_TOTAL

    # Deterministic order: by (proposition_id, document_id)
    picked.sort(key=lambda r: (r[0], r[2]))

    # Step 5: train/eval split. Disjoint by paragraph (NOT by proposition here —
    # propositions are already the primary axis; we want eval to cover the same
    # propositions so cross-prop generalisation is not confounded).
    # Shuffle with seed to avoid any document-id bias.
    idx = list(range(TARGET_TOTAL))
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(idx)
    train_idx = sorted(idx[:TARGET_TRAIN])
    eval_idx = sorted(idx[TARGET_TRAIN:])
    train_rows = [picked[i] for i in train_idx]
    eval_rows = [picked[i] for i in eval_idx]

    # Report clarity-quintile distribution in train and eval
    def quintile_distrib(rs):
        counts = [0] * N_QUINTILES
        for r in rs:
            counts[quintile_of(r[5])] += 1
        return counts

    print(f"train clarity quintile distrib: {quintile_distrib(train_rows)}")
    print(f"eval  clarity quintile distrib: {quintile_distrib(eval_rows)}")

    payload = {
        "n_propositions": N_PROPOSITIONS,
        "n_train": TARGET_TRAIN,
        "n_eval": TARGET_EVAL,
        "seed": SEED,
        "clarity_quintile_cuts": cuts,
        "propositions": [
            {"pid": pid, "proposition": prop, "stdev_agreement": sd, "mean_agreement": mn, "n_writers": n}
            for pid, prop, sd, mn, n in top25
        ],
        "rows": [
            {"document_id": r[2], "proposition_id": r[0], "proposition": r[1],
             "text": r[3], "word_count": r[4], "human_mean_clarity": r[5],
             "human_agreement_score": r[6], "split": ("train" if i < TARGET_TRAIN else "eval")}
            for i, r in enumerate(train_rows + eval_rows)
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
