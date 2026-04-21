"""Sample 100 flipped-stance rewrites for manual verification.

Flip definition (matches section 7.3b of analysis.ipynb):
- original agreement_score <= 0.25 and rewrite agreement_score >= 0.75, OR
- original agreement_score >= 0.75 and rewrite agreement_score <= 0.25.

Writes /home/max/attack-llm-judge/analysis/flipped_stance_sample.csv with columns:
proposition, original_paragraph, rewritten_paragraph,
original_agreement_score, rewritten_agreement_score, method, fold, rewrite_criterion, direction.
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB = Path('/home/max/attack-llm-judge/data/paragraphs.db')
OUT = Path('/home/max/attack-llm-judge/analysis/flipped_stance_sample.csv')
FLIP_LOW, FLIP_HIGH = 0.25, 0.75
N_SAMPLES = 100
SEED = 20260420

conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)

orig = pd.read_sql_query("""
    SELECT ar.source_doc_id, ar.text AS original_paragraph, aas.score AS original_agreement_score
    FROM attack_rewrites ar
    JOIN attack_agreement_scores aas ON aas.rewrite_id = ar.rewrite_id
    WHERE ar.method = 'original'
      AND ar.source_doc_id IN (SELECT DISTINCT source_doc_id FROM attack_rewrites WHERE method != 'original')
""", conn)

methods = ('naive', 'lit_informed_tight', 'rubric_aware', 'icir', 'bon_panel', 'grpo_400step')
rew = pd.read_sql_query(f"""
    SELECT ar.rewrite_id, ar.source_doc_id, ar.method, ar.fold,
           ar.criterion AS rewrite_criterion,
           ar.text AS rewritten_paragraph,
           aas.score AS rewritten_agreement_score
    FROM attack_rewrites ar
    JOIN attack_agreement_scores aas ON aas.rewrite_id = ar.rewrite_id
    WHERE ar.method IN {methods}
""", conn)

df = rew.merge(orig, on='source_doc_id', how='left')

flipped = df[
    ((df['original_agreement_score'] <= FLIP_LOW) & (df['rewritten_agreement_score'] >= FLIP_HIGH))
    | ((df['original_agreement_score'] >= FLIP_HIGH) & (df['rewritten_agreement_score'] <= FLIP_LOW))
].copy()

flipped['direction'] = [
    'anti_to_pro' if o <= FLIP_LOW else 'pro_to_anti'
    for o in flipped['original_agreement_score']
]
print(f'Total flipped rewrites available: {len(flipped)} '
      f'(anti_to_pro={sum(flipped["direction"]=="anti_to_pro")}, '
      f'pro_to_anti={sum(flipped["direction"]=="pro_to_anti")})')

# Attach proposition text from paul_data.
docs = pd.read_csv('/home/max/attack-llm-judge/paul_data/prepared/documents.csv',
                   usecols=['document_id', 'proposition_id', 'proposition'])
docs = docs.drop_duplicates('document_id').rename(columns={'document_id': 'source_doc_id'})
flipped = flipped.merge(docs[['source_doc_id', 'proposition']], on='source_doc_id', how='left')
assert flipped['proposition'].notna().all(), 'proposition text missing for some rows'

sample = flipped.sample(n=min(N_SAMPLES, len(flipped)), random_state=SEED).copy()

cols = [
    'proposition',
    'original_paragraph',
    'rewritten_paragraph',
    'original_agreement_score',
    'rewritten_agreement_score',
    'direction',
    'method',
    'fold',
    'rewrite_criterion',
]
sample = sample[cols].reset_index(drop=True)
sample.to_csv(OUT, index=False)
print(f'Wrote {len(sample)} rows to {OUT}')
print()
print('Sample breakdown:')
print(sample.groupby(['method', 'rewrite_criterion', 'direction']).size())
