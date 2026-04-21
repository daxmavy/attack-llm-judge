# Overnight mission results — 2026-04-19/20

## Table sizes
- attack_rewrites: 41075 rows
- attack_judge_scores: 285800 rows
- attack_agreement_scores: 41075 rows

## Methods × criteria × fold (attack_rewrites)

| method | criterion | fold | rows |
|---|---|---|---|
| bon_candidate | clarity | - | 11424 |
| bon_candidate | informativeness | - | 11424 |
| bon_panel | clarity | 1 | 714 |
| bon_panel | clarity | 2 | 714 |
| bon_panel | clarity | 3 | 714 |
| bon_panel | informativeness | 1 | 714 |
| bon_panel | informativeness | 2 | 714 |
| bon_panel | informativeness | 3 | 714 |
| grpo_400step | clarity | 1 | 714 |
| grpo_400step | clarity | 2 | 714 |
| grpo_400step | clarity | 3 | 714 |
| grpo_400step | informativeness | 1 | 714 |
| grpo_400step | informativeness | 2 | 714 |
| grpo_400step | informativeness | 3 | 714 |
| icir | clarity | 1 | 714 |
| icir | clarity | 2 | 714 |
| icir | clarity | 3 | 714 |
| icir | informativeness | 1 | 714 |
| icir | informativeness | 2 | 714 |
| icir | informativeness | 3 | 714 |
| lit_informed_tight | clarity | - | 714 |
| lit_informed_tight | informativeness | - | 714 |
| naive | clarity | - | 714 |
| original | clarity | - | 1805 |
| rubric_aware | clarity | - | 714 |
| rubric_aware | informativeness | - | 714 |

## Judge × criterion score coverage (excluding bon_candidate)

| judge | criterion | scored rows |
|---|---|---|
| judge_gemma9b | clarity | 18227 |
| judge_gemma9b | informativeness | 18227 |
| judge_llama8b | clarity | 18227 |
| judge_llama8b | informativeness | 18227 |
| judge_mistral7b | clarity | 18227 |
| judge_mistral7b | informativeness | 18227 |
| judge_phi35mini | clarity | 18227 |
| judge_phi35mini | informativeness | 18227 |
| judge_qwen7b | clarity | 714 |
| judge_qwen95b | clarity | 18227 |
| judge_qwen95b | informativeness | 18227 |

## Agreement score means (DeBERTa-v3-base regressor; scale 0-1)

| method | criterion | n | mean | min | max |
|---|---|---|---|---|---|
| bon_candidate | clarity | 11424 | 0.666 | 0.098 | 1.0 |
| bon_candidate | informativeness | 11424 | 0.671 | 0.097 | 1.0 |
| bon_panel | clarity | 2142 | 0.689 | 0.102 | 1.0 |
| bon_panel | informativeness | 2142 | 0.682 | 0.106 | 1.0 |
| grpo_400step | clarity | 2142 | 0.771 | 0.106 | 1.0 |
| grpo_400step | informativeness | 2142 | 0.69 | 0.105 | 1.0 |
| icir | clarity | 2142 | 0.684 | 0.1 | 1.0 |
| icir | informativeness | 2142 | 0.716 | 0.101 | 1.0 |
| lit_informed_tight | clarity | 714 | 0.675 | 0.103 | 1.0 |
| lit_informed_tight | informativeness | 714 | 0.694 | 0.107 | 1.0 |
| naive | clarity | 714 | 0.586 | 0.095 | 1.0 |
| original | clarity | 1805 | 0.523 | 0.1 | 0.995 |
| rubric_aware | clarity | 714 | 0.611 | 0.097 | 1.0 |
| rubric_aware | informativeness | 714 | 0.611 | 0.096 | 0.989 |

## Mission-panel judge means (in-panel + held-out) by method/criterion

In-panel = average over the 2 judges used during GRPO training for each fold.
Held-out = the 3rd judge, not seen during training.

| method | rewriter criterion | fold | qwen95b-cla | qwen95b-inf | llama8b-cla | llama8b-inf | gemma9b-cla | gemma9b-inf |
|---|---|---|---|---|---|---|---|---|
| bon_panel | clarity | 1 | 94.06 | 27.97 | 85.74 | 51.84 | 84.15 | 53.24 |
| bon_panel | clarity | 2 | 94.55 | 27.89 | 80.86 | 49.35 | 84.72 | 52.31 |
| bon_panel | clarity | 3 | 90.37 | 27.86 | 85.66 | 51.85 | 84.79 | 52.79 |
| bon_panel | informativeness | 1 | 81.19 | 45.78 | 79.87 | 69.35 | 79.15 | 61.94 |
| bon_panel | informativeness | 2 | 83.33 | 47.42 | 79.45 | 63.17 | 80.58 | 65.5 |
| bon_panel | informativeness | 3 | 82.9 | 38.27 | 81.34 | 70.2 | 80.75 | 66.0 |
| grpo_400step | clarity | 1 | 94.78 | 26.98 | 86.09 | 54.14 | 84.76 | 56.9 |
| grpo_400step | clarity | 2 | 94.66 | 27.23 | 85.54 | 54.4 | 84.63 | 57.24 |
| grpo_400step | clarity | 3 | 94.79 | 27.54 | 86.08 | 55.08 | 84.65 | 58.37 |
| grpo_400step | informativeness | 1 | 92.12 | 31.37 | 82.56 | 56.65 | 82.97 | 57.18 |
| grpo_400step | informativeness | 2 | 93.8 | 32.82 | 84.96 | 59.57 | 84.42 | 62.24 |
| grpo_400step | informativeness | 3 | 93.7 | 31.98 | 85.02 | 60.63 | 84.18 | 61.92 |
| icir | clarity | 1 | 86.09 | 26.08 | 76.22 | 45.38 | 78.58 | 46.39 |
| icir | clarity | 2 | 85.31 | 25.47 | 76.84 | 45.18 | 78.45 | 46.23 |
| icir | clarity | 3 | 85.26 | 26.27 | 76.21 | 45.72 | 78.36 | 46.28 |
| icir | informativeness | 1 | 85.41 | 28.08 | 77.94 | 49.34 | 78.89 | 50.14 |
| icir | informativeness | 2 | 86.1 | 28.31 | 78.76 | 49.88 | 79.86 | 50.69 |
| icir | informativeness | 3 | 85.74 | 27.68 | 78.22 | 49.48 | 79.66 | 50.66 |
| lit_informed_tight | clarity | - | 86.21 | 26.85 | 76.79 | 47.19 | 78.76 | 47.43 |
| lit_informed_tight | informativeness | - | 86.29 | 27.09 | 78.18 | 48.18 | 79.01 | 48.88 |
| naive | clarity | - | 81.14 | 26.02 | 71.75 | 42.98 | 73.95 | 42.53 |
| original | clarity | - | 55.17 | 26.89 | 52.02 | 40.32 | 51.19 | 37.28 |
| rubric_aware | clarity | - | 82.79 | 26.08 | 73.38 | 44.43 | 75.32 | 43.9 |
| rubric_aware | informativeness | - | 82.38 | 27.02 | 73.47 | 45.39 | 74.83 | 45.44 |

## Out-of-sample judge means (Mistral-7B-v0.3, Phi-3.5-mini)

| method | rewriter criterion | fold | mistral7b-cla | mistral7b-inf | phi35-cla | phi35-inf |
|---|---|---|---|---|---|---|
| bon_panel | clarity | 1 | 79.7 | 72.91 | 89.35 | 68.14 |
| bon_panel | clarity | 2 | 79.7 | 71.25 | 88.72 | 68.0 |
| bon_panel | clarity | 3 | 79.73 | 73.06 | 89.3 | 67.35 |
| bon_panel | informativeness | 1 | 77.98 | 75.68 | 87.61 | 73.2 |
| bon_panel | informativeness | 2 | 77.83 | 75.73 | 87.77 | 73.73 |
| bon_panel | informativeness | 3 | 79.09 | 78.04 | 88.64 | 75.59 |
| grpo_400step | clarity | 1 | 80.64 | 76.78 | 90.96 | 74.59 |
| grpo_400step | clarity | 2 | 80.62 | 77.4 | 90.96 | 74.32 |
| grpo_400step | clarity | 3 | 80.78 | 77.7 | 91.11 | 74.71 |
| grpo_400step | informativeness | 1 | 78.52 | 73.37 | 89.1 | 72.35 |
| grpo_400step | informativeness | 2 | 79.93 | 77.38 | 90.24 | 75.23 |
| grpo_400step | informativeness | 3 | 80.31 | 78.69 | 90.22 | 76.8 |
| icir | clarity | 1 | 75.65 | 61.08 | 86.08 | 59.58 |
| icir | clarity | 2 | 75.66 | 61.95 | 86.34 | 59.58 |
| icir | clarity | 3 | 75.6 | 60.86 | 86.44 | 60.42 |
| icir | informativeness | 1 | 76.56 | 67.46 | 87.13 | 64.61 |
| icir | informativeness | 2 | 77.3 | 66.42 | 87.1 | 64.15 |
| icir | informativeness | 3 | 76.99 | 67.52 | 87.21 | 64.92 |
| lit_informed_tight | clarity | - | 75.54 | 62.8 | 86.63 | 60.78 |
| lit_informed_tight | informativeness | - | 75.54 | 64.55 | 86.87 | 62.96 |
| naive | clarity | - | 70.39 | 54.78 | 84.0 | 55.28 |
| original | clarity | - | 47.14 | 47.31 | 60.65 | 47.56 |
| rubric_aware | clarity | - | 71.3 | 57.03 | 84.68 | 56.8 |
| rubric_aware | informativeness | - | 70.31 | 57.67 | 84.19 | 58.44 |

## HF models pushed
- daxmavy/grpo-400step-fold1 — clarity fold 1
- daxmavy/grpo-400step-fold2 — clarity fold 2
- daxmavy/grpo-400step-fold3 — clarity fold 3
- daxmavy/grpo-400step-fold1-informativeness
- daxmavy/grpo-400step-fold2-informativeness
- daxmavy/grpo-400step-fold3-informativeness

HF round-trip verified for daxmavy/grpo-400step-fold1 (downloaded, loaded on CPU, generated).

## Run timeline (2026-04-19 → 2026-04-20)
- ~20:19 — fold 1 informativeness training done (135.6 min)
- ~23:15 — fold 2 informativeness training done (142.0 min)
- ~01:35 — fold 3 informativeness training done (109.5 min)
- 01:54-03:32 — big scoring pass (qwen95b 54 min, llama8b 14 min, gemma9b 18 min) — 3 judges × 2 criteria across all non-candidate rewrites
- 03:34-04:50 — ICIR across 3 folds × 2 criteria (iter=4; fold 1 together + folds 2/3 in separate processes due to vLLM memory-release quirk)
- 04:50-05:37 — score ICIR rewrites with mission panel
- 05:38-06:06 — Mistral-7B-v0.3 judge (23 min)
- 06:07-07:20 — Phi-3.5-mini judge (69 min; slower than expected)

## NOT COMPLETED (bonus tasks)
- 3rd out-of-sample judge (Gemma-4-26B-A4B-it / Ministral-8B-2512): did not attempt due to time risk — Gemma-4 requires isolated vLLM 0.19.1 env, Ministral-8B-2512 existence uncertain.
- G=8 / half-data continuation on fold 1 clarity: scoping showed ~4h run time, would not finish before wake.
- Different-base-rewriter redo: explicitly optional in mission; not attempted.
