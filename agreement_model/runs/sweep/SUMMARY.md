# Agreement-score regression: hold-out strategy sweep

Model: `microsoft/deberta-v3-base` fine-tuned as regressor (fp32, 3 epochs, bs=32, lr=1e-5). Target: aggregated rater agreement score `paul_data/prepared/documents.csv::agreement_score` (0-1).

Single fixed proposition-level 80/10/10 split (seed=42). Test set: 1005 paragraphs across 10 unseen propositions. Only the training-set filter varies between runs. Subgroup metrics let us read held-out-distribution performance off the same test set.

## Overall test metrics

| Experiment | n_train | MAE | RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| baseline_all | 8007 | 0.0895 | 0.1221 | 0.923 | 0.900 |
| human_only | 3604 | 0.0995 | 0.1318 | 0.910 | 0.876 |
| human_plus_edited | 4403 | 0.0893 | 0.1190 | 0.930 | 0.902 |
| human_plus_claude | 4791 | 0.0880 | 0.1166 | 0.928 | 0.899 |
| human_plus_gpt | 4806 | 0.0900 | 0.1184 | 0.929 | 0.896 |
| human_plus_deepseek | 4819 | 0.0930 | 0.1229 | 0.926 | 0.897 |
| ai_only | 3604 | 0.0913 | 0.1242 | 0.920 | 0.896 |
| loo_claude (held out: claude) | 6021 | 0.0860 | 0.1160 | 0.933 | 0.905 |
| loo_gpt (held out: gpt-4o) | 6006 | 0.0871 | 0.1153 | 0.932 | 0.904 |
| loo_deepseek (held out: deepseek) | 5993 | 0.0903 | 0.1191 | 0.929 | 0.901 |
| claude_only | 1187 | 0.1802 | 0.2224 | 0.711 | 0.706 |
| gpt_only | 1202 | 0.1430 | 0.1856 | 0.817 | 0.791 |
| deepseek_only | 1215 | 0.1822 | 0.2284 | 0.683 | 0.679 |

## Plan-item answers (item 2 of `plan.md`)

**Train on human-only, evaluate on AI-rewritten (from `human_only`):**

| AI model at test | MAE | Pearson |
|---|---:|---:|
| Claude Sonnet 4 | 0.0846 | 0.941 |
| GPT-4o | 0.0906 | 0.929 |
| DeepSeek V3 | 0.0945 | 0.930 |
| (AI paragraphs, aggregated) | 0.0885 | 0.938 |
| Human writers (in-distribution) | 0.1116 | 0.874 |

A human-only-trained model transfers to AI-written paragraphs essentially as well as (actually slightly better than) it does on held-out human paragraphs. The AI models look like an *easier* distribution to score than humans — probably because human writers are more heterogeneous and contain unusual stance expressions.

**Train on human + 1 AI model, evaluate on the two held-out AI models:**

| Training AI | Held-out AI model | MAE | Pearson |
|---|---|---:|---:|
| Claude | GPT-4o | 0.0849 | 0.938 |
| Claude | DeepSeek | 0.0762 | 0.953 |
| GPT-4o | Claude | 0.0765 | 0.951 |
| GPT-4o | DeepSeek | 0.0818 | 0.953 |
| DeepSeek | Claude | 0.0745 | 0.954 |
| DeepSeek | GPT-4o | 0.0901 | 0.939 |

All transfer numbers sit in the same 0.07–0.09 MAE / 0.93–0.95 Pearson band. Held-out AI performance is actually **better** than the `human_only` baseline on the same held-out AI (Pearson +0.01-0.02), i.e. the model has already seen roughly the right AI-paragraph style by seeing any one AI source.

**Leave-one-AI-out (train on human + 2 AI, test on the third):**

| Held out | MAE | Pearson |
|---|---:|---:|
| Claude | 0.0704 | 0.956 |
| GPT-4o | 0.0849 | 0.942 |
| DeepSeek | 0.0830 | 0.955 |

Very small drop (or no drop) vs in-distribution. Strong evidence that the agreement signal transfers across AI providers.

## Key takeaways

1. **Human-only training already generalizes well to AI paragraphs.** MAE≈0.089 on AI paragraphs, Pearson≈0.94, barely worse than the baseline-all model which *does* see AI paragraphs during training (MAE=0.075, Pearson=0.95).
2. **Adding any AI data closes most of the remaining gap.** Going from human_only to human+1-AI improves AI-paragraph MAE from 0.089 to 0.075–0.083 and Pearson from 0.938 to 0.949–0.955, with minimal dependence on *which* AI model is in training.
3. **The hardest subgroup is human writers**, not AI.** All configs have higher MAE on paragraph_type=writer than on paragraph_type=model. The extra variance in how humans encode stance explains this.
4. **AI-only training does not break on human paragraphs** (ai_only → writer Pearson 0.88, MAE 0.103), but is strictly worse than training with any human data.
5. **Single-model training (claude_only, gpt_only, deepseek_only) is the one clear failure mode:** MAE doubles (0.14-0.18) and Pearson drops to 0.68-0.82. You need *breadth* of stance signal more than you need volume — `ai_only` (3.6k rows across three models) handily beats `gpt_only` (1.2k rows).

## Production pick

`loo_claude`-style training — humans + 2-of-3 AI models — is the best overall (MAE=0.086, overall Pearson=0.933). But the main 4-epoch model trained on everything (`runs/main`, MAE=0.080, Pearson=0.937) is only marginally better than `human_only` on out-of-sample AI paragraphs and is the most conservative choice for scoring new LLM-rewritten outputs.
