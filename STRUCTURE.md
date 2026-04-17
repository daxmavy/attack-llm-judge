# STRUCTURE.md — Research question and approach

Single source of truth for what this project is testing and how. Covers the research question, threat model, methods, evaluation axes, and cross-references to the documents that own each implementation detail. Sections marked **[FILL IN]** belong to specific agents and should be edited only by them.

For per-iteration history (bugs, design-decision rationales, model swaps), see `EXPERIMENT_NOTES.md`.
For prior art, see `LITERATURE.md`.
For the headline-chart plan and what goes in the paper, see `PAPER.md`.
For model identities and exact slugs, see `MODELS.md` — that file is the source of truth for the judge panel and rewriter base.

---

## 1. Research question

**How effective is adversarial optimisation against LLM-as-a-judge when the attacker does not know which model is used as the judge?**

Operational form: an attacker can query *N* known proxy judges of its choosing while building its attack. The deployed judge is a different model the attacker never sees during training, selection, or RL. The same prompt/rubric is used for all judges — only model identity is unknown. We measure how much of the attacker's uplift on the proxy judges transfers to the held-out judge, and how much of *that* reflects genuine quality improvement vs. judge-specific gaming.

**Why this question.** A 6-day study cannot resolve unknown-judge attack robustness in general. It can resolve a narrower, well-defined slice: for one criterion (clarity), one domain (paragraph rewriting on UK political-opinion paragraphs from `paul_data`), and one held-out judge, how do five attack-method families compare on a transfer-gap-and-gold-decomposition basis that almost no prior work has applied to the rewriting-attack setting on LLM judges?

**Defensible contribution.** The combination is novel even though the components aren't:
- **Rewriting attacks on LLM-as-a-judge with cross-judge transfer.** Closest prior art: PAA (Kaneko 2026, arXiv:2601.06884) — search-based paraphrase, 3 reviewers in a matched/mismatched design, no training-based methods, no separate gold reference.
- **Training-based rewriters (SFT, GRPO) under held-out-judge conditions.** Closest prior art: 2406.03363 and 2604.12770 — RL rewriters for argument appropriateness, no held-out judge test, no gold reference.
- **Gold-panel decomposition of attacker uplift into "real quality" vs. "judge-specific gaming".** This is the canonical RM-overoptimisation framework (Gao 2210.10760, Coste 2310.02743, Eisenstein 2312.09244) but it has not been applied attacker-side to LLM judges in the rewriting setting. RobustJudge's iSDR is the closest, with task-specific quality metrics (BLEURT/CodeBLEU) rather than a gold LLM panel.
- **Political-opinion domain with stance preservation as a first-class drift axis.** Other rewriting-attack papers use semantic-similarity / perplexity drift; none use a domain-specific stance regressor.

The full combination is uncovered by any single paper.

**What we explicitly do not claim.** We are not the first to study unknown-judge robustness (RobustJudge, AutoGEO, multiple suffix-attack papers all touch parts of it). We are not the first to do gold-vs-proxy decomposition (RM-overoptimisation literature). We are not the first to do training-based rewriting against LLM rewards (AutoGEO-Mini, 2406.03363). The contribution is the joining of these in one controlled comparison.

**Acknowledged limitations** (carry through every claim in the paper):
- Single held-out judge → point estimate, no CI on the unknown-judge penalty.
- Single criterion (clarity) → tight dynamic range; a null result is a real possible outcome.
- Non-frontier proxy and held-out judges (attack panel ≈ 7–9B; held-out 9B; see `MODELS.md`).
- One domain. Generalisation to non-political content is not tested.
- Gold panel is family-diverse but cost-downgraded (Haiku 4.5, Flash, GPT-5-mini, DeepSeek V3.2, Llama-4-Maverick — see `MODELS.md`); not a frontier reference.

---

## 2. Threat model

| Attacker knowledge | Yes / No |
|---|---|
| Identity and weights of N proxy judges (their choice) | Yes |
| Prompt / rubric used by all judges | Yes |
| Identity or weights of the deployed (held-out) judge | **No** |
| Sample of held-out judge's outputs | **No** — never used in training, selection, or reward |
| Compute on the rewriter side | Bounded (1× A100-80GB; budget ≤ ~$30 OpenRouter for judging + ~tens of GPU-hours for training) |
| Read/write access to the content corpus | Yes — attacker rewrites human-written paragraphs |

**Concretely:** N=2 proxy judges. They are different model families (Llama + Qwen). Held-out is a third family (Gemma). All three sit in the same parameter-tier (~7–9B). See `MODELS.md` for slugs.

**Defence side:** the deployed judge has no defence beyond model choice. We don't test prompt randomisation, response perplexity filters, or detection ensembles. Adding any of those would be a separate study.

This is **not** the strongest attacker scenario possible. A truly best-case attacker would also know the held-out judge's family, decoding policy, system prompt defaults, and prompt wording exactly. We test "prompt-shared, model-unknown" — narrower than "best case," wider than "fully blind".

---

## 3. Criterion

**Clarity of expression**, single-criterion, scored 0–100. Rubric implementation: see `judge/rubrics.py` (owned by the evaluation agent).

We are aware (see `EXPERIMENT_NOTES.md`) that clarity has a tighter dynamic range than informativeness in pilot data, which raises the chance the unknown-judge penalty falls below judge-judge noise. Adding informativeness for the training-based arms is a low-cost scope extension that has been discussed but not committed.

---

## 4. Method panel

The evaluation suite scores **6 inference-only rewriter methods**, grouped by family. Two training-based methods (SFT, GRPO) are owned by the training agent and will be added to the same evaluation DB when their checkpoints are produced — see §6.2.

All inference-only methods share the same rewriter model (Qwen-2.5-72B-Instruct via OpenRouter), the same stance-preservation constraint, and a one-shot length retry targeting ±10% of the source word count. Prompts and orchestrators live in `rewriters/rewrite_prompts.py` and `rewriters/orchestrators.py`; full specs with prompts in `background_docs/methods_5_10_design.md`. Registered in the DB's `methods` table.

### 4.a Prompt-only, single-call (no judge feedback)

| Slug | Family | Rationale | Cost-per-paragraph |
|---|---|---|---|
| `naive` | Floor baseline | Unguided paraphrase. Measures the free uplift from "a capable LLM rewrites it" with no further steering. | ~1.3 calls |
| `lit_informed_tight` | Prompt heuristics + strict length | 5 principles from the rewriter-vs-judge literature (GEO, Prometheus, G-Eval): concrete specifics, visible reasoning, authoritative neutral register, main-claim-first, fix flow. Enforces ±10% word bounds, no headings/bullets. Current strongest non-judge-feedback prompt method on clarity+length fidelity. | ~1.3 calls |
| `rubric_aware` | Judge-prompt-known oracle | Rewriter is given the **literal clarity rubric** and asked to target the 81–100 anchor band. Oracle upper bound for "attacker knows the judge's rubric." Expected large gap between proxy judges (share the rubric) and a held-out judge with a paraphrased rubric. | ~1.3 calls |
| `injection_leadin` | Declared "obvious hack" | Rotates 3 authoritative lead-ins (`"Expert analysis: "`, etc.) across documents. Direct test of the "One Token to Fool LLM-as-a-Judge" (arXiv:2507.08794) vulnerability; measures detectability of surface-level attacks on small vs. frontier judges. | ~1.3 calls |

### 4.b Judge-feedback (inference-time search)

| Slug | Family | Algorithm | Cost-per-paragraph |
|---|---|---|---|
| `icir_single` | In-context iterative refinement (Pan et al. 2024; PAA, Kaneko 2026) | Seed with `lit_informed_tight`. Up to 4 iterations: score current rewrite with **both in-panel judges** (mean), concatenate their reasoning into the next rewriter prompt, revise. Stop on plateau (Δ < 2 points) or n_max. No absolute-score early-stop — the point is to measure how far the in-panel score climbs relative to held-out (in-context reward hacking). | ~15 calls |
| `bon_panel` | Best-of-N with proxy-judge ensemble (Gao 2023; Coste et al. 2024) | Sample N=8 candidates at T=1.0, top-p=0.95. Drop out-of-range candidates. Score each survivor with both in-panel judges. Pick argmax of mean, tie-break on lowest across-judge variance then fewest words. | ~10 rewrite calls + ≤16 judge calls |

### 4.c Training-based (out of eval-suite scope; to be added by training agent)

| Slug | Family | Status |
|---|---|---|
| `sft_bon` | SFT on BoN-selected pairs | Planned — training agent. BoN pipeline from §4.b provides the pairs. |
| `grpo_ensemble` | GRPO with mean-of-proxy-judges reward | Planned — training agent. Serves judges via local HF transformers per `MODELS.md`. |

When the training agent produces checkpoints, their rewrites get the same treatment in the `paragraphs` table (new `method_slug`), the same attack/gold scoring, and the same drift/detector metrics.

### Why this panel

The 6 inference-only methods span three axes:
- **Judge-facing vs. content-facing attacks** — `injection_leadin` and `rubric_aware` are judge-facing by design; `naive` and `lit_informed_tight` improve the content itself.
- **Inference-time search cost vs. quality** — single-call prompt methods vs. `icir_single` (≈15 calls/doc, iterative feedback) vs. `bon_panel` (≈10 rewriter + 16 judge calls/doc, parallel-with-selection).
- **Declared-hack vs. earnest-quality floor** — `injection_leadin` is the labelled obvious hack; the others sit in various shades of "improving the content for real" vs. "learning the rubric's surface cues."

Expected unknown-judge transfer (worst-to-best, from `background_docs/methods_5_10_design.md`): `bon_panel` → `icir_single` → `injection_leadin` → `rubric_aware` → `lit_informed_tight`. The `bon_panel` − `lit_informed_tight` gap on held-out judges is the cleanest single-number summary of the unknown-judge penalty this panel can produce.

**Pruned from the active set (2026-04-17):** `lit_informed`, `naive_tight`, `rules_explicit`, `scaffolded_cot_distill`. These were implemented and scored during pilot runs; rewrites still sit in the `paragraphs` table but are excluded from the active runner and the headline analysis. If we want them back for an appendix comparison it's a one-line runner change — the data is there.

Methods explicitly out of scope across the whole project: suffix-injection attacks on the *judge prompt* (different threat model — adversary controls a suffix, not the content), white-box gradient attacks on the judge, and reward-model distillation (the training agent uses online judge calls instead; cut-RM rationale in `EXPERIMENT_NOTES.md`).

---

## 5. Evaluation axes

Every method × paragraph cell produces:

1. **Held-out judge clarity score** (primary outcome).
2. **Each proxy judge's clarity score** (transfer-gap calculation: held-out vs. mean of proxies).
3. **Gold-panel mean clarity score** — 5 strong LLMs, none of them in the proxy or held-out set. Used to decompose attacker uplift into "real quality improvement" (held-out *and* gold both rise) vs. "judge-specific gaming" (held-out rises but gold doesn't, or vice versa).
4. **Drift axes** vs. the original paragraph:
   - Political stance shift (DeBERTa-v3 agreement-score regressor; Pearson 0.94 on held-out; weights at `agreement_model/runs/main/final/`).
   - Length change (word count delta and ratio).
   - Cosine similarity in e5-large-v2 sentence embedding.
5. **AI-generation detectability** — supervised TF-IDF + LogReg classifier trained on `paul_data` writer-vs-model split (held-out AUROC 0.989). See `EXPERIMENT_NOTES.md` for why we ended up with this rather than Binoculars / Fast-DetectGPT.

The *combination* of (1)+(2)+(3) is the central evidence chart: a four-quadrant decomposition of each method's score gain (held-out × gold = {+,+}, {+,−}, {−,+}, {−,−}). To my knowledge no published paper has produced this for rewriting attacks on LLM judges.

---

## 6. Pipelines and ownership

This project has two parallel implementation tracks. Neither agent should edit the other's section.

### 6.1 Evaluation pipeline — owned by **eval-suite** agent

**Inputs:** original paragraphs from `data/paragraphs.db`, a method specification, optionally a sample tag.
**Outputs:** rows in the `evaluations` table for every (paragraph, metric, criterion, source) tuple needed for §5.

#### Driver and CLI

One master driver orchestrates rewriter → attack panel → gold panel → non-LLM metrics → DeBERTa regressors:

```
cd /home/max/attack-llm-judge
# 1. One-time: initialise DB + ingest paul_data (creates 10,008 rows in `paragraphs`,
#    computes top-decile flag within each (proposition × agreement-score quintile)).
python3 -m eval_suite.ingest

# 2. One-time: create a stratified sample of human-written paragraphs.
#    `main_20pct` is 903 writers balanced across 5 quintiles × 2 top-decile classes.
python3 -m eval_suite.sampling --tag main_20pct --frac 0.20 --seed 17

# 3. Run the whole pipeline for a tag + criterion. Idempotent — re-runs skip
#    rows already present.
python3 -m eval_suite.run_all --sample-tag main_20pct --criterion clarity \
    --max-workers 24 [--skip-rewrites] [--skip-attack] [--skip-gold] \
                      [--skip-metrics] [--skip-regressors]

# 4. Per-method analysis (after run_all).
python3 -m eval_suite.analyse --sample-tag main_20pct --criterion clarity
#   -> data/analysis_main_20pct/: method_summary.csv,
#      attack_vs_gold_gap.csv, loo_attack.csv, loo_gold.csv,
#      by_method_headlines.csv, headlines.json
```

Scope follows `scope_where()` in `eval_suite/run_all.py`: sampled writer originals ∪ their rewrites. Everything keyed off `sampled_writers.tag`.

#### DB schema (`eval_suite/schema.py`)

SQLite, WAL-journaled at `data/paragraphs.db`. Five tables; long-format `evaluations` table is the key design choice so new metrics / criteria / judges append rows instead of altering columns.

- `paragraphs` — one row per original or rewrite. `origin_kind ∈ {original_writer, original_model, original_edited, rewrite}`. `base_document_id` links rewrites to their source writer. Human rating aggregates (`human_mean_clarity`, `human_mean_informativeness`, `human_agreement_score`, `n_human_ratings`) populated on originals. Writer-only columns: `writer_is_top_decile`, `writer_agreement_quintile`.
- `methods` — rewriter configs (slug, model, temperature, n_samples, iterations, uses_length_retry).
- `evaluations` — long-format. PK `(paragraph_id, metric, criterion, source)`. Columns: value, panel ∈ {attack, gold, held_out, null}, extra_json. `INSERT OR REPLACE` gives free idempotency.
- `criteria` — clarity (in_scope=1), informativeness (in_scope=0 for now, parameterised so turning it on is one config row plus a re-run).
- `sampled_writers` — tagged subsample membership (tag, document_id, stratum, quintile, top_decile).

Rewrite document IDs are deterministic: `rw_<method>_<base_document_id>`, so re-runs are naturally idempotent.

#### Sampling strategy actually used

`main_20pct`: 903 of the 4,503 human-written paragraphs (20%), stratified proportionally across the 10 strata = 5 per-proposition agreement quintiles × 2 top-decile classes. Each stratum yields ~20 top-decile + ~160 non-top-decile rows, so the top-decile proportion in the sample matches the population (~11%).

The full 4,503 writers × 10 methods ≈ 45k rewrites was **not** feasible under the \$100 gold-panel budget (projected \$279 with the originally-spec'd panel, \$97 even with the downgraded Haiku + Flash panel). The 20% subsample keeps the full method × judge matrix intact and costs ~\$22 on the gold panel.

#### Judge runner (`eval_suite/judge_runner.py`)

- **Identity.** Attack panel (2 judges): `judge_qwen7b` (Qwen-2.5-7B-Instruct) + `judge_llama8b` (Llama-3.1-8B-Instruct). Held-out panel (1): `judge_gemma9b` (Gemma-2-9B-It). Gold panel (5): Claude-Haiku-4.5, Gemini-2.5-Flash, GPT-5-mini, DeepSeek-V3.2, Llama-4-Maverick. Full slugs, pricing, and substitution rationale in `MODELS.md` and `eval_suite/panels.py`.
- **Idempotency.** `(paragraph_id, metric='judge_score', criterion, source=<judge slug>)` PK blocks duplicate scoring; re-runs skip existing rows unless `--force`.
- **Serving.** All judges called via OpenRouter HTTPS (decoupled from local GPU, which the training agent owns). Reasoning models (Gemini-2.5-Pro, GPT-5-mini, DeepSeek-R1) get `reasoning.effort=low` and `max_tokens=500` so hidden reasoning tokens don't starve the answer — hardcoded in `judge/client.py::REASONING_MODELS`.
- **Retry behaviour.** 3 retries with exponential backoff on 429/5xx. Also retries once without `response_format=json_object` if a provider 400s on it. Treats 200-OK-with-no-`choices` as a transient error (we observed OpenRouter returning error objects with `200` under provider quota blips — the fix was committed 2026-04-17).
- **Parallelism.** `ThreadPoolExecutor(max_workers=24)` shared across judges. Empirically 7–8 calls/s steady-state across the 5-judge gold panel; 2.9 calls/s across the 2-judge attack panel (smaller panel = lower concurrency saturation since each judge has its own provider queue). No 429s observed at 24 workers on either panel. ICIR scores its 2 in-panel judges in parallel too (`rewriters/orchestrators.py::_score_with_panel`).
- **Per-call cost ballpark** (700 in + 60 out tokens). Attack panel: ~\$0.000023/call (Qwen-7B) – \$0.000032/call (Llama-8B). Gold panel: ~\$0.00014/call (Gemini-Flash, DeepSeek, Llama-4-Maverick) up to \$0.0008/call (Haiku-4.5). Full attack + gold pass on the 20% sample: ~\$22.

#### Metric modules (`eval_suite/metrics/`)

Each module exposes `.score()` that writes to `evaluations` and `.benchmark()` that reports labelled-split performance on `paul_data` (per operator instruction — every metric must be validated where defensible).

| Metric | Module | Source key in DB | Benchmark on paul_data |
|---|---|---|---|
| LLM judge scores | `judge_runner` | judge slug (attack + gold) | Pearson vs `human_mean_clarity` per judge and per origin_kind, produced after run_all by `eval_suite/benchmarks.py::judges_vs_humans`. |
| Word count + ratio vs base | `metrics/word_count.py` | `tokenizer_whitespace` | Distributional stats. |
| Embedding cosine similarity to base | `metrics/embedding.py` | `e5-large-v2` | Parallel edited-vs-writer pairs as a "true human rewrite" anchor; random-pair anchor as floor. |
| Hallucinated specifics (regex + spaCy NER) | `metrics/hallucinated_specifics.py` | `regex+spacy_en_core_web_sm` | Rate-per-100w on edited-vs-writer pairs as rewrite baseline. |
| Agreement-score prediction | `metrics/deberta_regressor.py::score_agreement` | `agreement_model_deberta_v3_base` | Pearson 0.937 (held-out test; trained by the agreement-model agent). |
| Clarity-score prediction | `metrics/deberta_regressor.py::score_clarity` | `clarity_regressor_deberta_v3_base` | **Pearson 0.758, MAE 0.072** (held-out test; writer cell 0.68, model cell 0.74). Trained 2026-04-17 on `mean_human_clarity` via `criterion_model/train_clarity.py`. |
| AI-generation probability | `metrics/ai_detector_supervised.py` | `tfidf_logreg_paul_split` | **AUROC 0.989** held-out group-by-proposition. Per-AI-model: Claude 0.996, DeepSeek 0.993, GPT-4o 0.978. Pickled model at `data/ai_detector_tfidf.pkl`. |
| (secondary) Fast-DetectGPT zero-shot | `metrics/ai_detector.py` | `fastdetectgpt_qwen2.5_1.5b` | AUROC 0.57 — too weak; kept for reference. Binoculars (Falcon-7B+Instruct, 28GB) didn't fit disk. See `EXPERIMENT_NOTES.md`. |

#### Cost ledger (as of 2026-04-17)

| Phase | Spend |
|---|---|
| Pilot rewriter runs + judge baselines (v1/v2_tight) | ~\$0.30 |
| Supervised AI-detector training + scoring all originals | ~\$0 (CPU + GPU time only) |
| Clarity regressor training (1× A100-80GB, 4 epochs, ~6 min) | ~\$0 |
| Rewrites for 10 methods × 903 paragraphs (main_20pct) | ~\$2.95 |
| Attack panel on main_20pct scope (19,844 calls) | \$0.47 |
| Gold panel on main_20pct scope (48,260 calls, in progress) | projected \$20 |
| **Total so far** | **~\$25 of \$100 budget** |

#### Known failure modes hit (each written up in `EXPERIMENT_NOTES.md`)

- Live OpenRouter slugs diverged from plan.md / sub-agent suggestions (retired mistral-7b, gemma-2-9b, phi-3.5-mini; claude-sonnet-4.5 vs 4.6; deepseek-chat renamed; llama-3.1-405b never hosted). Rebased panel against OpenRouter's live model list.
- Gemini-2.5-Pro and GPT-5-mini emit hidden reasoning tokens; default max_tokens=300 truncated answers to "Here is the JSON requested" stubs. Fixed with `reasoning.effort=low` + max_tokens=500 for models in `REASONING_MODELS`.
- NVIDIA Nemotron Nano 9B returns empty content under `response_format=json_object` (provider bug). Swapped for Grok-3-mini in the attack panel. [Later superseded when the panel shrank to 2 judges per MODELS.md.]
- OpenRouter occasionally returns `200` with an `error` object and no `choices` under provider quotas — crashed rewriter and judge clients until both were hardened to treat as a transient.
- DeBERTa-v3 + bf16 NaN'd from step 1 on the clarity regressor; forced fp32 + explicit `torch_dtype=float32` at model load.
- Binoculars (recommended by the research sub-agent) couldn't fit — Falcon-7B pair is ~28 GB and the overlay disk is only 20 GB. Tried Fast-DetectGPT with GPT2-XL (AUROC 0.60) then Qwen-2.5-1.5B (0.57) — zero-shot needs a scoring LM of comparable capability to the AIs being detected, which we don't have disk for. Pivoted to a supervised TF-IDF classifier (AUROC 0.989) and documented the corpus-specificity caveat.

#### Analysis outputs

`eval_suite/analyse.py --sample-tag main_20pct --criterion clarity` produces:
- `data/analysis_main_20pct/method_summary.csv` — one row per (method, metric, criterion, source, panel) with mean value and mean delta vs. base original.
- `attack_vs_gold_gap.csv` — gold_minus_attack per method/criterion (the unknown-judge-penalty proxy when the attack panel was the training signal).
- `loo_attack.csv`, `loo_gold.csv` — within-panel leave-one-out: mean rewrite score on each held-out judge vs. mean on the remaining judges.
- `by_method_headlines.csv` — one row per method with all key numbers (attack/gold clarity, word_count, embed_sim, hallucinated_specifics, detector_p_machine, agreement_pred, clarity_regressor_pred) side-by-side.
- `headlines.json` — machine-readable summary of the above.

The four-quadrant decomposition chart in §5 is built from the `by_method_headlines.csv` columns `gold_minus_attack` (unknown-judge gaming) and `delta_clarity_regressor` (domain-learned quality change) for each method.

### 6.2 Training pipeline — owned by **training** agent

**Inputs:** paragraphs from `data/paragraphs.db`, the 2 proxy judges (online via OpenRouter or HF transformers per `MODELS.md`), training prompt set.
**Outputs:** trained rewriter checkpoints for the SFT and GRPO arms; rollout / rollout-eval logs sufficient to reproduce the training curves; rewrites scored by the held-out judge and gold panel for the eval set.

**[FILL IN — training agent]**
- Current SFT recipe (data construction from BoN, loss, LR, batch, epochs, where the checkpoint lands).
- Current GRPO recipe (G, KL coefficient, learning rate, reward aggregation across the 2 proxy judges, rollout temperature, how many epochs / training prompts, how online judge calls are batched).
- How the 2-proxy ensemble is aggregated into a scalar reward.
- Rollout infrastructure choice: OpenRouter API vs. local vLLM/HF for the judges, with the per-step throughput you actually see.
- Train/eval split for training prompts (e.g. disjoint 200/50 from top-decile writers, per `MODELS.md`).
- Diagnostics tracked per step: proxy reward, rollout perplexity, KL to reference, length distribution, plus the held-out-judge eval cadence (e.g. every N steps on the eval set).
- Reward-hacking tripwires (e.g. rising proxy reward with rising drift / Binoculars-style detector / hallucinated specifics).
- Known failure modes (link to `EXPERIMENT_NOTES.md`).
- Cost: API spend on online-judge calls + GPU-hours used.

### 6.3 Shared infrastructure (not owned by either agent in particular)

- `paul_data/` — corpus. Read-only.
- `agreement_model/` — stance regressor. Trained, frozen. Don't retrain.
- `MODELS.md` — judge panel and rewriter base. Source of truth.
- `EXPERIMENT_NOTES.md` — design-decision and bugfix log. Both agents append.
- `LITERATURE.md` — prior art. Both agents read; literature additions go through whoever is writing the paper.

---

## 7. Open decisions still on the user's plate

- Whether to add informativeness as a second criterion for the training-based arms only.
- Whether to re-centre the paper title on "political-opinion + stance-preservation under unknown LLM judges" (sharper framing) or keep the generic "unknown-judge robustness" framing (broader but more crowded prior art).
- Primary effectiveness metric: raw uplift on held-out judge, transfer efficiency (held-out / proxy uplift), or drift-penalised uplift. Recommend transfer efficiency.
- Pre-analysis plan committing primary vs. exploratory hypotheses before running training. Cheap insurance against forking-paths under the 6-day deadline.
- Whether the four pruned methods (`lit_informed`, `naive_tight`, `rules_explicit`, `scaffolded_cot_distill`) come back for an appendix comparison — rewrites are still in the DB, so it's a one-line runner change and an analyser re-run, no re-generation needed.
