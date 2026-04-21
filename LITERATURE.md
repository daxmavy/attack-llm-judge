# LITERATURE.md — prior art with rubric

Per-paper rubric for prior art relevant to the research question in `STRUCTURE.md`. Papers are organised by tier (relevance to "rewriting attacks on LLM judges with cross-judge transfer and gold-reference decomposition").

## Rubric columns

For each paper:

- **a. Goal of the paper** — what they set out to study, in one short sentence.
- **b. Attacks LLM-as-a-judge?** — does the paper attack an LLM judge specifically (vs. a safety classifier, retriever, recommender, RM, etc.)?
- **c. Held-out judges for transfer?** — does it test whether attacks crafted on judge A transfer to a different (unseen) judge B?
- **d. Gold-standard reference for genuine vs. gaming?** — does it use a separate gold judge, human labels, or other ground truth to decompose attacker uplift into "real quality improvement" vs. "judge-specific gaming"?
- **e. Attack / adversarial methods used.**

Additional axes I've added because they materially affect comparability with this project:

- **f. Attack target type** — *content rewriting* (paraphrase / regenerate the content being judged), *suffix injection* (append adversarial tokens), *prompt injection* (alter the judge's instructions via the content), *gradient / energy* (white-box optimisation), or *other*.
- **g. Training-based?** — does the attacker train a rewriter (SFT / RL / RM-distill), or is it inference-only (prompt / search / iteration)?
- **h. Drift / fidelity axis?** — does the paper measure how much the attack changed the underlying content (semantic similarity, perplexity, stance, factuality)?
- **i. Domain.** — what content the attack operates on.
- **j. Verified by me?** — `[verified]` if I fetched the paper / HTML and checked claims this conversation; `[unverified]` if I'm relying on a previous research-brief author or only an abstract.

If a column doesn't apply or the paper doesn't address it, write "—" rather than guessing.

---

## Tier 1 — directly tests cross-judge transfer of *rewriting-style* attacks

### Kaneko, "Paraphrasing Adversarial Attack on LLM-as-a-Reviewer (PAA)" — arXiv:2601.06884 (Jan 2026)
- **a.** Test whether iteratively paraphrasing scientific paper abstracts can inflate scores given by LLM peer reviewers, while preserving meaning.
- **b.** Yes — three LLM "reviewer" models (GPT-4o, Gemini 2.5, Sonnet 4) are the attacked judges.
- **c.** Yes — explicit "matched vs. mismatched" axis: PAA optimised against reviewer A is also evaluated on reviewers B and C. Quantitative result: matched mean score 4.3, mismatched 3.5, baseline 2.7 (so ~50% of the matched uplift survives the model swap).
- **d.** Partial. Human evaluation only assesses semantic equivalence and naturalness of the rewrite, not the rewritten paper's actual review quality. §4.3 compares attacked LLM-reviewer scores to actual ICLR human reviewer scores (one-sided "deviates"). No systematic decomposition of uplift into real-quality vs. gaming.
- **e.** Black-box ICL search — previous (paraphrase, score) pairs guide candidate generation; 32 iterations × 8 candidates × 8 score samples (~2,048 reviewer queries per abstract). BERTScore (τ_sim=0.85) and perplexity (α_ppl=1.2) thresholds enforce drift constraints.
- **f.** Content rewriting (paraphrase).
- **g.** Inference-only (no SFT / RL).
- **h.** BERTScore + perplexity thresholds.
- **i.** Scientific peer review (5 ML/NLP conferences).
- **j.** [verified]

### Chen, Zhong, Kim & Xiong, "AutoGEO" — arXiv:2510.11438 (2025)
- **a.** Extract preference rules from generative search engines and use them to rewrite content for higher visibility.
- **b.** Indirectly — the "judge" is a generative search engine evaluating retrieved documents for inclusion in answers, not a clarity/quality judge.
- **c.** Yes — rules are extracted using one LLM (Gemini-2.5-pro/flash-lite) and applied to two others (GPT-4o-mini, Claude-3-haiku). Rule overlap 78–84% across pairs.
- **d.** No gold reference for "real vs. gaming"; the paper measures engine utility (KPR, Precision, Clarity, Insight) but as descriptive properties, not a gold quality reference.
- **e.** Two variants. AutoGEO-API: Explainer/Extractor/Merger/Filter pipeline + frontier-LLM rewrites. AutoGEO-Mini: Qwen3-1.7B fine-tuned with **GRPO** against compound reward (visibility + rule compliance + semantic preservation).
- **f.** Content rewriting (rule-conditioned and SFT+RL-distilled).
- **g.** Yes — both SFT cold-start and GRPO. Most developed public training-based rewriter pipeline.
- **h.** KPR / KPC semantic-preservation rewards.
- **i.** Generative search-engine retrieval (GEO-Bench, Researchy-GEO, e-commerce).
- **j.** [verified] (HTML)

---

## Tier 2 — cross-judge transfer of *non-rewriting* attacks on LLM judges

### Raina et al., "Is LLM-as-a-Judge Robust?" — arXiv:2402.14016 (EMNLP 2024)
- **a.** Test universal adversarial suffixes against zero-shot LLM judges on NLG evaluation tasks.
- **b.** Yes — judges are FlanT5 (surrogate), Llama2-7B, Mistral-7B, GPT-3.5 (held-out targets).
- **c.** Yes — surrogate→target setup: 4-token suffix learned on FlanT5-3B; transferred to the three larger models, achieving rank-1 (max scores) "irrespective of the input text" on SummEval and TopicalChat. Strongest published transfer result on judge attacks.
- **d.** Implicit — argues fakeness *by construction* (the universal suffix has no semantic content, and scores max out regardless of input). No human-vs-attacked correlation analysis; no separate gold LLM panel.
- **e.** Greedy word-by-word suffix optimisation. Also briefly tests GCG (worse). Tests both absolute scoring (very vulnerable) and comparative assessment (much more robust).
- **f.** Suffix injection.
- **g.** Inference-only.
- **h.** —
- **i.** SummEval (summarisation), TopicalChat (dialogue).
- **j.** [verified] (HTML)

### Shi et al., "JudgeDeceiver" — arXiv:2403.17710 (CCS 2024)
- **a.** Optimisation-based prompt injection in attacker-controlled candidate responses to force selection by an LLM-as-a-Judge in pairwise/comparative settings.
- **b.** Yes — pairwise LLM judges across Llama-3-8B, Llama-2-13B, GPT-3.5, etc.
- **c.** Yes — transfer across models reported. 99% ASR Llama-3-8B → Llama-2-13B; 70% ASR → GPT-3.5; lower against >70B models.
- **d.** Implicit reductio — attacker deliberately picks objectively bad target responses (generates "incorrect, illogical, malicious, absurd" responses with GPT-3.5-turbo and chooses the worst), so any selection is provably gaming. No systematic gold-vs-attacked comparison.
- **e.** Gradient-based optimisation of an injected suffix in the attacker's response. White-box on training-time judge.
- **f.** Suffix injection (in the candidate response, not as content rewriting).
- **g.** Per-instance optimisation; not a trained rewriter policy.
- **h.** —
- **i.** RLAIF / LLM-search / tool-selection candidate ranking.
- **j.** [verified] (HTML)

### "Adversarial Attacks on LLM-as-a-Judge Systems" — arXiv:2504.18333 (April 2025)
- **a.** Survey of prompt-injection attacks against judge systems with a transferability headline.
- **b.** Yes.
- **c.** Yes — transferability quoted at 50.5–62.6% across architectures; 22.4–35.2% between open-source and frontier; smaller models more vulnerable (65.9% ASR).
- **d.** Not specified in available content.
- **e.** Prompt-injection family.
- **f.** Prompt injection.
- **g.** Inference-only as far as the abstract reveals.
- **h.** —
- **i.** Text quality, code correctness, argument strength.
- **j.** [unverified — abstract only; full paper not fetched]

### Li et al., "Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks" — arXiv:2505.13348 (2025)
- **a.** Suffix-injection attacks against judge architectures.
- **b.** Yes — Qwen2.5-3B-Instruct, Falcon3-3B-Instruct.
- **c.** Limited — only open-source small judges tested; no clear surrogate→held-out setup in the abstract.
- **d.** —
- **e.** Adversarial suffix appended to one of the candidate responses. ASR >30%.
- **f.** Suffix injection.
- **g.** Inference-only.
- **h.** —
- **i.** General response evaluation.
- **j.** [unverified — abstract only]

### Wu et al., "RobustJudge / LLMs Cannot Reliably Judge (Yet?)" — arXiv:2506.09443 (2025)
- **a.** First benchmark explicitly designed to assess adversarial robustness of LLM judges across 15 attack methods × 12 judges × 7 defences.
- **b.** Yes — directly the topic.
- **c.** **No** — attacks are evaluated against each judge in isolation; no train-on-judge-A, evaluate-on-judge-B experiment. Optimisation-based attacks (PAIR, TAP, GCG, AutoDAN, AdvEval, Cheating, Greedy) use white-box access to the target judge's loss; heuristic attacks (naïve, escape-chars, context-ignoring, etc.) are judge-agnostic by construction.
- **d.** Yes — introduces **iSDR** (Improved Score Difference Rate), which subtracts a task-specific content-quality score change (BLEURT for text, CodeBLEU for code) from the attacker's raw score gain. The closest precedent in the judge-attack literature for the gold-reference decomposition this project plans to do, although iSDR uses an automatic quality metric rather than a gold LLM panel.
- **e.** 15 attacks across heuristic and optimisation families.
- **f.** Mixed (suffix, injection, optimisation).
- **g.** Inference-only.
- **h.** Yes — iSDR has a quality-change subtraction.
- **i.** General text evaluation + code.
- **j.** [verified] (HTML)

### Zhao et al., "One Token to Fool LLM-as-a-Judge" — arXiv:2507.08794 (2025)
- **a.** Show generative reward models / LLM judges have "master key" inputs (single tokens like ":" or generic openers like "Thought process:") that elicit false-positive rewards.
- **b.** Yes — including GPT-o1 and Claude-4 among many other models.
- **c.** Implicit — same trivial inputs work across many judges (broad cross-judge transfer of the trivial vulnerability).
- **d.** Yes — the "false positive" framing assumes a ground-truth correctness reference (RLVR / verifiable rewards).
- **e.** Single-token / short-string injection.
- **f.** Suffix / prompt injection (trivial).
- **g.** Inference-only (no optimisation needed).
- **h.** —
- **i.** RLVR-style verifiable-correctness tasks.
- **j.** [verified] (search summary)

### Pavlakos et al. (?), "Can You Trick the Grader? Adversarial Persuasion of LLM Judges" — arXiv:2508.07805 (2025)
- **a.** Test seven Aristotelian persuasion techniques as score-inflators on LLM judges scoring math reasoning.
- **b.** Yes.
- **c.** Across six math benchmarks; abstract doesn't specify cross-judge transfer.
- **d.** Yes — uses verifiable math correctness as gold; persuasion inflates scores by ~8% on *incorrect* solutions, where the gain is provably fake.
- **e.** Persuasion techniques (Majority, Consistency, Flattery, Reciprocity, Pity, Authority, Identity) embedded in responses.
- **f.** Content insertion (persuasive language); not a full rewrite.
- **g.** Inference-only.
- **h.** —
- **i.** Math reasoning.
- **j.** [verified — abstract via search]

---

## Tier 3 — benchmark cheating / evaluator-bias exploits without judge-targeted optimisation

### Zheng et al., "Cheating Automatic LLM Benchmarks: Null Models" — arXiv:2410.07137 (ICLR 2025 oral)
- **a.** Show that constant null-model responses cheat AlpacaEval 2.0 / Arena-Hard / MT-Bench.
- **b.** Yes — LLM-as-judge based benchmarks.
- **c.** Same null model wins across multiple benchmarks (cross-evaluator transfer of the trivial attack).
- **d.** Yes — null-model responses are by construction zero-quality; any high win rate is provably fake. No separate gold judge needed; the null is the gold.
- **e.** Null model + structured adversarial prefixes.
- **f.** Content replacement (not rewriting; the attack ignores the input entirely).
- **g.** Inference-only; some version uses a small optimisation pass on the prefix.
- **h.** —
- **i.** General instruction-following benchmarks.
- **j.** [verified] (HTML)

### Dubois et al., "Length-Controlled AlpacaEval" — arXiv:2404.04475 (2024)
- **a.** Identify and debias length bias in LLM-as-judge benchmarks.
- **b.** Yes (defender side).
- **c.** —
- **d.** Yes (uses human reference rankings as gold).
- **e.** Documents how longer responses get higher scores; debiasing via regression.
- **f.** Defender measurement, not attack.
- **g.** —
- **h.** —
- **i.** AlpacaEval.
- **j.** [unverified — search summary]

### Zhang et al., "From Lists to Emojis: How Format Bias Affects Model Alignment" — arXiv:2409.11704 (2024)
- **a.** Show LLM evaluators preferentially reward bold/list/exclamation formatting.
- **b.** Yes.
- **c.** Across several judges in their bias survey.
- **d.** Implicit — bias is shown by holding content quality constant.
- **e.** Format manipulation in responses.
- **f.** Content rewriting (formatting only).
- **g.** Inference-only.
- **h.** —
- **i.** General LLM alignment.
- **j.** [unverified]

### Pan, Chen & Zou, "Spontaneous Reward Hacking in Iterative Self-Refinement" — arXiv:2407.04549 (2024)
- **a.** Document in-context reward hacking (ICRH) in essay editing — online judge scores rise monotonically while held-out judges and humans diverge.
- **b.** Yes — both online judge (in-loop) and held-out judges + humans (gold).
- **c.** Yes — online vs. held-out comparison is the core result.
- **d.** Yes — held-out judges and human scores serve as gold.
- **e.** Iterative in-context refinement (judge gives feedback; rewriter revises).
- **f.** Content rewriting (essay editing, iterative).
- **g.** Inference-only.
- **h.** —
- **i.** Essay editing.
- **j.** [unverified — research-brief reference, not personally fetched]

---

## Tier 4 — RL-trained rewriters against LLM-judge / classifier rewards (closest training-side precedent)

### "Teaching LLMs Human-Like Editing of Inappropriate Argumentation via RL" — arXiv:2604.12770 (April 2026)
- **a.** Train a rewriter that edits arguments for appropriateness using GRPO with a multi-component reward.
- **b.** Yes — reward signals come from classifiers and quality scorers acting as judges.
- **c.** No — no held-out-judge generalisation test reported.
- **d.** No clear gold-vs-proxy decomposition.
- **e.** GRPO with multi-component reward (semantic similarity, fluency, pattern conformity, argument-level appropriateness).
- **f.** Content rewriting (sentence-level edit suggestions).
- **g.** Yes — RL-trained rewriter. **Methodologically the closest precedent for the GRPO arm in this project.**
- **h.** Edit-level semantic similarity in the reward.
- **i.** General argumentation.
- **j.** [unverified — abstract only]

### "LLM-based Rewriting of Inappropriate Argumentation using RL from Machine Feedback" — arXiv:2406.03363 (2024)
- **a.** Train a rewriter for argument appropriateness with RL from classifier feedback.
- **b.** Reward source is a classifier, not an LLM judge per se.
- **c.** No held-out-classifier test reported in abstract.
- **d.** Some absolute and relative human assessment; not a systematic gold-vs-proxy decomposition.
- **e.** RL with classifier-based reward; instruction-finetuned LLM as initial policy.
- **f.** Content rewriting.
- **g.** Yes — earlier RL-rewriter precedent.
- **h.** "Content preservation" criterion (not specified in detail).
- **i.** General argumentation.
- **j.** [unverified — abstract only]

### "Rewrite to Jailbreak: Discover Learnable and Transferable" — ACL Findings 2025
- **a.** Train a rewriter (SFT then RL) that rewrites benign prompts into jailbreaks.
- **b.** No — target is the safety-aligned victim model, not an LLM judge.
- **c.** Yes — extensively tested cross-model. 40–60% transfer of jailbreak success on unseen targets.
- **d.** Implicit — jailbreak success is binary; transfer of harmful-output generation is the gold.
- **e.** SFT cold-start then RL for higher attack-success.
- **f.** Content rewriting (the prompt is the content).
- **g.** Yes — methodologically very close to this project's SFT→GRPO arc, but for safety not quality-rewriting.
- **h.** Jailbreak-success classifier.
- **i.** Safety jailbreaking.
- **j.** [verified] (PDF; partial)

### Ho et al., "Rewrite-to-Rank" — arXiv:2507.21099 (2025)
- **a.** SFT + PPO for ad-visibility rewriting (semantic+fidelity loss).
- **b.** Yes (LLM ranker as judge).
- **c.** Likely no held-out evaluator (Algoverse student paper; details from research-brief).
- **d.** Not clear.
- **e.** SFT + PPO.
- **f.** Content rewriting.
- **g.** Yes.
- **h.** Semantic + fidelity loss in SFT.
- **i.** Ad ranking.
- **j.** [unverified — research-brief only]

---

## Tier 5 — RM overoptimisation / proxy-vs-gold (lineage for the gold-decomposition methodology)

### Gao, Schulman & Hilton, "Scaling Laws for Reward Model Overoptimization" — arXiv:2210.10760 (2023)
- **a.** Foundational empirical study of how RL/BoN against a proxy RM diverges from a "gold" RM.
- **b.** Reward models, not LLM judges; methodologically the canonical proxy-vs-gold setup.
- **c.** Yes — gold RM is held out from policy training; serves as the transfer target.
- **d.** Yes — gold RM is the gold reference. Hump-shaped curves are the canonical "uplift on proxy diverges from uplift on gold" diagnostic.
- **e.** PPO and BoN.
- **f.** Policy outputs (no rewriting; full generation).
- **g.** Yes — RL.
- **h.** —
- **i.** Synthetic preferences.
- **j.** [unverified — search summary]

### Coste et al., "Reward Model Ensembles Help Mitigate Overoptimization" — arXiv:2310.02743 (2024)
- **a.** Test whether ensembling proxy RMs reduces overoptimisation against gold.
- **b.** RMs.
- **c.** Yes — gold RM is held out.
- **d.** Yes — explicit proxy-vs-gold curves per optimisation step.
- **e.** PPO and BoN with ensemble proxies + KL regularisation.
- **f.** —
- **g.** —
- **h.** —
- **i.** AlpacaFarm setup.
- **j.** [verified]

### Eisenstein et al., "Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking" — arXiv:2312.09244 (2023)
- **a.** Show that ensembling RMs *doesn't* eliminate reward hacking because all members share systematic errors.
- **b.** RMs.
- **c.** Yes — RL against an ensemble proxy, evaluated against held-out RMs / human-aligned gold.
- **d.** Yes — RM-vs-gold tracking.
- **e.** PPO; ensembles varying by either fine-tuning seed or pretraining seed.
- **f.** —
- **g.** RL training.
- **h.** —
- **i.** Helpfulness, summarisation.
- **j.** [verified] (search summary)
- **Direct relevance to this project:** finding that *pretraining-seed diversity* matters more than *fine-tuning-seed diversity* for ensemble robustness motivates choosing 2 proxy judges from different model families (Llama + Qwen, per `MODELS.md`).

### Moskovitz et al., "Confronting Reward Model Overoptimization with Constrained RLHF" — arXiv:2310.04373 (2023)
- **a.** Composite-RM overoptimisation: derive thresholds at which optimisation against a composite of correlated RMs starts to hurt.
- **b.** RMs.
- **c.** Yes — gold reference.
- **d.** Yes.
- **e.** Constrained PPO.
- **g.** RL training.
- **j.** [unverified — research-brief reference]

### Eisenstein et al. follow-ups — "Catastrophic Goodhart: regularizing RLHF with KL divergence" (NeurIPS 2024)
- **a.** Show KL regularisation provably bounds proxy-target divergence under RL.
- **d.** Gold reference vs. proxy.
- **g.** RL.
- **Direct relevance:** justifies the KL coefficient choice in the GRPO arm.
- **j.** [unverified — search summary]

### Laidlaw et al., "Correlated Proxies: A New Definition and Improved Mitigation for Reward Hacking" — arXiv:2403.03185 (2024)
- **a.** Define reward hacking as breakdown of proxy-target correlation under optimisation; propose χ²-divergence regularisation.
- **b.** RMs.
- **c.** Yes — held-out target.
- **d.** Yes — formal proxy-target correlation framework.
- **g.** RL.
- **j.** [verified]
- **Direct relevance:** gives the formal vocabulary for what "ranking reversal" means in the H6 chart.

### Beyer et al. (?), "Detecting Proxy Gaming in RL and LLM Alignment via Evaluator Stress Tests" — arXiv:2507.05619 (2025)
- **a.** Defender-side framework that detects proxy gaming via invariance-preserving perturbations.
- **b.** Yes — covers LLM judges (2 judges + 1,200 human-annotated instances in the LLM domain).
- **c.** Cross-judge agreement is part of the framework.
- **d.** Yes — content-driven improvements vs. exploitable-sensitivity separation.
- **e.** Defender, not attacker.
- **g.** —
- **h.** Perturbation-based.
- **i.** RL + LLM alignment.
- **j.** [verified]
- **Direct relevance:** this project's gold-panel decomposition is the *attacker-side mirror* of EST's defender-side detection.

### Khalaf et al., "Inference-Time Reward Hacking in Large Language Models" — arXiv:2506.19248 (2025)
- **a.** BoN provably suffers from a winner's curse; introduces Best-of-Poisson and HedgeTune.
- **b.** RM/judge as scorer.
- **d.** Gold-vs-proxy at inference.
- **g.** Inference-time.
- **j.** [unverified — research-brief]

### "Scaling Laws for RM Overoptimization in DAA" — arXiv:2406.02900 (2024); "Reward Model Overoptimisation in Iterated RLHF" — arXiv:2505.18126 (2025); etc.
- Group of follow-ups that all use proxy-vs-gold framing. Useful background.
- **j.** [unverified]

---

## Tier 6 — judge robustness / agreement / reliability (defender-side context)

### "A Coin Flip for Safety: LLM Judges Fail to Reliably Measure Adversarial Robustness" — arXiv:2603.06594 (March 2026)
- **a.** Show LLM judges drop to near-random under adversarial-distribution shifts in safety evaluation.
- **b.** Defender side — LLM judges evaluating model outputs that have been adversarially perturbed.
- **c.** Tests judge agreement under distribution shift, but the attack is on the *target model under evaluation*, not on the judge.
- **d.** Yes — 6,642 human-verified labels as gold.
- **e.** Various jailbreak attacks on victim models.
- **f.** —
- **g.** —
- **h.** —
- **i.** Safety / harmfulness evaluation.
- **j.** [verified — abstract]
- **Direct relevance:** strongest published evidence that single-judge uplift can be entirely artefactual; cite for the validity argument behind H7 noise floor.

### "Pairwise or Pointwise? Evaluating Feedback Protocols for Bias in LLM-Based Evaluation" — arXiv:2504.14716 (April 2025)
- **a.** Compare pointwise (absolute) and pairwise feedback protocols for bias robustness.
- **b.** Defender measurement.
- **c.** —
- **d.** Yes — gold via human labels.
- **e.** Various distractor features.
- **g.** —
- **j.** [unverified]
- **Direct relevance:** finds pairwise flips ~35% of the time vs ~9% for absolute; **contradicts Raina et al.** which says absolute is *more* vulnerable to suffix attacks. Different vulnerability axes (order/position vs. suffix). Cite both to justify absolute-scoring choice in this project.

### "More is Less: The Pitfalls of Multi-Model Synthetic Preference Data in DPO" — arXiv:2504.02193 (2025)
- **a.** Show that multi-model preference data can *facilitate* reward hacking in DPO via easier separability.
- **b.** Defender measurement (DPO training).
- **g.** Training (DPO).
- **j.** [verified — abstract]
- **Direct relevance:** competing prior to Eisenstein for the 2-proxy-ensemble design. Predicts multi-proxy may be *worse* than single proxy. This project can adjudicate empirically.

### "Shelf Life of Fine-Tuned LLM Judges: Future Proofing, Backward Compatibility, and Question Generalization" — arXiv:2509.23542 (2025)
- **a.** Test fine-tuned-judge generalisation across questions, generators, and time.
- **b.** Yes (defender side).
- **c.** Yes — held-out questions / generators.
- **d.** Human preference labels.
- **g.** Fine-tuned judges.
- **j.** [unverified]

### Shi et al., "Judging the Judges: A Systematic Investigation of Position Bias" — arXiv:2406.07791 (2024)
- **a.** Familial agreement and position bias in pairwise LLM judges.
- **b.** Defender measurement.
- **j.** [unverified — research-brief]

### "Judge's Verdict: Benchmarking 54 Judges" — arXiv:2510.09738 (2025)
- **a.** Benchmark 54 judges for human-correlation tier. Shifts from Pearson to Cohen's κ.
- **b.** Defender benchmark.
- **j.** [unverified]

### "LLM Jury-on-Demand" — arXiv:2512.01786 (2025)
- **a.** Mixed panels of judges beat any single judge on held-out evaluation.
- **b.** Defender.
- **j.** [unverified]

### Zheng et al., MT-Bench — arXiv:2306.05685 (2023)
- **a.** Foundational LLM-as-a-judge agreement study.
- **b.** Defender benchmark.
- **j.** [unverified — research-brief]

---

## Tier 7 — adjacent: GEO / SEO for LLMs (content-rewriting target is engine ranking, not judge score)

### Aggarwal et al., "GEO: Generative Engine Optimization" — arXiv:2311.09735 (KDD 2024)
- **a.** Define visibility metrics for content in generative engines; benchmark heuristic rewrites.
- **b.** Engine, not judge.
- **g.** Inference-only (prompted rewrites).
- **i.** RAG-style answer-engine retrieval.
- **j.** [unverified — research-brief]

### Nestaas, Debenedetti & Tramèr, "Adversarial SEO for LLMs" — arXiv:2406.18382 (2024)
- **a.** Inject adversarial instructions into documents to manipulate engine preferences.
- **b.** Engine.
- **f.** Prompt injection.
- **j.** [unverified — research-brief]

### "StealthRank" — arXiv:2504.05804 (2025)
- **a.** Energy-based / Langevin optimisation jointly for ranking, fluency, and n-gram penalty.
- **f.** Gradient / energy-based optimisation.
- **j.** [unverified — research-brief]

### Pfrommer et al., "TAP — Tree of Attacks with Pruning" — arXiv:2406.03589 (2024)
- **a.** Tree-search baseline for ranking manipulation.
- **j.** [unverified]

### "Role-Augmented Intent-Driven GSE Optimization (RAID G-SEO)" — arXiv:2508.11158 (2025)
- **a.** Intent-aware rewriting for generative search engines; semantic-drift-aware step planning.
- **f.** Content rewriting.
- **g.** Inference-only (prompted).
- **j.** [unverified — search summary]

### Qian et al., "Ranking Blind Spot / Decision Hijacking" — EMNLP 2025
- **a.** LLM rankers under adversarial content; nDCG@10 drops from 0.66 to 0.16 on Llama-3-8B listwise.
- **f.** Content injection.
- **j.** [unverified — research-brief]

---

## Tier 8 — RM distillation and RLHF recipes (training-side reference, mostly methodological)

### Lee et al., "RLAIF vs RLHF" — arXiv:2309.00267 (2023)
- LLM-judge labels as drop-in for human labels at scale. **j.** [unverified]

### Liu et al., "Skywork-Reward / V2" — arXiv:2410.18451, 2507.01352
- Reference open-source RM training recipe; 26M curated pairs; 0.6B–8B backbones. **j.** [unverified]

### Kim et al., "Prometheus / Prometheus-2" — arXiv:2310.08491, 2405.01535
- Generative judges trained on synthetic rubric-conditioned feedback. **j.** [unverified]

### "RM-Distiller" — arXiv:2601.14032 (2026)
- Distil generative LLM judges into RMs; beats naive Bradley-Terry. **j.** [unverified]

### "RM-R1" — arXiv:2505.02387 (2025)
- Reasoning-then-scoring RM with GRPO after distillation. Flags overfitting risk. **j.** [unverified]

### "CGPO / Mixture of Judges" — arXiv:2409.20370 (2024)
- Constrained RLHF with multiple judges; relevant to ensemble-RM robustness. **j.** [unverified]

### "Ask a Strong LLM Judge When Your RM Is Uncertain" — arXiv:2510.20369 (2025)
- Uncertainty routing between cheap RM and expensive LLM judge. **j.** [unverified]

### "GRM: Regularizing Hidden States" — arXiv:2406.10216 (2024)
- Hidden-state regularisation improves OOD RewardBench. **j.** [unverified]

### Xiao et al., "On Limited Generalization of Implicit RM" — EMNLP Findings 2024
- DPO-induced implicit RMs generalise much worse than explicit RMs OOD. **j.** [unverified]

### "AgentRM" — arXiv:2502.18407 (2025)
- Compares explicit RM, implicit RM, LLM-as-judge at the BoN selection step. **j.** [unverified]

### Lambert et al., "RewardBench" — arXiv:2403.13787; Malik et al., "RewardBench 2" — arXiv:2506.01937
- Canonical RM benchmarks; r=0.87 between RM accuracy and downstream BoN/RLHF performance. **j.** [unverified]

---

## Tier 9 — classic transferable attacks (canon, methodology only)

### Zou et al., "Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)" — arXiv:2307.15043 (2023)
- **a.** Foundational suffix-optimisation attack; surrogate ensemble (Vicuna-7B/13B) → transfer to GPT-3.5/4, Claude, Bard, etc.
- **b.** No — safety jailbreaking, not judges.
- **c.** Yes — transfer is the headline result.
- **e.** GCG (Greedy Coordinate Gradient).
- **f.** Suffix injection.
- **j.** [verified — search summary]

### "AmpleGCG" — generates transferable suffixes
- Trains a generator of GCG-style suffixes; 99% ASR on GPT-3.5 (closed-source target). **j.** [unverified]

### "IRIS" — improves universal/transferable attack via refusal suppression
- **j.** [unverified]

### "BERT-Attack" — arXiv:2004.09984 (2020); **GBDA** (gradient-based)
- Older transferable text-adversarial attacks. Mostly methodological background. **j.** [unverified]

---

## Tier 10 — domain context for political-opinion paragraphs

### Röttger et al. and follow-ups, "Measuring Political Bias in LLMs"
- Domain context for the `paul_data` source corpus. Not adversarial. **j.** [unverified]

### "PoliTune: Adapting LLMs to Express Political Ideologies"
- Domain context for political-opinion fine-tuning. **j.** [unverified]

### "Prompt-Based Clarity Evaluation and Topic Detection in Political QA" — arXiv:2601.08176 (2026)
- **a.** Use SemEval 2026 CLARITY dataset to evaluate clarity in political QA.
- **b.** Defender measurement.
- **c.** Compares two LLMs (GPT-3.5, GPT-5.2) under different prompting; not full cross-LLM agreement.
- **d.** Yes — human clarity annotations as gold.
- **e.** —
- **i.** Political QA, **clarity criterion** — domain + criterion match for this project.
- **j.** [verified — abstract]
- **Direct relevance:** sister dataset / domain. Worth citing as evidence the clarity-on-political-QA setup is a recognised research target.

### Brown / MIT / Promptfoo political-bias work; "Fair or Framed?" (EMNLP 2025); "LLM-generated messages can persuade humans on policy issues" (Nature Communications 2025)
- Domain context. Not adversarial against LLM judges. **j.** [unverified]

---

## Tier 11 — surveys (for citing the field shape, not for evidence)

- Li et al., "A Survey on LLM-as-a-Judge" — arXiv:2411.15594 (2024). **j.** [unverified]
- Gu et al., "LLMs-as-Judges: Comprehensive Survey" — arXiv:2412.05579 (2024). **j.** [unverified]
- Lilian Weng, "Reward Hacking in RL" — blog post (2024). Best synthesis reference. **j.** [unverified]
- "Reward Hacking in the Era of Large Models" — survey, arXiv:2604.13602 (2026). **j.** [unverified]

---

## Net positioning summary

After this literature pass, the open ground for the paper is the joint of:

1. **Training-based (SFT and/or GRPO) rewriter attacks** — closest precedent is 2604.12770 / 2406.03363, neither of which tests held-out judges or gold decomposition.
2. **Tested under a held-out-judge transfer condition** — closest precedent is PAA (2601.06884) for rewriting attacks, but PAA is inference-only.
3. **With a separate gold LLM panel decomposing uplift into real-quality vs. judge-specific gaming** — RobustJudge's iSDR is the only attack-on-judge precedent and it uses task-specific BLEURT/CodeBLEU rather than a gold LLM panel; the RM-overoptimisation lineage (Gao, Coste, Eisenstein, Laidlaw) has the methodology but for RMs, not LLM judges, and defender-side.
4. **In a domain (political opinion) where stance preservation is a domain-specific, first-class drift axis** — no rewriting-attack paper uses stance; PAA uses BERTScore + perplexity; AutoGEO uses KPR/KPC; 2601.08176 uses the same domain but doesn't attack.

The full conjunction is uncovered. Each pair of conditions has prior art; the four together do not.
