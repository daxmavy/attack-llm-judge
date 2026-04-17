AIM: build and evaluate different methods for rewriting paragraphs (containing political opinions) to achieve better results under LLM-as-a-judge.

There are multiple coding agents working in parallel in this environment; each agent is responsible for a specific task.

Evaluate each method based on two criteria:
1) How effective is it if we don't know the exact prompt or model used for the LLM-as-a-judge?
2) How much semantic drift does it introduce? (operationalised as 'level of agreement with the proposition')

Parts of project:
1) Create judge: Find a realistic LLM-as-a-judge rubric and set up LLM-as-a-judge (clarity, informativeness to proposition)
2) Agreement model: Estimate agreement_score for new documents robustly:
    - Can it estimate agreement_score accurately when used on out-of-sample models?
        - train against human-only and check performance on AI-rewritten
        - train against human-only and only 1 AI model, check performance on the other two

3) Judge-free rewriting: Build and evaluate judge-free rewriting methods: (do with highly capable but not-too-expensive model)
    - Ask LLM to rewrite the text (no further prompting)
    - Ask LLM to rewrite the text (prompt draws from existing findings in literature on improving judgement results)
    - 

Implementation details:
- Need to figure out best way to use GPUs - might need to move this onto a remote server that I rent
- All generations should attempt to control the word count so that it is approximately the same as the original paragraph to be re-written.
- .env file has OpenRouter key
- To test generalisability, start small: 2 LLM judges, and 2 prompts (one judging for 'clarity' and one judging for 'informativeness'), for 4 combinations total.
- LLM judge should return a score between 0 (bad) and 100 (good) for quality judgements.
- LLM judges: Llama 3.3 70B (~$0.10/$0.32 per 1M in/out) and Gemini 2.0 Flash (~$0.10/$0.40 per 1M in/out), both via OpenRouter. Held-out OOD judges for transfer evaluation: GPT-4o-mini and Claude Haiku 3.5.
- Base model for RL/SFT training (single A100 40GB): Qwen 2.5 1.5B — full fine-tuning fits without LoRA, fast GRPO loops, strong reasoning baseline at this scale. Similar-size comparison models: Llama 3.2 1B, Gemma 2 2B.

Known problems:
- Is it possible to estimate the deviation from human judgement accurately? We could learn a model to estimate `criterion` of paragraphs using paul_data - but problem is, very high-quality generations might be created, whose true quality is above that observed in the distribution used to train the quality-estimating model. This suggests that we shouldn't try to estimate the deviation between LLM judge estimates and the 'true' estimate; am leaving this out of scope for the moment.

*ACTIVE AGENTS* (write down when you started, what your goal is, and what you're working on):
- 2026-04-17 (agent: agreement-model, status: IN PROGRESS training):
    - Goal: plan item (2) — train agreement-score regressor on paul_data/prepared/documents.csv (proposition + paragraph -> agreement_score, 0-1 using aggregated rater stance, not writer self-report).
    - Files owned: `agreement_model/` (train.py, runs/, model checkpoints). Please don't modify.
    - External side effects: GPU (A100, ~15 GB used during training). Will release when run finishes.
    - Main run (runs/main): DeBERTa-v3-base fp32, 4 epochs, group-by-proposition 80/10/10. Test MAE=0.080, Pearson=0.937 (overall).
    - Current: hold-out sweep running (13 configs, 3 epochs each, ~3 hrs wall). Writes runs/sweep/summary.{json,csv}.