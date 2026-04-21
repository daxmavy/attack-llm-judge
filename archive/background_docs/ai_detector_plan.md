# AI-text detector plan (metric 6)

*Produced by a research sub-agent, 2026-04-17.*

## Chosen detector: Binoculars (Hans et al., ICML 2024)

**Repo:** https://github.com/ahans30/Binoculars
**Paper:** https://arxiv.org/abs/2401.12070
**Mechanism:** Two Falcon-7B models (base + instruct); score is log-perplexity of the base over cross-perplexity of the instruct. Lower raw score → more machine-like.

Beats RADAR (65.6%), roberta-openai-detector (59.2%) on RAID; comparable to Fast-DetectGPT but slightly better at low-FPR operating points on news-like text.

## Install + minimal use

```bash
git clone https://github.com/ahans30/Binoculars.git && cd Binoculars && pip install -e .
# Accept TII Falcon license on HF first: tiiuae/falcon-7b and tiiuae/falcon-7b-instruct
```

```python
from binoculars import Binoculars
import numpy as np
bino = Binoculars()
raw = np.array(bino.compute_score(paragraphs))  # lower = more machine-like
# Calibrate to P(machine) via sigmoid((thr - raw)/tau) fitted on a labeled subset.
```

**Do NOT** use Binoculars' built-in `predict()` — its threshold is tuned for a different distribution. Fit your own logistic regression on a labeled subset (~500 human + 500 known-AI paragraphs) from paul_data (which already has that split).

## Compute

- Falcon-7B + Falcon-7B-Instruct in bf16: ~28 GB VRAM peak.
- Throughput: 2–4 s per 100 paragraphs at batch 32.
- Full corpus (~50k paragraphs × 11 = ~550k): ~4–8 GPU-hours on A100-40GB.

## Known failure modes

1. **Short-text variance** — Binoculars' headline AUROCs use 512 tokens; ~100–130 words ≈ 150–200 tokens, where variance rises. Report histograms, not just means.
2. **Paraphrase-attack leakage** — aggressive rewriters can drive AUROC toward chance (PADBen 2025, StealthRL 2026). For us this is the signal: a rewriter that moves "more human" under Binoculars may be exploiting the detector rather than producing truly human prose. Cross-reference with the other fidelity metrics (embedding sim, agreement drift).
3. **OOD domain** — Binoculars was calibrated on news/essays; political opinion writing is distinct. Calibrate thresholds on paul_data's writer vs. model split, not the authors' defaults.
4. **Repetition / quoted text** — low-entropy spans (slogans, quotes) confound two-model ratio.

## Honest caveat

All AI-detectors are imperfect; Binoculars' authors themselves advise against unsupervised use. Treat the score as a **distributional** signal across the 50k × 11 matrix — never as a verdict on any single paragraph. False-positive rates of 1–5% on adversarial human text are normal; ESL writers in particular can score as AI.

## Backup

**Fast-DetectGPT** (github.com/baoguangsheng/fast-detect-gpt). Zero-shot, single-model, ~14 GB VRAM with GPT-Neo-2.7B. Swap in if Falcon licensing or shared-A100 VRAM becomes a problem.
