# PAPER.md — reporting plan

What goes in the paper. Owned by whoever is writing the write-up; both agents read.

For the research question and method panel, see `STRUCTURE.md`.
For the design-decision history those choices sit inside, see `EXPERIMENT_NOTES.md`.
For prior art and the positioning argument, see `LITERATURE.md`.

---

## Headline charts

Numbering is stable so other docs (e.g. `EXPERIMENT_NOTES.md`, `STRUCTURE.md`) can cross-reference H-N.

- **H1. Headline effectiveness.** Mean held-out-judge clarity uplift per method. 5 bars, cluster-bootstrap CI over propositions.
- **H2. Transfer gap.** Per method: mean proxy uplift vs. mean held-out uplift, paired by paragraph. Reference line = judge–judge disagreement on originals (noise floor).
- **H3. Gold-panel decomposition.** Per method: held-out uplift vs. gold-panel uplift. Methods above the diagonal are gaming the held-out judge more than the gold panel rates the rewrite as genuinely better. The four quadrants are the central novelty argument.
- **H4. Drift–uplift trade-off.** One sub-panel per drift axis (stance, length, cosine). Held-out uplift on x, drift on y. Method means and per-paragraph scatter.
- **H5. Detection rate.** AI-detector AUROC (or detection rate at fixed FPR) per method. Original-text reference.
- **H6. Method ranking reversal.** Spearman ρ between (method ranking on mean-of-proxies) and (method ranking on held-out judge). Bootstrap CI.
- **H7. Noise floor inset.** Three numbers: proxy-vs-proxy Pearson, proxy-vs-held-out Pearson, MDE at sample size.

Detailed evidence plan, including appendix charts, was previously written in `EVIDENCE.md` (now archived). The above seven survive into the paper; appendix items will be regenerated against the real outputs once both pipelines have produced data.
