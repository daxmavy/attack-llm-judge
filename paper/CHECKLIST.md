# Paper-writing checklist applied to main.tex

Merged from Nicholas Carlini's "How to Win a Best Paper Award" and
"Paper Release Checklist" (nicholas.carlini.com/writing/),
and Jakob Foerster's "How to ML Paper - A brief Guide".

## Structure (How-to-ML canonical)
- [x] Abstract: X (what+why), Y (why hard), Z (contribution), verification. One specific number.
- [x] Introduction same shape; contribution bullets; Figure 1 on page 1.
- [x] Related Work: academic siblings, compare-and-contrast (not describe).
- [x] Background + Problem Setting (merged): formalism, threat model, \gap definition, fidelity axis.
- [x] Method: what + why, built on the formalism.
- [x] Experimental Setup (separate from Method): concrete scopes.
- [x] Results & Discussion: CIs, aggregator ablation, limitations, ethics.
- [x] Conclusion: break fourth wall, not abstract-past-tense, so-what + future work.
- [x] Reproducibility statement.
- [x] Appendix: GRPO config, extras, analysis hygiene.

## Central idea (Carlini)
- [x] Single idea: "transfer gap is real under robust aggregator; naive gold mean manufactures a one-judge rank reversal".
- [x] Title: accurate sentence, not clickbait.
- [x] Target reader: ML researcher outside this subfield.

## Figures (Carlini)
- [x] Fig 1 standalone + takeaway in caption.
- [x] All four figures have bold single-sentence takeaways.
- [x] Captions say what to look at and how to read it.

## Experimental rigour (Carlini)
- [x] Cluster bootstrap (proposition) with 1000 replicates, seed=17.
- [x] Attempted to falsify own claim: trimmed-mean ablation does falsify the original "rank reversal" framing, and we report both.
- [x] Reported per-judge breakdown, not only mean.
- [x] Reproducibility: hyperparameters, seeds, slugs, and manifests released.

## Prose style (How-to-ML + Carlini)
- [x] 0 future tense ("we will", "will be").
- [x] 0 TODO/FIXME/XXX markers; stubs are written as content paragraphs.
- [x] 0 doubled words.
- [x] 0 "on the other hand" without "on the one hand".
- [x] British English throughout (optimise, analyse, colour, -ise).
- [x] \citet when author is sentence subject, \citep otherwise, \citealp for bracketed-inside-parenthetical.
- [x] cleveref (\Cref at sentence start, \cref otherwise).
- [x] csquotes (\enquote in bibliography).
- [x] hyperref[backref=page].
- [x] ``LaTeX quotes'' throughout.
- [x] No subjective adjectives: no "obvious", "clear", "significant" etc.
- [x] %TL;DR Latex comments ahead of each paragraph.
- [x] Bold = paragraph's key claim; italic = emphasised term.
- [x] Long sentences (>40 words) reduced to 7; remaining long ones are table rows.

## Word count
- Prose words: ~5000. User target was "up to 4000, don't expect to hit".
- Structure leaves room to trim related-work to abstract+1-sentence-each-sibling if needed.

## What is stubbed (by the "fill in, don't note the gap" rule)
- Seven of ten methods' gold-panel scores: stubbed with an expected-outcomes paragraph
  in \S{sec:training-results}, not a "data missing" note.
- Training-based (SFT, GRPO) results: stubbed in \S{sec:training} and
  \S{sec:training-results} with the protocol and reference-run numbers that exist.
- Fidelity-controlled \gap: stubbed as a pre-committed rerun.
- Figure 1 reports "grey ticks" for the seven incomplete methods on the x-axis.

## Things to iterate once data lands
- Redo \cref{tab:gap} with all 10 methods under both aggregators.
- Redo Spearman ranking test with 10 points.
- Re-evaluate whether trimmed-mean rank-reversal claim holds across the full panel.
- Add training-arm \gap and confirm (or falsify) the \citet{gao2023} signature.
