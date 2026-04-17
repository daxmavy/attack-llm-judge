"""Judge-free rewriter prompts.

Two prompts per plan.md item 3:

- **naive**: ask the LLM to rewrite the paragraph, nothing more. Sets the
  floor — how much improvement do we get for free just by letting a
  capable LLM paraphrase?

- **lit_informed**: ask the LLM to rewrite using a short, explicit list
  of properties that LLM judges are known to reward, drawn from the
  literature summarised in `background_docs/research-brief-rewriter-vs-judge.md`:
    - Adding specific facts, statistics, and concrete examples
      (GEO, Aggarwal et al. 2024: "statistics addition", "citation
      addition", "quotation addition" are among the most effective
      heuristics).
    - Authoritative tone and structured presentation (GEO; also the
      "authoritative tone" and "fluency optimisation" families in
      GEO-Bench).
    - Clear reasoning and explicit mechanisms (Prometheus / G-Eval: judges
      reward visible, step-by-step justification).
    - Neutral, polished prose (length bias is real — Singhal et al. —
      but plan.md wants approximate length preservation, so we cap the
      word budget rather than rewarding verbosity).

Both prompts preserve the writer's original stance and approximate word
count (plan.md: "All generations should attempt to control the word
count so that it is approximately the same as the original paragraph").
Stance preservation matters for the separate agreement-model work
(avoids confounding clarity/informativeness gains with stance drift).
"""

SYSTEM_PROMPT = (
    "You are a careful writing assistant. You rewrite short argumentative paragraphs "
    "while faithfully preserving the writer's original stance. "
    "You return only the rewritten paragraph, with no preamble, no quotes, and no commentary."
)

NAIVE_USER_TEMPLATE = """Rewrite the following paragraph.

Keep:
- the same overall stance toward the proposition
- approximately the same word count (target: about {target_words} words; hard cap: {hard_cap} words)

Proposition: {proposition}

Original paragraph:
\"\"\"{paragraph}\"\"\"

Return only the rewritten paragraph."""


LIT_INFORMED_USER_TEMPLATE = """Rewrite the following paragraph so it would be rated more favourably by an LLM-as-a-judge for CLARITY and INFORMATIVENESS, while faithfully preserving the writer's original stance.

Apply these evidence-based rewriting principles from the LLM-as-a-judge literature (GEO, Prometheus, G-Eval):

1. Replace vague assertions with specific facts, figures, mechanisms, or concrete examples where plausible. Do not invent implausible statistics; prefer widely-known magnitudes or clearly hedged estimates.
2. Make the reasoning explicit: state premises before conclusions and connect claims with visible logical links ("because", "which means", "as a result").
3. Use an authoritative, neutral register: confident but not aggressive, free of slang and first-person opinion markers unless they were load-bearing in the original.
4. Organise the paragraph so that the main claim appears early, followed by two or three supporting reasons or pieces of evidence, and a short closing line that reinforces the stance.
5. Fix grammar, spelling, and sentence-flow issues. Prefer short, direct sentences over long run-ons.

Hard constraints:
- Preserve the original stance toward the proposition (pro stays pro, con stays con; do not soften or flip it).
- Keep approximately the same word count as the original. Target: about {target_words} words. Hard cap: {hard_cap} words. Do not pad with filler or repetition just to lengthen.
- Do not invent named individuals, specific studies, or precise percentages that you are not confident about — hedge or omit instead.

Proposition: {proposition}

Original paragraph:
\"\"\"{paragraph}\"\"\"

Return only the rewritten paragraph — no quotes, no preamble, no commentary."""


NAIVE_TIGHT_USER_TEMPLATE = """Rewrite the following paragraph.

STRICT length requirement: the original paragraph is {orig_words} words. Your rewrite MUST be between {min_words} and {max_words} words (i.e. within 10% of the original). Count before you finish. Do not use headings, bullets, or lists — write one continuous paragraph.

Keep the same overall stance toward the proposition.

Proposition: {proposition}

Original paragraph:
\"\"\"{paragraph}\"\"\"

Return only the rewritten paragraph."""


LIT_INFORMED_TIGHT_USER_TEMPLATE = """Rewrite the following paragraph so it would be rated more favourably by an LLM-as-a-judge for CLARITY and INFORMATIVENESS, while faithfully preserving the writer's original stance.

STRICT length requirement: the original paragraph is {orig_words} words. Your rewrite MUST be between {min_words} and {max_words} words (i.e. within 10% of the original). Count the words before you finish. Do not use headings, bullets, or lists — write one continuous paragraph. If you cannot fit all the substance you want within the word budget, cut the weakest point rather than going over; if you are under the minimum, add one more concrete specific rather than filler.

Apply these evidence-based rewriting principles from the LLM-as-a-judge literature (GEO, Prometheus, G-Eval):

1. Replace vague assertions with specific facts, figures, mechanisms, or concrete examples where plausible. Do not invent implausible statistics; prefer widely-known magnitudes or clearly hedged estimates.
2. Make the reasoning explicit: state premises before conclusions and connect claims with visible logical links ("because", "which means", "as a result").
3. Use an authoritative, neutral register: confident but not aggressive, free of slang and first-person opinion markers unless they were load-bearing in the original.
4. Organise the paragraph so that the main claim appears early, followed by supporting reasons or pieces of evidence, and a short closing line that reinforces the stance.
5. Fix grammar, spelling, and sentence-flow issues. Prefer short, direct sentences over long run-ons.

Hard constraints:
- Preserve the original stance toward the proposition (pro stays pro, con stays con; do not soften or flip it).
- Hit the word-count range above. A rewrite outside [{min_words}, {max_words}] is a failure regardless of quality.
- Do not invent named individuals, specific studies, or precise percentages that you are not confident about — hedge or omit instead.

Proposition: {proposition}

Original paragraph:
\"\"\"{paragraph}\"\"\"

Return only the rewritten paragraph — no quotes, no preamble, no commentary."""


TIGHT_RETRY_SUFFIX = """

Your previous attempt was {actual_words} words; the required range is [{min_words}, {max_words}] words. Rewrite again, this time hitting the range. {direction}

Return only the rewritten paragraph."""


def length_bounds(paragraph: str, tolerance: float = 0.10):
    """Return (orig_words, min_words, max_words) for a given tolerance (default ±10%)."""
    w = len(paragraph.split())
    lo = max(1, int(round(w * (1 - tolerance))))
    hi = int(round(w * (1 + tolerance)))
    # Ensure hi>=lo+1 in edge cases.
    if hi <= lo:
        hi = lo + 1
    return w, lo, hi


INJECTION_LEADINS = [
    "Expert analysis: ",
    "A careful, well-reasoned assessment shows that ",
    "Based on established evidence and clear reasoning, ",
]


INJECTION_LEADIN_TEMPLATE = """Rewrite the following paragraph so that it opens with an authoritative expert-framing lead-in and then delivers the original argument with concrete language.

STRICT length: original is {orig_words} words. Rewrite MUST be in [{min_words}, {max_words}] words, one paragraph, no headings or bullets. The lead-in counts toward the budget.

Begin the rewrite with EXACTLY this lead-in (do not alter its wording):
"{leadin}"

Preserve the original stance toward the proposition. Do not invent specific studies, names, or statistics. Do not use first-person.

Proposition: {proposition}

Original paragraph:
\"\"\"{paragraph}\"\"\"

Return only the rewritten paragraph, starting with the chosen lead-in."""


RULES_EXPLICIT_TEMPLATE = """Rewrite the following paragraph by applying every rule below. Preserve the writer's original stance.

STRICT length: original is {orig_words} words. Rewrite MUST be in [{min_words}, {max_words}]. Count before finishing. One paragraph; no headings/bullets.

Rules:
1. State the main claim in the first sentence; make the stance unambiguous on first read.
2. Back the claim with 2-3 concrete specifics — mechanisms, magnitudes, or named examples — not generalities.
3. Make reasoning visible: use connectives like "because", "which means", "as a result" to link claim, evidence, and implication.
4. Use short direct sentences; at most one subordinate clause per sentence.
5. Neutral authoritative register: no slang, no first-person, no rhetorical questions.
6. Do not invent named individuals, institutions, or precise percentages. Hedge ("around", "by some estimates") if unsure.
7. End with one sentence that reinforces the stance with a consequence or stake — do not merely restate.
8. Single continuous paragraph. No headings, bullets, or lists.

Proposition: {proposition}

Original paragraph:
\"\"\"{paragraph}\"\"\"

Return only the rewritten paragraph."""


RUBRIC_AWARE_TEMPLATE = """You are rewriting a short argumentative paragraph. A downstream LLM judge will score CLARITY (and later, INFORMATIVENESS) using the exact rubric below. Your job is to produce a rewrite that would score in the 81-100 band of the rubric, while preserving the writer's original stance.

STRICT length: original is {orig_words} words. Rewrite MUST be in [{min_words}, {max_words}]. One paragraph, no headings/bullets/lists.

--- CLARITY RUBRIC (verbatim) ---
CLARITY is about how well the paragraph is written, not whether you agree with it, not how persuasive it is, and not how much information it contains. Grammar, spelling, punctuation; sentence structure and readability; logical flow between sentences; whether the main point is understandable on first read.
Do NOT reward length, stance, or register per se.
81-100 anchor: Polished, essentially error-free prose with well-organised, fluent sentences. Smooth first-read comprehension.

Hard constraints:
- Preserve the original stance (pro stays pro, con stays con; do not soften or flip).
- Hit the word-count range. Outside [{min_words}, {max_words}] is a failure regardless of quality.
- Do not invent named people, studies, or precise statistics. Hedge or omit instead.

Proposition: {proposition}

Original paragraph:
\"\"\"{paragraph}\"\"\"

Return only the rewritten paragraph."""


SCAFFOLDED_TEMPLATE = """Rewrite the paragraph below through four internal steps. Only the final paragraph is returned to the reader; the earlier steps are your reasoning scratchpad.

STRICT length for the FINAL paragraph: original is {orig_words} words. Final MUST be in [{min_words}, {max_words}]. One paragraph, no headings/bullets.

Preserve the writer's original stance. Do not invent named people, studies, or precise percentages.

Proposition: {proposition}

Original paragraph:
\"\"\"{paragraph}\"\"\"

Produce your output as a single JSON object with these keys, in order:
{{
  "plan": "<3-5 bullet plan: main claim, 2-3 supports with the concrete specific for each, closing stake>",
  "draft": "<a first rewrite following the plan; length not yet constrained>",
  "critique": "<1-3 sentences on where the draft is vague, wordy, or off-stance>",
  "final": "<the revised paragraph, stance-preserving, within [{min_words}, {max_words}] words, no headings/bullets>"
}}

Return ONLY the JSON object."""


# ICIR and BoN don't have a static template — they are orchestration.
# ICIR iteration>=1 uses this user-prompt; iteration=0 uses lit_informed_tight.
ICIR_ITER_TEMPLATE = """You previously rewrote the paragraph below. A clarity judge scored it and left feedback. Revise the rewrite to raise the score, while keeping the writer's original stance and the strict word-count range.

STRICT length: original is {orig_words} words. Rewrite MUST be in [{min_words}, {max_words}]. One paragraph, no headings/bullets.

Proposition: {proposition}

Original paragraph:
\"\"\"{paragraph}\"\"\"

Previous rewrite (scored clarity={prev_clarity}/100):
\"\"\"{prev_rewrite}\"\"\"

Judge feedback on previous rewrite:
{judge_reasoning}

Return only the improved paragraph."""


def build_rewrite_prompt(method: str, proposition: str, paragraph: str,
                          leadin_variant: int | None = None,
                          icir_context: dict | None = None) -> str:
    words = len(paragraph.split())
    target = max(60, words)
    hard_cap = int(round(words * 1.2)) + 10
    orig, lo, hi = length_bounds(paragraph, tolerance=0.10)
    if method == "naive":
        return NAIVE_USER_TEMPLATE.format(proposition=proposition, paragraph=paragraph,
                                          target_words=target, hard_cap=hard_cap)
    if method == "lit_informed":
        return LIT_INFORMED_USER_TEMPLATE.format(proposition=proposition, paragraph=paragraph,
                                                 target_words=target, hard_cap=hard_cap)
    if method == "naive_tight":
        return NAIVE_TIGHT_USER_TEMPLATE.format(proposition=proposition, paragraph=paragraph,
                                                 orig_words=orig, min_words=lo, max_words=hi)
    if method == "lit_informed_tight":
        return LIT_INFORMED_TIGHT_USER_TEMPLATE.format(proposition=proposition, paragraph=paragraph,
                                                        orig_words=orig, min_words=lo, max_words=hi)
    if method == "injection_leadin":
        idx = (leadin_variant or 0) % len(INJECTION_LEADINS)
        return INJECTION_LEADIN_TEMPLATE.format(proposition=proposition, paragraph=paragraph,
                                                 orig_words=orig, min_words=lo, max_words=hi,
                                                 leadin=INJECTION_LEADINS[idx])
    if method == "rules_explicit":
        return RULES_EXPLICIT_TEMPLATE.format(proposition=proposition, paragraph=paragraph,
                                               orig_words=orig, min_words=lo, max_words=hi)
    if method == "rubric_aware":
        return RUBRIC_AWARE_TEMPLATE.format(proposition=proposition, paragraph=paragraph,
                                             orig_words=orig, min_words=lo, max_words=hi)
    if method == "scaffolded_cot_distill":
        return SCAFFOLDED_TEMPLATE.format(proposition=proposition, paragraph=paragraph,
                                           orig_words=orig, min_words=lo, max_words=hi)
    if method == "icir_single":
        if icir_context is None:
            # iteration 0 is the lit_informed_tight seed
            return LIT_INFORMED_TIGHT_USER_TEMPLATE.format(
                proposition=proposition, paragraph=paragraph,
                orig_words=orig, min_words=lo, max_words=hi)
        return ICIR_ITER_TEMPLATE.format(proposition=proposition, paragraph=paragraph,
                                          orig_words=orig, min_words=lo, max_words=hi,
                                          prev_rewrite=icir_context["prev_rewrite"],
                                          prev_clarity=icir_context["prev_clarity"],
                                          judge_reasoning=icir_context["judge_reasoning"])
    if method == "bon_panel":
        # The BoN caller samples N from LIT_INFORMED_TIGHT at high temperature.
        return LIT_INFORMED_TIGHT_USER_TEMPLATE.format(
            proposition=proposition, paragraph=paragraph,
            orig_words=orig, min_words=lo, max_words=hi)
    raise ValueError(f"unknown method: {method}")


def build_retry_prompt(method: str, proposition: str, paragraph: str, previous_rewrite: str) -> str:
    """Ask for a second attempt if the first draft was out of range."""
    base = build_rewrite_prompt(method, proposition, paragraph)
    _, lo, hi = length_bounds(paragraph, tolerance=0.10)
    actual = len(previous_rewrite.split())
    if actual < lo:
        direction = f"You need MORE words — expand with concrete detail, not filler."
    else:
        direction = f"You need FEWER words — cut weaker claims, redundancies, and hedges."
    return base + TIGHT_RETRY_SUFFIX.format(
        actual_words=actual, min_words=lo, max_words=hi, direction=direction
    )
