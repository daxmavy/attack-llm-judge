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


def build_rewrite_prompt(method: str, proposition: str, paragraph: str) -> str:
    words = len(paragraph.split())
    target = max(60, words)
    hard_cap = int(round(words * 1.2)) + 10
    if method == "naive":
        return NAIVE_USER_TEMPLATE.format(proposition=proposition, paragraph=paragraph,
                                          target_words=target, hard_cap=hard_cap)
    if method == "lit_informed":
        return LIT_INFORMED_USER_TEMPLATE.format(proposition=proposition, paragraph=paragraph,
                                                 target_words=target, hard_cap=hard_cap)
    if method == "naive_tight":
        orig, lo, hi = length_bounds(paragraph, tolerance=0.10)
        return NAIVE_TIGHT_USER_TEMPLATE.format(proposition=proposition, paragraph=paragraph,
                                                 orig_words=orig, min_words=lo, max_words=hi)
    if method == "lit_informed_tight":
        orig, lo, hi = length_bounds(paragraph, tolerance=0.10)
        return LIT_INFORMED_TIGHT_USER_TEMPLATE.format(proposition=proposition, paragraph=paragraph,
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
