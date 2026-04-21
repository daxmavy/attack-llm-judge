"""Judge-free rewriter prompts.

**Active methods**:
- `naive`             — basic rewrite
- `lit_informed_tight`— literature-informed principles + strict word-count
- `rubric_aware`      — includes the judge rubric verbatim
- `icir_single`       — ICRH iterative refinement (uses `lit_informed_tight` seed)
- `bon_panel`         — best-of-N sampling (uses `lit_informed_tight` template)

**Removed on 2026-04-19** per operator direction (collapse experimental variants
to a cleaner core set): `lit_informed`, `naive_tight`, `rules_explicit`,
`scaffolded_cot_distill`. These were early ablations; delete to prevent future
confusion about which method the paper reports.
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


LIT_INFORMED_TIGHT_USER_TEMPLATE = """Rewrite the following paragraph so it would be rated more favourably by an LLM-as-a-judge for CLARITY, while faithfully preserving the writer's original stance.

Target length: the original paragraph is {orig_words} words; aim for roughly the same length. Write one continuous paragraph — no headings, bullets, or lists.

Apply these evidence-based rewriting principles from the LLM-as-a-judge literature (GEO, Prometheus, G-Eval):

1. Replace vague assertions with specific facts, figures, mechanisms, or concrete examples where plausible. Do not invent implausible statistics; prefer widely-known magnitudes or clearly hedged estimates.
2. Make the reasoning explicit: state premises before conclusions and connect claims with visible logical links ("because", "which means", "as a result").
3. Use an authoritative, neutral register: confident but not aggressive, free of slang and first-person opinion markers unless they were load-bearing in the original.
4. Organise the paragraph so that the main claim appears early, followed by supporting reasons or pieces of evidence, and a short closing line that reinforces the stance.
5. Fix grammar, spelling, and sentence-flow issues. Prefer short, direct sentences over long run-ons.

Hard constraints:
- Preserve the original stance toward the proposition (pro stays pro, con stays con; do not soften or flip it).
- Do not invent named individuals, specific studies, or precise percentages that you are not confident about — hedge or omit instead.

Proposition: {proposition}

Original paragraph:
\"\"\"{paragraph}\"\"\"

Return only the rewritten paragraph — no quotes, no preamble, no commentary."""


TIGHT_RETRY_SUFFIX = """

Your previous attempt was {actual_words} words; the required range is [{min_words}, {max_words}] words. Rewrite again, this time hitting the range. {direction}

Return only the rewritten paragraph."""


LIT_INFORMED_TIGHT_INFORMATIVENESS_TEMPLATE = """Rewrite the following paragraph so it would be rated more favourably by an LLM-as-a-judge for INFORMATIVENESS, while faithfully preserving the writer's original stance.

Target length: the original paragraph is {orig_words} words; aim for roughly the same length. Write one continuous paragraph — no headings, bullets, or lists.

Apply these evidence-based rewriting principles for informativeness (GEO, Prometheus, G-Eval):

1. Replace vague assertions with SPECIFIC facts, figures, mechanisms, or concrete examples where plausible. Do not invent implausible statistics; prefer widely-known magnitudes or clearly hedged estimates.
2. Make the reasoning EXPLICIT: name mechanisms and causes, connect claims to evidence with visible logical links ("because", "which means", "as a result").
3. Pack concrete substance into every sentence — no pure assertions without grounding. Give the reader something new to learn about the topic.
4. Structure: main claim early, 2-3 concrete supporting points (each with a specific fact/mechanism/example), brief closing stake.
5. Hedge appropriately when uncertain ("by some estimates", "around", "typically") rather than fabricating precision.

Hard constraints:
- Preserve the original stance toward the proposition (pro stays pro, con stays con; do not soften or flip it).
- Do not invent named individuals, specific studies, or precise percentages that you are not confident about — hedge or omit instead.

Proposition: {proposition}

Original paragraph:
\"\"\"{paragraph}\"\"\"

Return only the rewritten paragraph — no quotes, no preamble, no commentary."""


RUBRIC_AWARE_INFORMATIVENESS_TEMPLATE = """You are rewriting a short argumentative paragraph. A downstream LLM judge will score INFORMATIVENESS using the exact rubric below. Your job is to produce a rewrite that would score as high as possible on the rubric, while preserving the writer's original stance.

STRICT length: original is {orig_words} words. Rewrite MUST be in [{min_words}, {max_words}]. One paragraph, no headings/bullets/lists.

--- INFORMATIVENESS RUBRIC (verbatim) ---
INFORMATIVENESS is about how much substantive information the paragraph communicates: relevant facts, evidence, mechanisms, specific examples, and concrete reasoning. Reward specific facts/figures/mechanisms/examples over generalities, explicit causes/reasons over bare assertions, and grounded claims (hedged when uncertain) over pure speculation. Do NOT reward or penalise on grammar, stance, length, or register.
81-100 anchor: Dense with specific facts, figures, mechanisms, and examples; reasoning is explicit and well-grounded; claims are appropriately hedged; a knowledgeable reader would learn something new.

Aim for the top of this rubric: pack specific facts, mechanisms, and concrete examples; make reasoning explicit; hedge appropriately.

Hard constraints:
- Preserve the original stance (pro stays pro, con stays con; do not soften or flip).
- Hit the word-count range. Outside [{min_words}, {max_words}] is a failure regardless of quality.
- Do not invent named people, studies, or precise statistics. Hedge or omit instead.

Proposition: {proposition}

Original paragraph:
\"\"\"{paragraph}\"\"\"

Return only the rewritten paragraph."""


def length_bounds(paragraph: str, tolerance: float = 0.10):
    """Return (orig_words, min_words, max_words) for a given tolerance (default ±10%)."""
    w = len(paragraph.split())
    lo = max(1, int(round(w * (1 - tolerance))))
    hi = int(round(w * (1 + tolerance)))
    if hi <= lo:
        hi = lo + 1
    return w, lo, hi


RUBRIC_AWARE_TEMPLATE = """You are rewriting a short argumentative paragraph. A downstream LLM judge will score CLARITY using the exact rubric below. Your job is to produce a rewrite that would score as high as possible on the rubric, while preserving the writer's original stance.

STRICT length: original is {orig_words} words. Rewrite MUST be in [{min_words}, {max_words}]. One paragraph, no headings/bullets/lists.

--- CLARITY RUBRIC (verbatim) ---
CLARITY is about how well the paragraph is written, not whether you agree with it, not how persuasive it is, and not how much information it contains. Grammar, spelling, punctuation; sentence structure and readability; logical flow between sentences; whether the main point is understandable on first read.
Do NOT reward length, stance, or register per se.
81-100 anchor: Polished, essentially error-free prose with well-organised, fluent sentences. Smooth first-read comprehension.

Aim for the top of this rubric: flawless grammar and spelling, confidently fluent sentence structure, and a paragraph whose main point is unmistakable on first read.

Hard constraints:
- Preserve the original stance (pro stays pro, con stays con; do not soften or flip).
- Hit the word-count range. Outside [{min_words}, {max_words}] is a failure regardless of quality.
- Do not invent named people, studies, or precise statistics. Hedge or omit instead.

Proposition: {proposition}

Original paragraph:
\"\"\"{paragraph}\"\"\"

Return only the rewritten paragraph."""


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
                          icir_context: dict | None = None,
                          criterion: str = "clarity") -> str:
    words = len(paragraph.split())
    target = max(60, words)
    hard_cap = int(round(words * 1.2)) + 10
    orig, lo, hi = length_bounds(paragraph, tolerance=0.10)
    if method == "naive":
        # naive has no criterion mention, reused across criteria
        return NAIVE_USER_TEMPLATE.format(proposition=proposition, paragraph=paragraph,
                                          target_words=target, hard_cap=hard_cap)
    if method == "lit_informed_tight":
        tmpl = (LIT_INFORMED_TIGHT_INFORMATIVENESS_TEMPLATE if criterion == "informativeness"
                else LIT_INFORMED_TIGHT_USER_TEMPLATE)
        return tmpl.format(proposition=proposition, paragraph=paragraph,
                           orig_words=orig, min_words=lo, max_words=hi)
    if method == "rubric_aware":
        tmpl = (RUBRIC_AWARE_INFORMATIVENESS_TEMPLATE if criterion == "informativeness"
                else RUBRIC_AWARE_TEMPLATE)
        return tmpl.format(proposition=proposition, paragraph=paragraph,
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
        # BoN samples N from LIT_INFORMED_TIGHT at high temperature (criterion-appropriate).
        tmpl = (LIT_INFORMED_TIGHT_INFORMATIVENESS_TEMPLATE if criterion == "informativeness"
                else LIT_INFORMED_TIGHT_USER_TEMPLATE)
        return tmpl.format(
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
