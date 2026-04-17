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
    raise ValueError(f"unknown method: {method}")
