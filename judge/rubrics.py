"""Judge rubrics for clarity and informativeness.

Design choices (drawn from LLM-as-a-judge literature):

- Rubric-based single-criterion scoring (Prometheus/Prometheus-2,
  Kim et al. 2023/2024). Each criterion gets its own prompt so the judge
  focuses on one axis and we avoid compound-score smearing.
- Anchor descriptions at 5 equally spaced score bands (0-20, 21-40, ...,
  81-100). Anchors reduce scoring drift and calibration noise that appear
  with bare numeric scales (Zheng et al. 2023; Liu et al. G-Eval 2023).
- Structured output (JSON: reasoning then score). Forcing the reasoning
  before the score is a chain-of-thought evaluation step — G-Eval
  (arXiv:2303.16634) and Prometheus both find this raises human
  correlation. Placing the score last also mitigates rationalisation
  anchoring, where the model justifies a pre-committed number.
- Single-pass absolute scoring (not pairwise) because the downstream
  application needs an absolute 0-100 score per document (per plan.md),
  and pairwise scoring would multiply cost and add position bias
  (Shi et al. 2024).
- Criteria are deliberately decoupled from stance/agreement with the
  proposition: the agreement_model covers that separately; judging
  clarity and informativeness should not reward one side of the argument
  over the other. The prompt says so explicitly.
- Length-bias guard: verbosity is a known LLM-judge failure mode
  (Singhal et al. 2024). The rubrics flag it so the judge doesn't simply
  reward long paragraphs.
"""

SYSTEM_PROMPT = (
    "You are a careful expert evaluator of short argumentative paragraphs. "
    "You follow the rubric exactly, and you return your answer as valid JSON."
)


CLARITY_RUBRIC = """You are judging the CLARITY of a short argumentative paragraph written in response to a political/policy proposition.

CLARITY is about *how well the paragraph is written*, not whether you agree with it, not how persuasive it is, and not how much information it contains. Focus strictly on:
- Grammar, spelling, and punctuation
- Sentence structure and readability
- Logical flow between sentences
- Whether the main point is expressed in a way a general reader can understand on first read

Do NOT reward or penalise based on:
- Stance toward the proposition (pro or con)
- Amount or quality of evidence (that is a separate criterion)
- Length (a short, clear paragraph can score as high as a long one)
- Formal vs. informal register, as long as the meaning is clear

Score on an integer scale from 0 to 100, using these anchors:

- 0-20 (Very unclear): Pervasive grammar/spelling errors, broken sentences, or jumbled ideas that obstruct understanding. A reader would struggle to extract the point.
- 21-40 (Unclear): Frequent errors, awkward phrasing, or disorganised reasoning. The main point is recoverable only with effort.
- 41-60 (Moderately clear): Noticeable errors or some confusing sentences, but the main idea comes through on a normal read.
- 61-80 (Clear): Well-written with at most minor errors or mild awkwardness. Easy to follow.
- 81-100 (Very clear): Polished, essentially error-free prose with well-organised, fluent sentences. Smooth first-read comprehension.

Proposition: {proposition}

Paragraph:
\"\"\"{paragraph}\"\"\"

Think step by step about the paragraph's clarity, citing specific features (grammar, structure, flow) as needed. Then output a single integer score from 0 to 100.

Return ONLY a JSON object of the form:
{{"reasoning": "<1-3 sentences>", "score": <integer 0-100>}}"""


INFORMATIVENESS_RUBRIC = """You are judging how INFORMATIVE a short argumentative paragraph is with respect to a political/policy proposition.

INFORMATIVENESS is about *substantive content that helps a reader think about the proposition*. Focus on:
- Whether the paragraph offers specific reasons, evidence, examples, mechanisms, or data
- Whether the reasons are concrete and substantive rather than vague assertions
- Whether the content is genuinely relevant to the proposition (not just topic-adjacent)
- Whether a reader would come away better equipped to reason about the proposition

Do NOT reward or penalise based on:
- Stance toward the proposition (pro or con are equally welcome)
- Whether you personally agree with the claims
- Writing style, grammar, or polish (that is a separate criterion)
- Length per se — a compact paragraph packed with specifics can score as high as a long one. Do not reward padding, lists of empty generalities, or repetition.

Score on an integer scale from 0 to 100, using these anchors:

- 0-20 (Not informative): Restates the proposition, asserts a feeling, or makes only vague claims. No reasons, evidence, or examples.
- 21-40 (Minimally informative): One or two generic reasons with no specifics; claims are asserted but not supported.
- 41-60 (Moderately informative): Some concrete reasoning or examples, but shallow coverage or gaps between claims and support.
- 61-80 (Informative): Multiple specific reasons, examples, or mechanisms that meaningfully bear on the proposition. A thoughtful reader would learn something.
- 81-100 (Highly informative): Rich in specifics (facts, data, concrete examples, mechanisms, trade-offs, counterpoints) tightly linked to the proposition. Substantially advances a reader's thinking.

Proposition: {proposition}

Paragraph:
\"\"\"{paragraph}\"\"\"

Think step by step about what specific information the paragraph provides, how concretely it supports its claims, and how directly it bears on the proposition. Then output a single integer score from 0 to 100.

Return ONLY a JSON object of the form:
{{"reasoning": "<1-3 sentences>", "score": <integer 0-100>}}"""


def build_prompt(criterion: str, proposition: str, paragraph: str) -> str:
    template = {"clarity": CLARITY_RUBRIC, "informativeness": INFORMATIVENESS_RUBRIC}[criterion]
    return template.format(proposition=proposition, paragraph=paragraph)
