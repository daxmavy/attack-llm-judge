"""Shared rubric text + parsing utilities for judge scoring.

Imported by both sides of the judge HTTP split:
- `judge/server.py` — the vLLM-backed HTTP server that actually runs generate.
- `judge/http_client.py` / consumer scripts — needs RUBRICS keys for validation
  plus `parse_score_and_reasoning` for ICIR-style feedback extraction.

Keeping the rubric text in one module guarantees the server and any out-of-band
consumer (e.g. ICIR parsing a batch of raw judge responses) stay in sync.

Historical note: these were inlined in `training/scripts/run_pilot_len_pen.py`
and re-imported from there by consumer scripts. That made it impossible to
spin up the judge server without loading the whole GRPO trainer module first,
and it entangled the rubric text with training hyperparameters. Moving them
here is the prerequisite for the out-of-process judge refactor.
"""
from __future__ import annotations

import json
import re


JUDGE_SYSTEM = (
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


INFORMATIVENESS_RUBRIC = """You are judging the INFORMATIVENESS of a short argumentative paragraph written in response to a political/policy proposition.

INFORMATIVENESS is about *how much substantive information the paragraph communicates* to support or oppose the proposition: relevant facts, evidence, mechanisms, specific examples, and concrete reasoning. Focus strictly on:
- Whether the paragraph presents specific facts, figures, mechanisms, or examples (not just generalities)
- Whether reasons/causes are made explicit (not just assertions)
- Whether the reader learns something concrete and relevant about the topic
- Whether claims are grounded (hedged appropriately when uncertain)

Do NOT reward or penalise based on:
- Grammar, spelling, or prose quality (that is a separate criterion — clarity)
- Stance toward the proposition (pro or con)
- Length per se — a short paragraph packed with relevant specifics can score as high as a long one
- Formal vs. informal register

Score on an integer scale from 0 to 100, using these anchors:

- 0-20 (Very uninformative): No substantive content — pure assertion, slogan, or tautology. Reader learns nothing new or relevant.
- 21-40 (Uninformative): Only vague general claims with no specifics. Few or no concrete reasons, mechanisms, or examples.
- 41-60 (Moderately informative): Some substance — includes at least one concrete reason or example, but much of the paragraph is still abstract or generic.
- 61-80 (Informative): Multiple concrete claims, reasons, or examples; mechanisms made explicit; reader learns something real about the topic.
- 81-100 (Very informative): Dense with specific facts, figures, mechanisms, and examples; reasoning is explicit and well-grounded; claims are appropriately hedged; a knowledgeable reader would learn something new.

Proposition: {proposition}

Paragraph:
\"\"\"{paragraph}\"\"\"

Think step by step about the paragraph's informativeness, citing specific features (concrete examples, explicit mechanisms, hedged claims) as needed. Then output a single integer score from 0 to 100.

Return ONLY a JSON object of the form:
{{"reasoning": "<1-3 sentences>", "score": <integer 0-100>}}"""


RUBRICS: dict[str, str] = {
    "clarity": CLARITY_RUBRIC,
    "informativeness": INFORMATIVENESS_RUBRIC,
}


# Two parse pipelines — the loose one (JSON-ish object anywhere in the reply)
# used in the original training reward, and the tight one (only braces with
# "score" inside) used by ICIR's reasoning-extraction path.
_SCORE_RE_JSON = re.compile(r"\{[\s\S]*?\}")
_SCORE_RE_JSON_TIGHT = re.compile(
    r"\{[^{}]*\"score\"\s*:\s*\d+(?:\.\d+)?[^{}]*\}", re.S
)
_SCORE_RE_KV = re.compile(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)')
_REASON_RE = re.compile(r"\"reasoning\"\s*:\s*\"((?:[^\"\\]|\\.)*)\"", re.S)


def parse_score(text: str | None) -> float | None:
    """Best-effort extraction of numeric `score` from a judge response.

    Returns None if no score can be recovered. Callers typically fall back
    to 50.0 on None (legacy training-reward behaviour).
    """
    if not text:
        return None
    m = _SCORE_RE_JSON.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "score" in obj:
                return float(obj["score"])
        except Exception:
            pass
    m = _SCORE_RE_KV.search(text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return None


def parse_score_and_reasoning(text: str | None) -> tuple[float | None, str]:
    """Return (score, reasoning) — used by ICIR iteration feedback."""
    if not text:
        return None, ""
    m = _SCORE_RE_JSON_TIGHT.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            return float(obj["score"]), str(obj.get("reasoning", ""))[:600]
        except Exception:
            pass
    score = parse_score(text)
    reasoning = ""
    m = _REASON_RE.search(text)
    if m:
        try:
            reasoning = m.group(1).encode().decode("unicode_escape", errors="replace")[:600]
        except Exception:
            reasoning = m.group(1)[:600]
    return score, reasoning


def supports_system_role(tok) -> bool:
    """Some tokenizer chat templates reject the `system` role (e.g. older
    Gemma-2). Probe with a dummy chat and fall back to concatenated user
    content if it raises.
    """
    try:
        tok.apply_chat_template(
            [{"role": "system", "content": "x"},
             {"role": "user", "content": "y"}],
            tokenize=False, add_generation_prompt=True,
        )
        return True
    except Exception:
        return False
