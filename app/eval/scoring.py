"""Groundedness/relevance scoring for the evaluation harness.

Heuristic scoring is the default, always-on path: zero-cost, reproducible,
and safe to run in CI without network access. LLM-as-judge is an explicit
opt-in companion for richer signal, since it costs money and network access
per call.
"""

import json
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "will", "with",
}


def _significant_words(text: str) -> set[str]:
    words = _WORD_PATTERN.findall(text.lower())
    return {word for word in words if word not in _STOPWORDS and len(word) > 2}


def score_groundedness_heuristic(answer: str, source_documents: list[str]) -> float:
    """Fraction of the answer's significant vocabulary also present in the sources.

    Directional signal only (an LLM could paraphrase and still be grounded,
    or reuse source words while contradicting them) -- not a substitute for
    the LLM-judge path below.
    """

    answer_words = _significant_words(answer)
    if not answer_words:
        return 0.0

    source_words: set[str] = set()
    for document in source_documents:
        source_words |= _significant_words(document)

    if not source_words:
        return 0.0

    return len(answer_words & source_words) / len(answer_words)


def score_relevance_heuristic(answer: str, reference_answer: str) -> float:
    """Jaccard similarity between the answer's and reference's vocabulary."""

    answer_words = _significant_words(answer)
    reference_words = _significant_words(reference_answer)

    if not answer_words and not reference_words:
        return 1.0
    if not answer_words or not reference_words:
        return 0.0

    intersection = len(answer_words & reference_words)
    union = len(answer_words | reference_words)
    return intersection / union if union else 0.0


@dataclass(frozen=True)
class JudgeVerdict:
    groundedness: float
    reasoning: str


_JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge. Score how well the ANSWER is grounded in "
    "the SOURCES on a scale from 0.0 (unsupported/fabricated) to 1.0 "
    "(fully supported). Return only JSON: "
    '{"groundedness": <float 0-1>, "reasoning": "<one sentence>"}'
)


def score_groundedness_llm_judge(
    answer: str,
    source_documents: list[str],
    model: BaseChatModel,
) -> JudgeVerdict:
    """LLM-as-judge groundedness score. Costs money/network -- opt-in only."""

    sources_block = "\n\n".join(source_documents)
    prompt = f"SOURCES:\n{sources_block}\n\nANSWER:\n{answer}"

    response = model.invoke(
        [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw.strip())

    return JudgeVerdict(
        groundedness=max(0.0, min(1.0, float(result["groundedness"]))),
        reasoning=str(result.get("reasoning", "")),
    )
