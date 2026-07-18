import json
from types import SimpleNamespace
from unittest.mock import Mock

from app.eval.scoring import (
    score_groundedness_heuristic,
    score_groundedness_llm_judge,
    score_relevance_heuristic,
)


def test_groundedness_heuristic_scores_high_for_overlapping_answer():
    source = ["NVIDIA Blackwell GPU platforms accelerate model training and inference."]
    answer = "NVIDIA's Blackwell GPU platforms accelerate model training for AI infrastructure."

    score = score_groundedness_heuristic(answer, source)

    assert score > 0.5


def test_groundedness_heuristic_scores_low_for_unrelated_answer():
    source = ["NVIDIA Blackwell GPU platforms accelerate model training and inference."]
    answer = "The weather in Paris is sunny with a light breeze today."

    score = score_groundedness_heuristic(answer, source)

    assert score < 0.2


def test_groundedness_heuristic_handles_empty_inputs():
    assert score_groundedness_heuristic("", ["some source text"]) == 0.0
    assert score_groundedness_heuristic("some answer text", []) == 0.0


def test_relevance_heuristic_scores_high_for_matching_reference():
    reference = "IBM watsonx supports enterprise AI governance and model lifecycle management."
    answer = "IBM's watsonx platform supports enterprise AI governance and lifecycle management."

    score = score_relevance_heuristic(answer, reference)

    assert score > 0.5


def test_relevance_heuristic_scores_low_for_unrelated_reference():
    reference = "IBM watsonx supports enterprise AI governance and model lifecycle management."
    answer = "The stock market closed higher today amid strong retail earnings."

    score = score_relevance_heuristic(answer, reference)

    assert score < 0.2


def test_groundedness_llm_judge_parses_structured_json_response(monkeypatch):
    fake_model = Mock()
    fake_model.invoke.return_value = SimpleNamespace(
        content=json.dumps({"groundedness": 0.85, "reasoning": "Answer matches the source."})
    )

    verdict = score_groundedness_llm_judge("answer text", ["source text"], fake_model)

    assert verdict.groundedness == 0.85
    assert verdict.reasoning == "Answer matches the source."


def test_groundedness_llm_judge_strips_markdown_code_fence():
    fake_model = Mock()
    fake_model.invoke.return_value = SimpleNamespace(
        content='```json\n{"groundedness": 0.4, "reasoning": "Partially supported."}\n```'
    )

    verdict = score_groundedness_llm_judge("answer text", ["source text"], fake_model)

    assert verdict.groundedness == 0.4


def test_groundedness_llm_judge_clamps_out_of_range_scores():
    fake_model = Mock()
    fake_model.invoke.return_value = SimpleNamespace(
        content=json.dumps({"groundedness": 1.5, "reasoning": "Overclaimed score."})
    )

    verdict = score_groundedness_llm_judge("answer text", ["source text"], fake_model)

    assert verdict.groundedness == 1.0
