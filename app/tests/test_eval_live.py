import os

import pytest

from app.config import OPENAI_API_KEY, TAVILY_API_KEY
from app.eval.dataset import GOLDEN_EVAL_CASES
from app.eval.scoring import score_groundedness_llm_judge
from app.eval.trajectory import classify_and_score, run_agent_and_score
from app.providers import build_chat_model

pytestmark = [pytest.mark.integration, pytest.mark.live, pytest.mark.slow, pytest.mark.eval]


def _require_live_environment() -> None:
    if os.getenv("RUN_LIVE_AGENT_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set RUN_LIVE_AGENT_TESTS=true to run live eval tests")
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY is required for live eval tests")


def test_routing_trajectory_across_golden_dataset_meets_minimum_accuracy():
    _require_live_environment()

    scores = [classify_and_score(case) for case in GOLDEN_EVAL_CASES]

    rag_accuracy = sum(1 for score in scores if score.rag_correct) / len(scores)
    avg_recall = sum(score.tool_recall for score in scores) / len(scores)

    assert rag_accuracy >= 0.7
    assert avg_recall >= 0.6


def test_full_agent_trajectory_for_private_rag_case():
    _require_live_environment()
    if not TAVILY_API_KEY:
        pytest.skip("TAVILY_API_KEY is required for full-agent live eval tests")

    case = next(case for case in GOLDEN_EVAL_CASES if case.category == "private_rag")

    score = run_agent_and_score(case)

    assert score.rag_correct is True


def test_llm_judge_groundedness_on_reference_answer():
    _require_live_environment()

    case = next(case for case in GOLDEN_EVAL_CASES if case.reference_answer)
    judge_model = build_chat_model(provider="openai", model="gpt-4o-mini", temperature=0.0)

    verdict = score_groundedness_llm_judge(
        case.reference_answer, [case.reference_answer], judge_model
    )

    assert verdict.groundedness >= 0.7
