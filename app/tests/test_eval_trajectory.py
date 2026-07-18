import pytest

from app.eval.dataset import GOLDEN_EVAL_CASES, EvalCase
from app.eval.trajectory import score_routing_decision, score_tool_trajectory
from app.routing.policy import ModelTier, RoutingDecision

pytestmark = pytest.mark.eval


def test_perfect_match_scores_full_precision_recall_and_correct_rag():
    case = EvalCase(
        id="test-case",
        query="irrelevant for pure scoring",
        expected_tools=("get_stock_price", "get_stock_history"),
        expected_rag_engaged=True,
    )

    score = score_tool_trajectory(
        case,
        actual_tool_names=("get_stock_price", "get_stock_history"),
        rag_engaged=True,
    )

    assert score.tool_precision == 1.0
    assert score.tool_recall == 1.0
    assert score.tool_f1 == 1.0
    assert score.rag_correct is True


def test_missing_expected_tool_lowers_recall_not_precision():
    case = EvalCase(
        id="test-case",
        query="irrelevant for pure scoring",
        expected_tools=("get_stock_price", "get_stock_history"),
        expected_rag_engaged=False,
    )

    score = score_tool_trajectory(case, actual_tool_names=("get_stock_price",), rag_engaged=False)

    assert score.tool_precision == 1.0
    assert score.tool_recall == 0.5


def test_extra_unexpected_tool_lowers_precision_not_recall():
    case = EvalCase(
        id="test-case",
        query="irrelevant for pure scoring",
        expected_tools=("get_stock_price",),
        expected_rag_engaged=False,
    )

    score = score_tool_trajectory(
        case,
        actual_tool_names=("get_stock_price", "search_financial_news"),
        rag_engaged=False,
    )

    assert score.tool_precision == 0.5
    assert score.tool_recall == 1.0


def test_wrong_rag_engagement_is_flagged_incorrect():
    case = EvalCase(
        id="test-case",
        query="irrelevant for pure scoring",
        expected_tools=(),
        expected_rag_engaged=True,
    )

    score = score_tool_trajectory(case, actual_tool_names=(), rag_engaged=False)

    assert score.rag_correct is False


def test_score_routing_decision_excludes_rag_tool_from_general_tool_scoring():
    case = EvalCase(
        id="rag-case",
        query="irrelevant for pure scoring",
        expected_tools=("get_stock_price",),
        expected_rag_engaged=True,
    )
    decision = RoutingDecision(
        model_tier=ModelTier.CAPABLE,
        provider="openai",
        model_name="gpt-4o",
        tool_names=("get_stock_price", "query_private_database"),
        rag_engaged=True,
        matched_rules=("llm_classifier:test",),
    )

    score = score_routing_decision(case, decision)

    assert score.tool_precision == 1.0
    assert score.tool_recall == 1.0
    assert score.rag_correct is True


def test_golden_dataset_cases_are_internally_scoreable():
    # Sanity check that every case can be scored against itself perfectly --
    # catches malformed dataset entries without needing a live classifier.
    for case in GOLDEN_EVAL_CASES:
        score = score_tool_trajectory(
            case,
            actual_tool_names=case.expected_tools,
            rag_engaged=case.expected_rag_engaged,
        )
        assert score.tool_precision == 1.0
        assert score.tool_recall == 1.0
        assert score.rag_correct is True
