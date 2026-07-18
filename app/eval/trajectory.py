"""Tool-selection/trajectory accuracy scoring for the evaluation harness.

Scoring math (this module) is pure and offline-testable -- it takes
already-computed tool names in, it never calls a model itself. Two live
call sites use it:
  - classify_and_score(): calls the real routing classifier (network).
  - run_agent_and_score(): runs the real agent end-to-end (network).
Both live call sites are opt-in and used by app/eval/run.py and the
`live`-marked eval tests, not by the default offline suite.
"""

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, ToolMessage

from app.agent.graph import create_financial_agent, tools_from_names
from app.eval.dataset import EvalCase
from app.routing.policy import RoutingDecision, classify_query


@dataclass(frozen=True)
class TrajectoryScore:
    case_id: str
    tool_precision: float
    tool_recall: float
    tool_f1: float
    rag_correct: bool


def _precision_recall_f1(expected: set[str], actual: set[str]) -> tuple[float, float, float]:
    true_positives = len(expected & actual)
    precision = true_positives / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = true_positives / len(expected) if expected else (1.0 if not actual else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def score_tool_trajectory(
    case: EvalCase,
    actual_tool_names: tuple[str, ...],
    rag_engaged: bool,
) -> TrajectoryScore:
    """Pure scoring: compare actual tool usage against an EvalCase's expectations."""

    precision, recall, f1 = _precision_recall_f1(
        set(case.expected_tools), set(actual_tool_names)
    )
    return TrajectoryScore(
        case_id=case.id,
        tool_precision=precision,
        tool_recall=recall,
        tool_f1=f1,
        rag_correct=rag_engaged == case.expected_rag_engaged,
    )


def score_routing_decision(case: EvalCase, decision: RoutingDecision) -> TrajectoryScore:
    general_tools = tuple(name for name in decision.tool_names if name != "query_private_database")
    return score_tool_trajectory(case, general_tools, decision.rag_engaged)


def classify_and_score(case: EvalCase) -> TrajectoryScore:
    """Live: calls the real routing classifier and scores its decision."""

    decision = classify_query(case.query, with_rag_requested=case.with_rag_requested)
    return score_routing_decision(case, decision)


def run_agent_and_score(case: EvalCase) -> TrajectoryScore:
    """Live: runs the real agent end-to-end and scores its actual tool calls."""

    decision = classify_query(case.query, with_rag_requested=case.with_rag_requested)
    agent = create_financial_agent(
        agent_type="full",
        with_memory=True,
        tools=tools_from_names(decision.tool_names),
        model_tier=decision.model_tier.value,
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content=case.query)]},
        config={"configurable": {"thread_id": f"eval-{case.id}"}},
    )
    actual_tool_names = {
        message.name
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.name
    }
    rag_engaged = "query_private_database" in actual_tool_names
    general_tools = tuple(name for name in actual_tool_names if name != "query_private_database")
    return score_tool_trajectory(case, general_tools, rag_engaged)
