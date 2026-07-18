"""Evaluation harness CLI.

Usage:
    python -m app.eval.run
    python -m app.eval.run --full-agent
    python -m app.eval.run --full-agent --judge llm --report json

Always runs the routing-classification trajectory check for every golden
case (one cheap LLM call per case). --full-agent additionally runs each
case through the real agent end-to-end (more calls, more cost) and scores
groundedness/relevance for cases with a reference answer.

This is a live evaluation run against real providers -- it requires
OPENAI_API_KEY (and TAVILY_API_KEY for news-touching cases) and will incur
API cost. It is intentionally separate from the pytest suite so it can
later be invoked directly from a CI job without redesign.
"""

import argparse
import json
import sys
from dataclasses import asdict

from langchain_core.messages import HumanMessage, ToolMessage

from app.agent.graph import create_financial_agent, tools_from_names
from app.eval.dataset import GOLDEN_EVAL_CASES, EVAL_DATASET_VERSION, EvalCase
from app.eval.scoring import (
    score_groundedness_heuristic,
    score_groundedness_llm_judge,
    score_relevance_heuristic,
)
from app.eval.trajectory import score_tool_trajectory
from app.providers import build_chat_model
from app.routing.policy import RoutingDecision, classify_query


def _run_agent(case: EvalCase, decision: RoutingDecision) -> tuple[str, set[str]]:
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
    answer = str(result["messages"][-1].content)
    tool_names = {
        message.name
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.name
    }
    return answer, tool_names


def _evaluate_case(case: EvalCase, *, run_full_agent: bool, judge: str) -> dict:
    row: dict = {"id": case.id, "category": case.category}

    decision = classify_query(case.query, with_rag_requested=case.with_rag_requested)
    row["model_tier"] = decision.model_tier.value
    row["reasoning"] = decision.matched_rules

    general_tools = tuple(name for name in decision.tool_names if name != "query_private_database")
    routing_score = score_tool_trajectory(case, general_tools, decision.rag_engaged)
    row["routing"] = asdict(routing_score)

    if run_full_agent:
        answer, actual_tools = _run_agent(case, decision)
        rag_engaged = "query_private_database" in actual_tools
        general_actual = tuple(name for name in actual_tools if name != "query_private_database")
        agent_score = score_tool_trajectory(case, general_actual, rag_engaged)
        row["agent"] = asdict(agent_score)

        if case.reference_answer:
            row["relevance_heuristic"] = score_relevance_heuristic(answer, case.reference_answer)
            row["groundedness_heuristic"] = score_groundedness_heuristic(
                answer, [case.reference_answer]
            )
            if judge == "llm":
                judge_model = build_chat_model(provider="openai", model="gpt-4o-mini", temperature=0.0)
                verdict = score_groundedness_llm_judge(answer, [case.reference_answer], judge_model)
                row["groundedness_llm_judge"] = verdict.groundedness
                row["groundedness_llm_reasoning"] = verdict.reasoning

    return row


def _print_text_report(rows: list[dict]) -> None:
    print(f"Eval dataset version: {EVAL_DATASET_VERSION}  ({len(rows)} cases)\n")
    for row in rows:
        routing = row["routing"]
        print(
            f"[{row['id']:<28}] tier={row['model_tier']:<8} "
            f"routing(P/R/F1)={routing['tool_precision']:.2f}/"
            f"{routing['tool_recall']:.2f}/{routing['tool_f1']:.2f} "
            f"rag_correct={routing['rag_correct']}"
        )
        if "agent" in row:
            agent = row["agent"]
            print(
                f"    agent(P/R/F1)={agent['tool_precision']:.2f}/"
                f"{agent['tool_recall']:.2f}/{agent['tool_f1']:.2f} "
                f"rag_correct={agent['rag_correct']}"
            )
        if "groundedness_heuristic" in row:
            print(
                f"    groundedness_heuristic={row['groundedness_heuristic']:.2f} "
                f"relevance_heuristic={row['relevance_heuristic']:.2f}"
                + (
                    f" groundedness_llm_judge={row['groundedness_llm_judge']:.2f}"
                    if "groundedness_llm_judge" in row
                    else ""
                )
            )

    avg_precision = sum(r["routing"]["tool_precision"] for r in rows) / len(rows)
    avg_recall = sum(r["routing"]["tool_recall"] for r in rows) / len(rows)
    rag_accuracy = sum(1 for r in rows if r["routing"]["rag_correct"]) / len(rows)
    print(
        f"\nSummary: avg routing precision={avg_precision:.2f} "
        f"avg routing recall={avg_recall:.2f} rag_accuracy={rag_accuracy:.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Argent evaluation harness")
    parser.add_argument("--full-agent", action="store_true", help="Also run each case through the real agent")
    parser.add_argument("--judge", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--report", choices=["text", "json"], default="text")
    args = parser.parse_args()

    rows = [
        _evaluate_case(case, run_full_agent=args.full_agent, judge=args.judge)
        for case in GOLDEN_EVAL_CASES
    ]

    if args.report == "json":
        print(json.dumps(rows, indent=2, default=str))
    else:
        _print_text_report(rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
