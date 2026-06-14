from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException, tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent import graph as graph_module
from app.agent.state import SimpleAgentState


def _tool_names(tools: list) -> list[str]:
    return [tool.name for tool in tools]


def test_default_tools_include_private_database_when_rag_enabled():
    tools = graph_module._default_tools(with_rag=True)

    assert "query_private_database" in _tool_names(tools)


def test_default_tools_exclude_private_database_when_rag_disabled():
    tools = graph_module._default_tools(with_rag=False)

    assert "query_private_database" not in _tool_names(tools)
    assert len(tools) == 4


@pytest.mark.parametrize("with_rag", [True, False])
def test_enhanced_agent_delegates_to_full_agent(monkeypatch, with_rag: bool):
    create_agent = Mock(return_value="compiled-graph")
    monkeypatch.setattr(graph_module, "create_financial_agent", create_agent)

    result = graph_module.create_enhanced_financial_agent(
        with_rag=with_rag,
        with_memory=False,
    )

    assert result == "compiled-graph"
    create_agent.assert_called_once_with(
        agent_type="full",
        with_memory=False,
        with_rag=with_rag,
    )


def test_financial_agent_rejects_empty_explicit_tool_list():
    with pytest.raises(ValueError, match="At least one tool"):
        graph_module.create_financial_agent(tools=[])


def test_tool_error_handler_preserves_safe_tool_exception():
    message = graph_module._handle_tool_error(
        ToolException("Private analyst reports are unavailable")
    )

    assert message == "Private analyst reports are unavailable"


def test_tool_error_handler_hides_unexpected_error_details():
    message = graph_module._handle_tool_error(
        RuntimeError("credential=/secret/path")
    )

    assert message == (
        "The tool is temporarily unavailable. Continue with available data."
    )
    assert "secret" not in message


def test_tool_node_converts_tool_exception_to_error_message():
    @tool
    def unavailable_private_database(query: str) -> str:
        """Query a private database that is unavailable in this test."""
        raise ToolException("Private analyst reports are unavailable")

    node = ToolNode(
        [unavailable_private_database],
        handle_tool_errors=graph_module._handle_tool_error,
    )
    workflow = StateGraph(SimpleAgentState)
    workflow.add_node("tools", node)
    workflow.set_entry_point("tools")
    workflow.add_edge("tools", END)
    graph = workflow.compile()

    result = graph.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "unavailable_private_database",
                            "args": {"query": "NVIDIA AI"},
                            "id": "rag-call",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }
    )

    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert message.content == "Private analyst reports are unavailable"
