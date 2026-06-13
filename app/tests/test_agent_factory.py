from unittest.mock import Mock

import pytest

from app.agent import graph as graph_module


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
