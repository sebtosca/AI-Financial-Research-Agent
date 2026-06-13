import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langchain_core.tools import ToolException

tool_module = importlib.import_module("app.tools.query_private_database")


def test_query_private_database_rejects_empty_query():
    with pytest.raises(ToolException, match="cannot be empty"):
        tool_module.query_private_database.invoke({"query": "   "})


def test_query_private_database_returns_no_results_message(monkeypatch):
    retriever = Mock()
    retriever.invoke.return_value = []
    model = Mock()

    monkeypatch.setattr(tool_module, "_get_retriever", lambda: retriever)
    monkeypatch.setattr(tool_module, "_get_model", lambda: model)

    result = tool_module.query_private_database.invoke(
        {"query": "What is the private AI roadmap?"}
    )

    assert result == tool_module.PRIVATE_DATABASE_NO_RESULTS_MESSAGE
    model.invoke.assert_not_called()


def test_query_private_database_uses_retrieved_context(monkeypatch):
    retriever = Mock()
    retriever.invoke.return_value = [
        Document(
            page_content="Microsoft is expanding its Copilot program.",
            metadata={"company": "Microsoft", "source": "analyst-report.pdf"},
        )
    ]
    model = Mock()
    model.invoke.return_value = SimpleNamespace(
        content="Microsoft is expanding Copilot [analyst-report.pdf]."
    )

    monkeypatch.setattr(tool_module, "_get_retriever", lambda: retriever)
    monkeypatch.setattr(tool_module, "_get_model", lambda: model)

    result = tool_module.query_private_database.invoke(
        {"query": "What is Microsoft doing with Copilot?"}
    )

    assert result == "Microsoft is expanding Copilot [analyst-report.pdf]."
    retriever.invoke.assert_called_once_with(
        "What is Microsoft doing with Copilot?"
    )
    messages = model.invoke.call_args.args[0]
    assert "analyst-report.pdf" in messages[1].content
    assert "Microsoft is expanding its Copilot program." in messages[1].content


def test_query_private_database_hides_internal_failures(monkeypatch):
    def fail_to_load_retriever():
        raise RuntimeError("database password should not leak")

    monkeypatch.setattr(
        tool_module,
        "_get_retriever",
        fail_to_load_retriever,
    )

    with pytest.raises(ToolException) as exc_info:
        tool_module.query_private_database.invoke({"query": "test query"})

    assert str(exc_info.value) == tool_module.PRIVATE_DATABASE_ERROR_MESSAGE
    assert "password" not in str(exc_info.value)
