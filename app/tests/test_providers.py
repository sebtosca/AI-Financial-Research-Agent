import importlib

import pytest
from langchain_openai import ChatOpenAI

from app.providers.chat import build_chat_model, get_default_chat_model
from app.providers.embeddings import build_embedding_model


def test_build_chat_model_dispatches_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    model = build_chat_model(provider="openai", model="gpt-4o-mini", temperature=0.0)

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"


def test_build_chat_model_unknown_provider_raises_value_error():
    with pytest.raises(ValueError, match="Unknown chat provider"):
        build_chat_model(provider="not-a-provider", model="x", temperature=0.0)


def test_build_chat_model_anthropic_requires_api_key_before_import(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        build_chat_model(provider="anthropic", model="claude-3-5-sonnet-latest", temperature=0.0)


def test_build_chat_model_google_requires_api_key_before_import(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
        build_chat_model(provider="google", model="gemini-1.5-pro", temperature=0.0)


def test_build_embedding_model_unknown_provider_raises_value_error():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        build_embedding_model(provider="not-a-provider", model="x")


def test_build_embedding_model_anthropic_not_implemented():
    with pytest.raises(NotImplementedError):
        build_embedding_model(provider="anthropic", model="x")


def test_get_default_chat_model_matches_prior_openai_behavior(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("app.providers.chat.OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setattr("app.providers.chat.OPENAI_TEMPERATURE", 0.0)

    model = get_default_chat_model()

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"
    assert model.temperature == 0.0


def test_sentiment_build_model_still_produces_chat_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    sentiment_module = importlib.import_module("app.tools.sentiment_analysis")

    model = sentiment_module._build_model()

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == sentiment_module.MODEL_NAME
    assert model.temperature == sentiment_module.TEMPERATURE


def test_private_database_get_model_still_produces_chat_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    private_db_module = importlib.import_module("app.tools.query_private_database")
    private_db_module._get_model.cache_clear()

    model = private_db_module._get_model()

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == private_db_module.PRIVATE_DATABASE_MODEL

    private_db_module._get_model.cache_clear()


def test_graph_build_model_still_produces_chat_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    graph_module = importlib.import_module("app.agent.graph")

    model_with_tools = graph_module.build_model([])

    assert isinstance(model_with_tools.bound, ChatOpenAI)
