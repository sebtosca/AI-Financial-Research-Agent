import logging
import os
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.tools import ToolException
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes import create_agent_node, create_tool_node_with_logging
from .prompts import PROMPT_VERSION, get_prompt
from .state import SimpleAgentState
from ..providers import get_chat_model_for_tier, get_default_chat_model
from ..tools import (
    analyze_sentiment,
    get_stock_history,
    get_stock_price,
    query_private_database,
    search_financial_news,
)

load_dotenv()

logger = logging.getLogger(__name__)


def _handle_tool_error(error: Exception) -> str:
    if isinstance(error, ToolException):
        return str(error)

    logger.error(
        "Unhandled tool error converted to safe tool response | error_type=%s",
        type(error).__name__,
    )
    return "The tool is temporarily unavailable. Continue with available data."


def _default_tools(with_rag: bool) -> list:
    tools = [
        get_stock_price,
        get_stock_history,
        search_financial_news,
        analyze_sentiment,
    ]

    if with_rag:
        tools.append(query_private_database)

    return tools


_ALL_TOOLS = {
    tool.name: tool
    for tool in (
        get_stock_price,
        get_stock_history,
        search_financial_news,
        analyze_sentiment,
        query_private_database,
    )
}


def tools_from_names(tool_names: tuple[str, ...]) -> list:
    return [_ALL_TOOLS[name] for name in tool_names if name in _ALL_TOOLS]


def should_continue(state: SimpleAgentState) -> Literal["tools", "end"]:
    """
    Determine whether the workflow should continue to the tool node
    or terminate with a final response.
    """

    messages = state.get("messages", [])

    if not messages:
        logger.error("Routing failed | no messages found in state")
        raise ValueError("State contains no messages")

    last_message = messages[-1]

    tool_calls = getattr(last_message, "tool_calls", None)

    if tool_calls:
        logger.info(
            "Routing decision | next_node=tools | tool_call_count=%d",
            len(tool_calls),
        )
        return "tools"

    logger.info("Routing decision | next_node=end")
    return "end"


def validate_environment() -> None:
    """
    Validate required environment variables.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is missing from environment variables"
        )


def build_model(tools: list, *, model_tier: Optional[str] = None):
    """
    Initialize and configure the LLM with bound tools.
    """

    validate_environment()

    if model_tier:
        logger.info("Initializing model | tier=%s", model_tier)
        model = get_chat_model_for_tier(model_tier)
    else:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
        logger.info(
            "Initializing model | model=%s | temperature=%.2f",
            model_name,
            temperature,
        )
        model = get_default_chat_model()

    return model.bind_tools(tools)


def create_financial_agent(
    agent_type: str = "full",
    with_memory: bool = True,
    with_rag: bool = True,
    tools: Optional[list] = None,
    model_tier: Optional[str] = None,
):
    """
    Create and compile the financial research agent graph.

    Args:
        agent_type: Prompt profile to use: traditional, basic, or full.
        with_memory: Whether to enable conversation checkpoint memory.
        with_rag: Whether default tools include the private database tool.
        tools: Optional explicit tool list. When provided, it takes precedence
            over with_rag.
        model_tier: Optional routing tier ("fast"/"capable") selecting the
            provider/model. Defaults to the OPENAI_MODEL configuration.
    """

    if tools is None:
        tools = _default_tools(with_rag)

    if not tools:
        raise ValueError("At least one tool must be configured")

    logger.info(
        (
            "Creating financial agent | agent_type=%s | tool_count=%d | "
            "rag_enabled=%s | memory_enabled=%s | prompt_version=%s"
        ),
        agent_type,
        len(tools),
        any(tool.name == query_private_database.name for tool in tools),
        with_memory,
        PROMPT_VERSION,
    )

    logger.info(
        "Registered tools | tools=%s",
        ", ".join(tool.name for tool in tools),
    )

    try:
        system_prompt = get_prompt(agent_type)

        model_with_tools = build_model(tools, model_tier=model_tier)

        workflow = StateGraph(SimpleAgentState)

        agent_node = create_agent_node(
            model_with_tools=model_with_tools,
            system_prompt=system_prompt,
        )

        original_tool_node = ToolNode(
            tools,
            handle_tool_errors=_handle_tool_error,
        )

        tool_node = create_tool_node_with_logging(
            original_tool_node
        )

        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END,
            },
        )

        workflow.add_edge("tools", "agent")

        if with_memory:
            logger.info("Compiling workflow | memory=enabled")

            memory = MemorySaver()

            graph = workflow.compile(
                checkpointer=memory
            )

        else:
            logger.info("Compiling workflow | memory=disabled")

            graph = workflow.compile()

        logger.info("Financial agent created successfully")

        return graph

    except Exception as e:
        logger.exception("Failed to create financial agent")

        raise RuntimeError(
            f"Failed to create financial agent: {e}"
        ) from e


def create_enhanced_financial_agent(
    with_rag: bool = True,
    with_memory: bool = True,
):
    """Create the full financial research agent with optional RAG access."""
    return create_financial_agent(
        agent_type="full",
        with_memory=with_memory,
        with_rag=with_rag,
    )
