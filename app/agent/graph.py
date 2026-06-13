import logging
import os
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes import create_agent_node, create_tool_node_with_logging
from .prompts import PROMPT_VERSION, get_prompt
from .state import SimpleAgentState
from ..tools import (
    analyze_sentiment,
    get_stock_history,
    get_stock_price,
    query_private_database,
    search_financial_news,
)

load_dotenv()

logger = logging.getLogger(__name__)


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


def build_model(tools: list):
    """
    Initialize and configure the LLM with bound tools.
    """

    validate_environment()

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    logger.info(
        "Initializing model | model=%s | temperature=%.2f",
        model_name,
        temperature,
    )

    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE") or None,
    )

    return model.bind_tools(tools)


def create_financial_agent(
    agent_type: str = "full",
    with_memory: bool = True,
    tools: Optional[list] = None,
):
    """
    Create and compile the financial research agent graph.
    """

    if tools is None:
        tools = [
            get_stock_price,
            get_stock_history,
            search_financial_news,
            analyze_sentiment,
            query_private_database,
        ]

    logger.info(
        "Creating financial agent | agent_type=%s | tool_count=%d | prompt_version=%s",
        agent_type,
        len(tools),
        PROMPT_VERSION,
    )

    logger.info(
        "Registered tools | tools=%s",
        ", ".join(tool.name for tool in tools),
    )

    try:
        system_prompt = get_prompt(agent_type)

        model_with_tools = build_model(tools)

        workflow = StateGraph(SimpleAgentState)

        agent_node = create_agent_node(
            model_with_tools=model_with_tools,
            system_prompt=system_prompt,
        )

        original_tool_node = ToolNode(tools)

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
