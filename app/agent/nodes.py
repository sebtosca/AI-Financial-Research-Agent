import logging
import time
from typing import Any, Callable

from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)


def create_agent_node(model_with_tools: Any, system_prompt: str) -> Callable:
    """
    Create the main agent node.

    This node:
    - validates the graph state
    - prepends the system prompt
    - calls the LLM with tools bound
    - logs tool-call decisions
    - returns the model response to LangGraph state
    """

    def agent_node(state: dict) -> dict:
        start_time = time.perf_counter()

        logger.info("Agent node started")

        try:
            if "messages" not in state or not state["messages"]:
                raise ValueError("State does not contain any messages")

            system_message = SystemMessage(content=system_prompt)

            messages = [system_message] + list(state["messages"])

            logger.info(
                "Calling LLM with tools | input_messages=%d",
                len(messages),
            )

            response = model_with_tools.invoke(messages)

            tool_calls = getattr(response, "tool_calls", None)

            if tool_calls:
                logger.info(
                    "Agent requested tool calls | tool_call_count=%d",
                    len(tool_calls),
                )

                for index, tool_call in enumerate(tool_calls, start=1):
                    logger.info(
                        "Tool call requested | index=%d | tool_name=%s",
                        index,
                        tool_call.get("name", "unknown"),
                    )

            else:
                logger.info("Agent generated final response")

            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "Agent node completed | duration_ms=%.2f",
                duration_ms,
            )

            return {"messages": [response]}

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.exception(
                "Agent node failed | duration_ms=%.2f",
                duration_ms,
            )

            raise RuntimeError(f"Agent node failed: {e}") from e

    return agent_node


def create_tool_node_with_logging(original_tool_node: Any) -> Callable:
    """
    Wrap a LangGraph ToolNode with production-style logging and error handling.
    """

    def tool_node_with_logging(state: dict) -> dict:
        start_time = time.perf_counter()

        logger.info("Tool node started")

        try:
            if "messages" not in state or not state["messages"]:
                raise ValueError("State does not contain any messages")

            result = original_tool_node.invoke(state)

            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "Tool node completed | duration_ms=%.2f",
                duration_ms,
            )

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.exception(
                "Tool node failed | duration_ms=%.2f",
                duration_ms,
            )

            raise RuntimeError(f"Tool node failed: {e}") from e

    return tool_node_with_logging