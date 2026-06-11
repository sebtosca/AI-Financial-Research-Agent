import os 
import logging 
from typing import Literal # value must be exactly one of these literaly
from dotenv import load_dotenv

load_dotenv()

from app import agent
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .state import SimpleAgentState
from ..tools import get_stock_price, get_stock_history, search_financial_news, analyze_sentiment
from .prompts import TRADITIONAL_PROMPT, AGENT_CHARTER_BASIC, AGENT_CHARTER_FULL
from .nodes import create_agent_node, create_tool_node_with_logging

logger = logging.getLogger(__name__)

# Routing decision in the graph. SHould it continue and use a tool or end? 
def should_continue(state: SimpleAgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info("ROUTING: Continuing to TOOLS node")
        return "tools"

    logger.info("ROUTING: Ending workflow (final response ready)")
    return "end"

def create_financial_agent(agent_type: str = "full", with_memory: bool = True, tools: list = None):
    prompt_map = {
        "traditional": TRADITIONAL_PROMPT,
        "basic": AGENT_CHARTER_BASIC,
        "full": AGENT_CHARTER_FULL,
    }

    #Select system prompt based on agent type
    system_prompt = prompt_map.get(agent_type, AGENT_CHARTER_BASIC)

    if tools is None:
        tools = [
            get_stock_price,
            get_stock_history,
            search_financial_news,
            analyze_sentiment,
        ]

    logger.info(f"Creating {agent_type.upper()} agent with {len(tools)} tools")
    logger.info(f"   Tools: {', '.join(t.name for t in tools)}")

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_api_base=os.environ.get("OPENAI_API_BASE"),
    )

    model_with_tools = model.bind_tools(tools)
    workflow = StateGraph(SimpleAgentState)

    # create main AI agent node
    agent_node = create_agent_node(model_with_tools, system_prompt)

    # create original tool execution node
    original_tool_node = ToolNode(tools)

    tool_node = create_tool_node_with_logging(original_tool_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")

    # Add conditional routing logic
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
        logger.info("Enabling conversation memory")
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)
    else:
        logger.info("Memory disabled - stateless mode")
        graph = workflow.compile()

    logger.info("Agent created successfully")
    return graph
    