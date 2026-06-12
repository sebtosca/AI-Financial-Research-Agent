from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SimpleAgentState(TypedDict):
    """
    State for the financial research agent.

    messages:
        Conversation history. LangGraph appends new messages using add_messages.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]