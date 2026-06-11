from typing_extensions import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages

# Sequence: collection of things where order matters - similar to a list
# Annotated: Attach instructions to a variable
# add_messages: merge rule. when new messages arrive, append them.

class SimpleAgentState(TypedDict):
    """
    State for the financial research agent.
    Tracks the conversation history with message accumalation.
    """
    messages: Annotated[Sequence, add_messages]


