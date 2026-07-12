from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """The state of our LangGraph agent."""
    messages: Annotated[list, add_messages]
