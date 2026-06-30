from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import (
    agent_node,
    tool_node,
    summarizer_node
)

# Create graph builder using AgentState as shared state
graph_builder = StateGraph(AgentState)


# -------------------------
# Register graph nodes
# -------------------------

graph_builder.add_node(
    "agent_node",
    agent_node
)

graph_builder.add_node(
    "tool_node",
    tool_node
)

graph_builder.add_node(
    "summarizer_node",
    summarizer_node
)


# -------------------------
# Routing logic after agent
# -------------------------

def route_after_agent(state):

    # Get latest message produced by agent
    last_message = state["messages"][-1]

    # If agent requested a tool, go to tool node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"

    # Otherwise stop execution
    return END


# -------------------------
# Define graph edges
# -------------------------

# Start execution from agent node
graph_builder.add_edge(
    START,
    "agent_node"
)

# Agent decides whether to use tools or finish
graph_builder.add_conditional_edges(
    "agent_node",
    route_after_agent
)

# Tool output is sent for summarization
graph_builder.add_edge(
    "tool_node",
    "summarizer_node"
)

# End graph after summary generation
graph_builder.add_edge(
    "summarizer_node",
    END
)


# -------------------------
# Memory configuration
# -------------------------

# Store conversation state across turns
memory = MemorySaver()


# -------------------------
# Compile executable graph
# -------------------------

graph = graph_builder.compile(
    checkpointer=memory
)