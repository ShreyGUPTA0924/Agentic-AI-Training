import os
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from state import PipelineState
from agents.supervisor import supervisor_node
from agents.query_rewriter import query_rewriter_node
from agents.researcher import researcher_node
from agents.writer import writer_node

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def human_review_node(state: PipelineState) -> dict:
    """Human review node that pauses graph execution using LangGraph interrupt."""
    logger.info("Human Review Node entered. Pausing for human feedback...")
    
    # Pause execution and yield retrieved docs for review
    decision = interrupt({
        "retrieved_docs": state.get("retrieved_docs", []),
        "message": "Review the retrieved context above. Approve? (y/n/r <new query>)"
    })
    
    logger.info(f"Human Review Node resumed. User decision: '{decision}'")
    decision_str = str(decision).strip()
    
    if decision_str.lower() in ["y", "yes"]:
        return {"approved": True}
    elif decision_str.lower() in ["n", "no"]:
        return {"approved": False}
    elif decision_str.lower().startswith("r "):
        new_query = decision_str[2:].strip()
        logger.info(f"User requested research query refinement: '{new_query}'")
        return {
            "query": new_query,
            "retrieved_docs": [],  # Clear retrieved docs to force research reload
            "approved": False
        }
        
    logger.warning(f"Unrecognized decision '{decision_str}'. Defaulting to not approved.")
    return {"approved": False}

# Routing function from supervisor node
def route_supervisor(state: PipelineState) -> str:
    next_agent = state.get("next_agent", "researcher")
    logger.info(f"Routing from Supervisor based on next_agent: '{next_agent}'")
    if next_agent == "researcher":
        return "query_rewriter"
    elif next_agent == "writer":
        return "writer"
    return "end"

# Routing function from human review node
def route_human_review(state: PipelineState) -> str:
    approved = state.get("approved", False)
    query = state.get("query", "")
    retrieved_docs = state.get("retrieved_docs", [])
    
    logger.info(f"Routing from Human Review -> approved: {approved}, query_has_docs: {bool(retrieved_docs)}")
    
    if approved:
        return "writer"
        
    # If not approved and retrieved_docs was cleared (Case 3: r <new query>)
    # we route back to the query_rewriter to start research again
    if not retrieved_docs and query:
        return "query_rewriter"
        
    # If just rejected (Case 2: n), route back to supervisor to let it decide next steps
    return "supervisor"

# Initialize state graph
workflow = StateGraph(PipelineState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("query_rewriter", query_rewriter_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("human_review_node", human_review_node)
workflow.add_node("writer", writer_node)

# Set entry point
workflow.set_entry_point("supervisor")

# Configure conditional edges from supervisor
workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "query_rewriter": "query_rewriter",
        "writer": "writer",
        "end": END
    }
)

# Add basic transition edges
workflow.add_edge("query_rewriter", "researcher")
workflow.add_edge("researcher", "human_review_node")

# Configure conditional edges from human review
workflow.add_conditional_edges(
    "human_review_node",
    route_human_review,
    {
        "writer": "writer",
        "query_rewriter": "query_rewriter",
        "supervisor": "supervisor"
    }
)

# Writer flows back to supervisor for terminal check
workflow.add_edge("writer", "supervisor")

# Compile with checkpoint memory saver and pause before human_review_node
checkpointer = MemorySaver()
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review_node"]
)
