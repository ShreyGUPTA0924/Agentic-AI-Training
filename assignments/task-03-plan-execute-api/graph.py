import os
import sqlite3
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.sqlite import SqliteSaver

from state import PlanExecuteState, SubTask
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.aggregator import aggregator_node
from agents.writer import writer_node
from agents.evaluator import evaluator_node

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Ensure data directory exists
os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), "data")), exist_ok=True)

def dispatch_researchers(state: PlanExecuteState) -> list[Send]:
    """Conditional edge that dynamic fans-out one researcher node per sub-task."""
    sub_tasks = state.get("sub_tasks", [])
    logger.info(f"Dispatching researchers for {len(sub_tasks)} sub-tasks.")
    return [
        Send("researcher_node", {**state, "current_sub_task": task})
        for task in sub_tasks
    ]

def route_after_evaluation(state: PlanExecuteState) -> str:
    """Conditional edge deciding whether to replan/loop or terminate after evaluation."""
    ev = state.get("evaluation", {})
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    logger.info(f"Routing after evaluation. Iteration: {iteration}/{max_iterations}. Evaluation: {ev}")

    # Check if we hit the hard cap limit
    if iteration >= max_iterations:
        logger.info("Reached maximum iterations. Terminating pipeline.")
        return "end"

    # Route based on quality scores
    faithfulness = ev.get("faithfulness", 0)
    relevance = ev.get("relevance", 0)
    
    if faithfulness < 3 or relevance < 3:
        logger.info(f"Draft scores (Faithfulness={faithfulness}, Relevance={relevance}) below threshold (<3). Routing back to Planner.")
        return "planner_node"

    logger.info(f"Draft scores (Faithfulness={faithfulness}, Relevance={relevance}) acceptable. Terminating pipeline.")
    return "end"

# Initialize StateGraph
builder = StateGraph(PlanExecuteState)

# Add Nodes
builder.add_node("planner_node", planner_node)
builder.add_node("researcher_node", researcher_node)
builder.add_node("aggregator_node", aggregator_node)
builder.add_node("writer_node", writer_node)
builder.add_node("evaluator_node", evaluator_node)

# Wire Nodes
builder.add_edge(START, "planner_node")
builder.add_conditional_edges("planner_node", dispatch_researchers, ["researcher_node"])
builder.add_edge("researcher_node", "aggregator_node")
builder.add_edge("aggregator_node", "writer_node")
builder.add_edge("writer_node", "evaluator_node")
builder.add_conditional_edges(
    "evaluator_node", 
    route_after_evaluation, 
    {
        "planner_node": "planner_node",
        "end": END
    }
)

# Set up SqliteSaver checkpointer for state persistence
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "checkpoints.db"))
logger.info(f"Initializing SqliteSaver database at '{db_path}'")
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)

# Compile Graph
graph = builder.compile(checkpointer=memory)
