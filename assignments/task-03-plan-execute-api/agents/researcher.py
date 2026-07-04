import logging
import asyncio
from rag.retriever import retrieve
from state import PlanExecuteState, SubTask

logger = logging.getLogger(__name__)

async def researcher_node(state: PlanExecuteState) -> dict:
    """Parallel researcher agent that retrieves document chunks for a specific sub-task."""
    # current_sub_task is injected into the state by the Send() dispatcher
    task: SubTask = state.get("current_sub_task")
    if not task:
        logger.warning("No current sub-task found in state for researcher node.")
        return {"retrieved_docs": []}

    sub_id = task.get("id")
    question = task.get("question", "")
    
    logger.info(f"[Researcher-{sub_id}] Starting research for: '{question}'")
    
    # Introduce a brief delay to simulate work and allow visual confirmation of parallelism in logs
    await asyncio.sleep(1.0)
    
    try:
        # Query ChromaDB retrieve function
        chunks = retrieve(question, k=4)
        logger.info(f"[Researcher-{sub_id}] Finished. Found {len(chunks)} chunks.")
        return {"retrieved_docs": chunks}
    except Exception as e:
        logger.error(f"[Researcher-{sub_id}] Error in retrieval: {e}", exc_info=True)
        return {"retrieved_docs": []}
