import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Load env variables
load_dotenv()

from api.schemas import ResearchRequest, ResearchStateResponse
from graph import builder
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rag.ingest import ingest_docs

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Task 03 - Plan-and-Execute Agent API")

@app.post("/research")
async def research(body: ResearchRequest):
    """Initiates a research workflow and streams progress events using SSE."""
    logger.info(f"Received research request on thread: '{body.thread_id}' with query: '{body.query}'")
    
    # 1. Idempotent Ingestion Check
    db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db"))
    if not os.path.exists(db_dir):
        logger.info("ChromaDB directory not found. Starting document ingestion...")
        try:
            ingest_docs()
        except Exception as e:
            logger.error(f"Failed to ingest documents during API request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Database ingestion error: {e}")

    # 2. Define Initial State
    initial_state = {
        "messages": [],
        "original_query": body.query,
        "sub_tasks": [],
        "retrieved_docs": [],
        "aggregated_context": [],
        "draft": "",
        "final_answer": "",
        "evaluation": {
            "faithfulness": 0,
            "relevance": 0,
            "feedback": ""
        },
        "iteration": 0,
        "max_iterations": body.max_iterations
    }

    config = {"configurable": {"thread_id": body.thread_id}}
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints.db"))

    async def event_generator():
        try:
            async with AsyncSqliteSaver.from_conn_string(db_path) as memory:
                graph = builder.compile(checkpointer=memory)
                
                # Stream events using graph.astream_events
                async for event in graph.astream_events(initial_state, config=config, version="v2"):
                    # Filter by node completion events (on_chain_end)
                    if event["event"] == "on_chain_end":
                        node_name = event["metadata"].get("langgraph_node")
                        if not node_name or event["name"] != node_name:
                            continue
                        
                        output = event["data"].get("output")
                        input_state = event["data"].get("input")
                        if not output:
                            continue
                        
                        if node_name == "planner_node":
                            sub_tasks = output.get("sub_tasks", [])
                            logger.info(f"Planner node completed. Emitting sub-tasks event.")
                            yield f"data: {json.dumps({'type': 'planner', 'sub_tasks': sub_tasks})}\n\n"
                            
                        elif node_name == "researcher_node":
                            sub_task = input_state.get("current_sub_task", {})
                            sub_task_id = sub_task.get("id", "unknown")
                            chunks_found = len(output.get("retrieved_docs", []))
                            logger.info(f"Researcher node {sub_task_id} completed. Emitting researcher event.")
                            yield f"data: {json.dumps({'type': 'researcher', 'sub_task_id': sub_task_id, 'chunks_found': chunks_found})}\n\n"
                            
                        elif node_name == "aggregator_node":
                            total_chunks = len(input_state.get("retrieved_docs", []))
                            after_dedup = len(output.get("aggregated_context", []))
                            logger.info(f"Aggregator node completed. Emitting aggregator event.")
                            yield f"data: {json.dumps({'type': 'aggregator', 'total_chunks': total_chunks, 'after_dedup': after_dedup})}\n\n"
                            
                        elif node_name == "writer_node":
                            draft = output.get("draft", "")
                            logger.info(f"Writer node completed. Emitting writer event.")
                            yield f"data: {json.dumps({'type': 'writer', 'draft_length': len(draft)})}\n\n"
                            
                        elif node_name == "evaluator_node":
                            eval_res = output.get("evaluation", {})
                            iteration = output.get("iteration", 0)
                            logger.info(f"Evaluator node completed. Emitting evaluator event.")
                            yield f"data: {json.dumps({'type': 'evaluator', 'faithfulness': eval_res.get('faithfulness', 0), 'relevance': eval_res.get('relevance', 0), 'iteration': iteration})}\n\n"
                
                # Fetch the final state checkpoint to get the final compiled answer
                state_val = await graph.aget_state(config)
                final_answer = state_val.values.get("draft", "")
                logger.info("Pipeline run finished. Emitting final event.")
                yield f"data: {json.dumps({'type': 'final', 'answer': final_answer})}\n\n"
            
        except Exception as stream_err:
            logger.error(f"Error during graph execution stream: {stream_err}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(stream_err)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/research/{thread_id}", response_model=ResearchStateResponse)
async def get_research_status(thread_id: str):
    """Retrieves the latest checkpointed state for a specific thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints.db"))
    try:
        async with AsyncSqliteSaver.from_conn_string(db_path) as memory:
            graph = builder.compile(checkpointer=memory)
            state_val = await graph.aget_state(config)
            if not state_val or not state_val.values:
                raise HTTPException(status_code=404, detail=f"Thread ID '{thread_id}' not found.")
                
            # Determine status based on checkpoint state
            eval_res = state_val.values.get("evaluation", {})
            iteration = state_val.values.get("iteration", 0)
            max_iterations = state_val.values.get("max_iterations", 3)
            
            faithfulness = eval_res.get("faithfulness", 0)
            relevance = eval_res.get("relevance", 0)
            
            if iteration >= max_iterations or (faithfulness >= 3 and relevance >= 3):
                status = "complete"
            else:
                status = "running"

            return ResearchStateResponse(
                status=status,
                state=state_val.values
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching checkpoint for thread '{thread_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
