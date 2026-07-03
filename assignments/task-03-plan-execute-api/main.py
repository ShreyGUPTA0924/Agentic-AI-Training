import os
import argparse
import asyncio
import logging
from dotenv import load_dotenv

# Load env variables before importing other modules
load_dotenv()

from rag.ingest import ingest_docs
from graph import builder
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Configure logging to be clean and readable for the user
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

async def run_pipeline(query: str, thread_id: str, max_iterations: int, re_ingest: bool):
    # 1. Idempotent Ingestion Check
    db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "chroma_db"))
    if re_ingest or not os.path.exists(db_dir):
        logger.info("ChromaDB directory not found or re-ingest requested. Starting ingestion...")
        ingest_docs()
    else:
        logger.info("ChromaDB vector database exists. Skipping ingestion.")

    # 2. Define Initial State
    initial_state = {
        "messages": [],
        "original_query": query,
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
        "max_iterations": max_iterations
    }

    config = {"configurable": {"thread_id": thread_id}}

    logger.info(f"Starting pipeline run on Thread ID: '{thread_id}' with Query: '{query}'")

    # 3. Stream graph execution steps
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "checkpoints.db"))
    try:
        async with AsyncSqliteSaver.from_conn_string(db_path) as memory:
            graph = builder.compile(checkpointer=memory)
            
            async for event in graph.astream(initial_state, config=config, stream_mode="updates"):
                for node_name, state_update in event.items():
                    print("\n" + "="*50)
                    print(f"*** Node Completed: {node_name} ***")
                    print("="*50)
                    
                    if "sub_tasks" in state_update and state_update["sub_tasks"]:
                        print("Sub-tasks Generated:")
                        for task in state_update["sub_tasks"]:
                            print(f"  - [{task['id']}] {task['question']}")
                            
                    if "retrieved_docs" in state_update and state_update["retrieved_docs"]:
                        print(f"Retrieved Chunks Count: {len(state_update['retrieved_docs'])}")
                        
                    if "aggregated_context" in state_update and state_update["aggregated_context"]:
                        print(f"Deduplicated Chunks Count: {len(state_update['aggregated_context'])}")
                        
                    if "draft" in state_update and state_update["draft"]:
                        print("\n--- Current Draft Answer ---")
                        print(state_update["draft"])
                        print("-" * 30)
                        
                    if "evaluation" in state_update and state_update["evaluation"]:
                        eval_res = state_update["evaluation"]
                        print("\n--- Evaluation Score ---")
                        print(f"  Faithfulness Score (1-5): {eval_res.get('faithfulness')}")
                        print(f"  Relevance Score (1-5):    {eval_res.get('relevance')}")
                        print(f"  Critique Feedback:        {eval_res.get('feedback')}")
                        
                    if "iteration" in state_update:
                        print(f"Next Iteration Index: {state_update['iteration']}")
                        
            # 4. Fetch the final state
            final_state = await graph.aget_state(config)
            draft = final_state.values.get("draft", "")
            eval_res = final_state.values.get("evaluation", {})
            iterations_run = final_state.values.get("iteration", 0)

            print("\n" + "#"*60)
            print("### PIPELINE COMPLETED ###")
            print("#"*60)
            print(f"Total Iterations: {iterations_run}")
            print(f"Final Scores -> Faithfulness: {eval_res.get('faithfulness')}, Relevance: {eval_res.get('relevance')}")
            print("\nFinal Answer:")
            print(draft)
            print("#"*60 + "\n")

    except Exception as e:
        logger.error(f"Failed to execute pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 03 Plan-and-Execute Agent CLI")
    parser.add_argument("--query", type=str, help="The query/question to search and answer")
    parser.add_argument("--thread-id", type=str, default="cli_test_thread", help="The unique thread ID for checkpointer persistence")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max loops allowed for self-reflection critique")
    parser.add_argument("--re-ingest", action="store_true", help="Force database document re-ingestion")
    
    args = parser.parse_args()

    query_input = args.query
    if not query_input:
        query_input = input("Enter your research question/query: ").strip()
        if not query_input:
            print("Query cannot be empty. Exiting.")
            exit(1)

    asyncio.run(run_pipeline(
        query=query_input,
        thread_id=args.thread_id,
        max_iterations=args.max_iterations,
        re_ingest=args.re_ingest
    ))
