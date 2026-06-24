import os
import sys
import logging
from dotenv import load_dotenv
from graph import app
from langgraph.types import Command

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
# Suppress noisy library logs to keep CLI output clean
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def run_pipeline() -> None:
    """CLI flow to run the Multi-Agent RAG pipeline with Human-in-the-Loop."""
    print("\n" + "="*60)
    print("      MULTI-AGENT RAG PIPELINE WITH HUMAN-IN-THE-LOOP      ")
    print("="*60)
    
    # 1. Check API Key
    if not os.getenv("GROQ_API_KEY"):
        print("[Error] GROQ_API_KEY is not set in the environment or .env file.")
        print("Please configure your GROQ_API_KEY to proceed.")
        return

    # 2. Get User Question
    query = input("\nAsk a question (e.g. about Transformer architecture or RLHF):\n> ").strip()
    if not query:
        print("Empty query. Exiting.")
        return

    # Thread ID configuration for graph checkpointing
    config = {"configurable": {"thread_id": "rag_session_1"}}

    # Initialize graph input
    initial_input = {
        "query": query,
        "messages": [],
        "approved": False,
        "draft": "",
        "final_answer": "",
        "retrieved_docs": []
    }

    print("\n[System] Starting Multi-Agent workflow...")
    
    # Current input parameter for streaming
    current_input = initial_input

    while True:
        try:
            # Execute/Resume the graph and stream values
            events = app.stream(current_input, config, stream_mode="values")
            for event in events:
                # We can trace routing changes or updates
                pass

            # Inspect graph execution state
            state = app.get_state(config)

            # Case A: Graph completed execution (no next node)
            if not state.next:
                final_answer = state.values.get("final_answer", "")
                print("\n" + "="*60)
                print("                     FINAL ANSWER                      ")
                print("="*60)
                print(final_answer)
                print("="*60 + "\n")
                break

            # Case B: Graph paused before/at the human review node
            if "human_review_node" in state.next:
                # Identify if there is a dynamic active interrupt
                active_tasks = state.tasks
                has_active_interrupt = False
                interrupt_value = None

                for task in active_tasks:
                    if task.name == "human_review_node" and task.interrupts:
                        has_active_interrupt = True
                        interrupt_value = task.interrupts[0].value
                        break

                # Subcase B1: Paused BEFORE entering human_review_node (due to interrupt_before)
                if not has_active_interrupt:
                    logger.info("Graph paused before human_review_node. Resuming to run node...")
                    current_input = None  # Resuming execution requires passing None
                    continue

                # Subcase B2: Paused INSIDE human_review_node (due to dynamic interrupt() call)
                retrieved_docs = interrupt_value.get("retrieved_docs", [])
                
                print("\n" + "="*60)
                print("                 HUMAN CONTEXT REVIEW                 ")
                print("="*60)
                print(f"Current Query: '{state.values.get('query')}'")
                print(f"\nRetrieved Chunks ({len(retrieved_docs)}):")
                
                for idx, chunk in enumerate(retrieved_docs):
                    print(f"\n[Doc {idx + 1}]:")
                    print(chunk)
                print("="*60)

                # Prompt user for approval or query refinement
                user_choice = ""
                while not user_choice:
                    print("\nOptions:")
                    print("  y / yes : Approve context and proceed to Writer")
                    print("  n / no  : Reject context and return to Supervisor")
                    print("  r <query>: Refine search with a new custom query")
                    
                    raw_input = input("\nYour choice: ").strip()
                    
                    if raw_input.lower() in ["y", "yes", "approve"]:
                        user_choice = "y"
                    elif raw_input.lower() in ["n", "no", "reject"]:
                        user_choice = "n"
                    elif raw_input.lower().startswith("r "):
                        user_choice = raw_input  # Keep the 'r <query>' input
                    else:
                        print("[Error] Invalid option. Please choose y, n, or r.")

                # Resume the graph by passing the user decision
                current_input = Command(resume=user_choice)
                print("\n[System] Resuming workflow with user feedback...")
                continue
            
            else:
                # Handle unexpected node pauses
                logger.warning(f"Workflow paused at unexpected node: {state.next}. Resuming...")
                current_input = None
                continue

        except Exception as e:
            logger.error(f"Error during graph execution: {e}", exc_info=True)
            print(f"\n[Error] Execution Error: {e}")
            break

    print("[System] Session ended. Goodbye!")

if __name__ == "__main__":
    run_pipeline()
