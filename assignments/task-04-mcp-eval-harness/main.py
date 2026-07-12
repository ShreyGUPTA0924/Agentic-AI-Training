import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Ensure assignments/task-04-mcp-eval-harness is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.graph import get_agent_graph

# Configure logging to show info messages but suppress noisy library logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

async def interactive_chat():
    load_dotenv()
    
    print("\n========================================================")
    print("Welcome to the Task 04 MCP-Powered LangGraph Agent!")
    print("Connecting to local tools server and external filesystem server...")
    print("Type 'exit' or 'quit' to end the session.")
    print("========================================================\n")
    
    # Initialize the compiled LangGraph agent graph
    try:
        graph = await get_agent_graph()
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        print("Please check your .env API keys and ensure you run under the correct virtual environment.")
        return

    # Use a single thread ID to maintain state across conversation turns
    config = {
        "configurable": {
            "thread_id": "interactive_session"
        },
        "metadata": {
            "mode": "interactive"
        }
    }
    
    # Initial state
    state = {"messages": []}
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
                
            print("\nAgent: thinking...")
            
            # Run the agent graph
            state = await graph.ainvoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            
            # Print the agent's response
            last_msg = state["messages"][-1]
            if isinstance(last_msg, AIMessage):
                print(f"\nAgent: {last_msg.content}")
            else:
                print(f"\nAgent: {last_msg}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error during agent invocation: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(interactive_chat())
