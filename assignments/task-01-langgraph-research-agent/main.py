from langchain_core.messages import HumanMessage

from agent.graph import graph


# Unique conversation id used by MemorySaver
# Same thread_id = same conversation memory
config = {
    "configurable": {
        "thread_id": "research-session"
    }
}


def main():

    # Application banner
    print("=" * 60)
    print("Personal Research Agent")
    print("Type 'exit' to quit")
    print("=" * 60)

    # Keep chat running until user exits
    while True:

        user_input = input("\nYou: ")

        # Gracefully stop the application
        if user_input.lower() in ["exit", "quit"]:
            print("\nAgent: Goodbye!")
            break

        # Create initial state required by AgentState
        initial_state = {

            # Current user message
            "messages": [
                HumanMessage(content=user_input)
            ],

            # Research topic (can be updated later)
            "topic": "",

            # Raw tool outputs
            "search_results": [],

            # Final summarized answer
            "final_summary": ""
        }

        # Execute LangGraph workflow
        result = graph.invoke(
            initial_state,
            config=config
        )

        # Display final AI response
        print(
            "\nAgent:",
            result["messages"][-1].content
        )


# Program entry point
if __name__ == "__main__":
    main()