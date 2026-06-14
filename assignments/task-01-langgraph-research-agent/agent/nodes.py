from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    ToolMessage,
    SystemMessage
)

from .tools import (
    search_web,
    calculate,
    get_current_date
)

load_dotenv()


# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# Bind tools to the LLM
llm_with_tools = llm.bind_tools(
    [
        search_web,
        calculate,
        get_current_date
    ]
)


# Tool registry
available_tools = {
    "search_web": search_web,
    "calculate": calculate,
    "get_current_date": get_current_date
}


# -------------------------
# Agent Node
# -------------------------

def agent_node(state):

    # Instructions to improve tool selection
    system_message = SystemMessage(
        content="""
You are a helpful research assistant.

Tool Usage Rules:
- Use search_web for research, facts, companies, technologies, news, and information gathering.
- Use get_current_date for any date-related question.
- Use calculate only for mathematical expressions.
- Use previous conversation context for follow-up questions.
- If no tool is needed, answer directly.
"""
    )

    response = llm_with_tools.invoke(
        [system_message] + state["messages"]
    )

    return {
        "messages": [response]
    }


# -------------------------
# Tool Node
# -------------------------

def tool_node(state):

    # Latest AI message containing tool requests
    last_message = state["messages"][-1]

    tool_messages = []

    search_results = []

    topic = state.get("topic", "")

    for tool_call in last_message.tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call["args"]

        selected_tool = available_tools[tool_name]

        result = selected_tool.invoke(tool_args)

        # Save search results in state
        if tool_name == "search_web":

            search_results = result

            if isinstance(tool_args, dict):
                topic = tool_args.get("query", "")

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

    return {
        "messages": tool_messages,
        "search_results": search_results,
        "topic": topic
    }


# -------------------------
# Summarizer Node
# -------------------------

def summarizer_node(state):

    latest_user_query = ""

    # Find latest user message
    for message in reversed(state["messages"]):

        if message.__class__.__name__ == "HumanMessage":

            latest_user_query = message.content

            break

    summary_prompt = f"""
You are a professional research assistant.

Answer ONLY the latest user question.

Latest User Question:
{latest_user_query}

Research Topic:
{state.get("topic", "")}

Search Results:
{state.get("search_results", [])}

Instructions:
- Give a direct answer.
- Use bullet points when appropriate.
- If the question is a research query, provide:
    - Overview
    - Key Findings
    - Important Companies (if relevant)
    - Conclusion
- Do not repeat old conversation history.
- Focus only on the current question.
"""

    response = llm.invoke(summary_prompt)

    return {
        "messages": [response],
        "final_summary": response.content
    }