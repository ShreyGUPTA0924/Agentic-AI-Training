import logging
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from agent.llm import get_llm
from agent.mcp_client import get_mcp_client

logger = logging.getLogger(__name__)

# Minimal system prompt to prevent confusing the LLM about tool usage formatting.
# The tool definitions themselves contain the required schemas and descriptions.
SYSTEM_INSTRUCTION = (
    "You are an expert technical assistant with access to tools from two MCP servers: "
    "a local 'task04_tools' server (retrieve_context, calculate, get_current_date) and an external "
    "'filesystem' server (list_directory, read_file, etc.) scoped to the data directory.\n"
    "- For any question about document content (RAG, vector databases, LangGraph, prompt engineering, ML), "
    "ALWAYS call retrieve_context first instead of reading files directly with the filesystem tools. "
    "Rely strictly on the retrieved context to answer, do not invent facts, and cite the source document names.\n"
    "- Use the filesystem tools only for questions about what files/directories exist, not to read document content.\n"
    "- Use calculate for math and get_current_date for the current date."
)

async def get_agent_graph():
    """Build and compile the LangGraph ReAct agent using dynamically loaded MCP tools."""
    logger.info("Initializing MCP client and fetching tools...")
    client = get_mcp_client()
    
    # Retrieve tools from all connected MCP servers
    tools = await client.get_tools()
    logger.info(f"Loaded {len(tools)} tools from MCP servers: {[t.name for t in tools]}")
    
    # Initialize LLM using the shared factory
    llm = get_llm(temperature=0.0)
    
    # Use LangGraph's prebuilt React agent helper for robust tool history serialization
    graph = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_INSTRUCTION,
        checkpointer=MemorySaver()
    )
    
    return graph
