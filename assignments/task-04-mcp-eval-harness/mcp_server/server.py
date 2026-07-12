import os
import sys
import logging

# `mcp dev` imports this file directly by path and only adds this file's own
# directory to sys.path, so the project root (needed to resolve the
# `mcp_server` package for the absolute import below) must be added manually.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP
from mcp_server.tools import (
    retrieve as tools_retrieve,
    calculate as tools_calculate,
    get_current_date as tools_get_current_date
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("task04-tools")

@mcp.tool()
def retrieve_context(query: str, k: int = 4) -> list[str]:
    """Retrieve the top-k relevant chunks from the local document store."""
    return tools_retrieve(query, k)

@mcp.tool()
def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    return tools_calculate(expression)

@mcp.tool()
def get_current_date() -> str:
    """Return today's date."""
    return tools_get_current_date()

# --- Bonus Challenges ---
# 1. MCP Resource: Exposing eval/report.md as a resource
@mcp.resource("eval://report")
def get_eval_report() -> str:
    """Read the latest evaluation report."""
    report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval", "report.md"))
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading report: {e}"
    return "No evaluation report found yet. Run evaluation harness first."

# 2. MCP Prompt: Exposing agent's system prompt
@mcp.prompt()
def agent_system_prompt() -> str:
    """System prompt for the agent, helping configure its behavior."""
    return (
        "You are an expert technical assistant with access to local documentation, filesystem, and calculation tools.\n"
        "Always rely on the tools at your disposal to answer questions:\n"
        "- Use retrieve_context for any question about RAG, vector databases, LangGraph, prompt engineering, or machine learning.\n"
        "- Use filesystem tools (like list_directory, view_file, etc.) to explore the files and folders in your data directory.\n"
        "- Use calculate for mathematical expressions.\n"
        "- Use get_current_date for the current date or year.\n\n"
        "Guidelines:\n"
        "1. Write clear, comprehensive, and well-structured responses.\n"
        "2. Rely strictly on retrieved document contents for RAG queries and do not make up facts.\n"
        "3. Explicitly cite the document sources when answering questions using retrieved context."
    )

if __name__ == "__main__":
    mcp.run(transport="stdio")
