import os
import sys
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

def get_mcp_client() -> MultiServerMCPClient:
    """Initialize and return the MultiServerMCPClient connecting to the tools and filesystem servers."""
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    
    # We run the Python MCP server using the same executable as the running client
    python_exe = sys.executable
    server_cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # On Windows, subprocess.Popen requires 'npx.cmd' or 'npx.bat' to launch successfully without shell=True
    npx_command = "npx.cmd" if sys.platform == "win32" else "npx"
    
    logger.info(f"Configuring MultiServerMCPClient:")
    logger.info(f" - Local tools server command: {python_exe} -m mcp_server.server in {server_cwd}")
    logger.info(f" - Filesystem server command: {npx_command} -y @modelcontextprotocol/server-filesystem {data_dir}")
    
    client = MultiServerMCPClient({
        "task04_tools": {
            "command": python_exe,
            "args": ["-m", "mcp_server.server"],
            "transport": "stdio",
            "cwd": server_cwd,
        },
        "filesystem": {
            "command": npx_command,
            "args": ["-y", "@modelcontextprotocol/server-filesystem", data_dir],
            "transport": "stdio",
        },
    })
    
    return client
