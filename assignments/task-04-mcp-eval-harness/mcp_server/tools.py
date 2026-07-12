import os
import ast
import sys
import operator
import logging
import contextlib
from datetime import datetime

logger = logging.getLogger(__name__)

# Directory paths relative to tools.py
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db"))

# Suppress progress bars and warnings from huggingface and libraries writing to stdout
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize HuggingFace embeddings globally while redirecting stdout to stderr
# to prevent any progress bars or stdout output from corrupting MCP stdio transport.
with contextlib.redirect_stdout(sys.stderr):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    logger.info("Initializing HuggingFaceEmbeddings globally (redirecting stdout to stderr)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Retrieve Tool Implementation ---
def retrieve(query: str, k: int = 4) -> list[str]:
    """Retrieve top-k relevant document chunks from the local vector database."""
    logger.info(f"Retrieving top-{k} chunks for query: '{query}'")

    # Connect to ChromaDB
    if not os.path.exists(DB_DIR):
        logger.warning(f"ChromaDB directory '{DB_DIR}' not found. Ingestion may not have run.")
        return ["Error: Vector database not initialized. Please run document ingestion."]

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    # Perform similarity search
    results = db.similarity_search(query, k=k)
    logger.info(f"Retrieved {len(results)} chunks successfully.")

    # Return page contents
    return [doc.page_content for doc in results]

# --- Safe Calculator Implementation ---
operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        return operators[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)
        return operators[type(node.op)](operand)
    raise TypeError("Unsupported expression")

def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        expression = expression.strip()
        tree = ast.parse(expression, mode="eval")
        result = safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"

# --- Get Current Date Implementation ---
def get_current_date() -> str:
    """Return today's date."""
    return datetime.now().strftime("%Y-%m-%d")
