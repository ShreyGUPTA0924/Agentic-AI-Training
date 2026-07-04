import os
import logging
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db"))

def retrieve(query: str, k: int = 4) -> list[str]:
    """Retrieve top-k relevant document chunks from the local vector database."""
    logger.info(f"Retrieving top-{k} chunks for query: '{query}'")

    # 1. Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Connect to ChromaDB
    if not os.path.exists(DB_DIR):
        logger.warning(f"ChromaDB directory '{DB_DIR}' not found. Ingestion may not have run.")
        return []

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    # 3. Perform similarity search
    results = db.similarity_search(query, k=k)
    logger.info(f"Retrieved {len(results)} chunks successfully.")

    # Return only page contents
    return [doc.page_content for doc in results]

if __name__ == "__main__":
    # Test retrieval in isolation
    query = "What is the self-attention mechanism?"
    print(f"\n--- Testing retrieval with query: '{query}' ---")
    chunks = retrieve(query, k=2)
    for i, chunk in enumerate(chunks):
        print(f"\nResult {i+1}:")
        print(chunk)
    print("\n--- Testing retrieval completed ---")
