import logging
from rag.retriever import retrieve
from state import PipelineState

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def researcher_node(state: PipelineState) -> dict:
    """Researcher agent node that retrieves document chunks from the local database."""
    query = state.get("query", "")
    logger.info(f"Researcher Agent invoked. Query: '{query}'")

    if not query:
        logger.warning("Query is empty. Cannot retrieve documents.")
        return {"retrieved_docs": []}

    try:
        # Call the retrieve function (fetches top 4 chunks)
        docs = retrieve(query, k=4)
        logger.info(f"Researcher retrieved {len(docs)} document chunks.")
        
        # Log a snippet of retrieved docs for verification
        for idx, doc in enumerate(docs):
            snippet = doc[:80].replace('\n', ' ') + "..."
            logger.info(f"  [Chunk {idx+1}] {snippet}")

        return {"retrieved_docs": docs}
    except Exception as e:
        logger.error(f"Error in Researcher Agent retrieval: {e}", exc_info=True)
        return {"retrieved_docs": []}
