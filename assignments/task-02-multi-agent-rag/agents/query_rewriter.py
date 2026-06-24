import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from state import PipelineState

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def query_rewriter_node(state: PipelineState) -> dict:
    """Query Rewriter node that rephrases the user query to optimize for semantic retrieval."""
    query = state.get("query", "")
    logger.info(f"Query Rewriter invoked. Original Query: '{query}'")

    if not query:
        return {}

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found. Skipping query rewriting.")
        return {}

    prompt = (
        "You are an expert Query Rewriter for semantic search systems.\n"
        "Your task is to rewrite the user's input query to optimize its recall and precision when searching a vector database.\n"
        "The database contains documentation on the Transformer Architecture and Reinforcement Learning from Human Feedback (RLHF).\n"
        "Rewrite the query by expanding abbreviations, adding relevant synonyms, or framing it as a search query, while keeping the core intent unchanged.\n"
        "Do NOT include any introduction, explanation, or conversational text. Output ONLY the rewritten query text.\n\n"
        f"Original Query: {query}\n"
        "Rewritten Query:"
    )

    try:
        llm = ChatGroq(
            api_key=api_key,
            model="llama-3.1-8b-instant",
            temperature=0.0
        )
        response = llm.invoke(prompt)
        rewritten_query = response.content.strip().strip('"')
        logger.info(f"Query Rewriter output: '{rewritten_query}'")
        return {"query": rewritten_query}
    except Exception as e:
        logger.error(f"Error in query rewriter: {e}", exc_info=True)
        # In case of API failure, return unchanged query
        return {"query": query}
