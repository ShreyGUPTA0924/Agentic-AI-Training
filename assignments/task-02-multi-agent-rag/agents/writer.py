import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from state import PipelineState

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def writer_node(state: PipelineState) -> dict:
    """Writer agent node that synthesizes the retrieved chunks into a cited, structured answer."""
    logger.info("Writer Agent invoked.")

    query = state.get("query", "")
    retrieved_docs = state.get("retrieved_docs", [])
    approved = state.get("approved", False)

    logger.info(f"Writer inputs -> Query: '{query}', Docs: {len(retrieved_docs)}, Approved: {approved}")

    # Safety check: ensure retrieved docs are approved before writing
    if not approved:
        logger.warning("Retrieved docs are not approved. Writer node should not execute.")
        return {"draft": "", "final_answer": "Error: Document chunks were not approved."}

    if not retrieved_docs:
        logger.warning("No retrieved docs to draft response from.")
        no_context_msg = "I do not have any source documents to answer this question."
        return {"draft": no_context_msg, "final_answer": no_context_msg}

    # Format the retrieved documents with indices for citation
    docs_formatted = ""
    for idx, doc in enumerate(retrieved_docs):
        docs_formatted += f"--- Document [Doc {idx + 1}] ---\n{doc}\n\n"

    # Define the writer prompt
    prompt = (
        "You are an expert technical writer and research synthesis assistant.\n"
        "Your task is to answer the user query based ONLY on the retrieved documents provided below.\n"
        "Do NOT make up facts or use outside knowledge. Rely strictly on the text provided.\n\n"
        "Guidelines:\n"
        "1. Write a clear, comprehensive, and well-structured answer.\n"
        "2. You MUST cite the specific document using its identifier (e.g., [Doc 1], [Doc 2]) whenever you reference facts from it.\n"
        "3. If the retrieved documents do not contain the answer, state: 'I could not find the answer in the retrieved context.'\n\n"
        f"Retrieved Documents:\n{docs_formatted}\n"
        f"User Query: {query}\n\n"
        "Answer:"
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not found. Cannot invoke Writer Agent.")
        error_msg = "Error: GROQ_API_KEY is missing."
        return {"draft": error_msg, "final_answer": error_msg}

    try:
        # Using Llama 3.1 8B for fast generation
        llm = ChatGroq(
            api_key=api_key,
            model="llama-3.1-8b-instant",
            temperature=0.3  # slightly creative but aligned
        )
        response = llm.invoke(prompt)
        answer = response.content.strip()
        
        logger.info("Successfully generated cited answer draft.")
        return {
            "draft": answer,
            "final_answer": answer
        }
    except Exception as e:
        logger.error(f"Error in Writer Agent LLM call: {e}", exc_info=True)
        error_msg = f"Error generating answer: {e}"
        return {
            "draft": error_msg,
            "final_answer": error_msg
        }
    
if __name__ == "__main__":
    # Isolated test logic
    state_mock = {
        "query": "Explain self attention.",
        "retrieved_docs": [
            "At the core of the Transformer is the self-attention mechanism which computes token associations.",
            "Self-attention uses Q, K, V matrices to calculate softmax score."
        ],
        "approved": True,
        "draft": "",
        "final_answer": ""
    }
    print(writer_node(state_mock))
