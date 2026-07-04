import os
import logging
from langchain_groq import ChatGroq
from state import PlanExecuteState

logger = logging.getLogger(__name__)

def writer_node(state: PlanExecuteState) -> dict:
    """Writer agent node that synthesizes the aggregated chunks into a cited final answer."""
    logger.info("Writer Agent invoked.")
    
    query = state.get("original_query", "")
    aggregated_context = state.get("aggregated_context", [])
    evaluation = state.get("evaluation", {})
    feedback = evaluation.get("feedback", "") if evaluation else ""
    iteration = state.get("iteration", 0)

    logger.info(f"Writer inputs -> Query: '{query}', Context length: {len(aggregated_context)}, Iteration: {iteration}")

    if not aggregated_context:
        no_context_msg = "I do not have any retrieved documents or source material to answer this question."
        return {"draft": no_context_msg}

    # Format the aggregated documents for the prompt
    docs_formatted = ""
    for idx, doc in enumerate(aggregated_context):
        docs_formatted += f"--- Document [Doc {idx + 1}] ---\n{doc}\n\n"

    # Standard writer guidelines
    prompt = (
        "You are an expert technical writer and research synthesis assistant.\n"
        "Your task is to answer the user query based ONLY on the retrieved documents provided below.\n"
        "Do NOT make up facts or use outside knowledge. Rely strictly on the text provided.\n\n"
        "Guidelines:\n"
        "1. Write a clear, comprehensive, and well-structured answer.\n"
        "2. You MUST cite the specific document using its identifier (e.g. [Doc 1], [Doc 2]) whenever you reference facts from it.\n"
        "3. If the retrieved documents do not contain the answer, state: 'I could not find the answer in the retrieved context.'\n\n"
        f"Retrieved Documents:\n{docs_formatted}\n"
    )

    if iteration > 0 and feedback:
        # Incorporate critique feedback
        prompt += (
            f"IMPORTANT: A previous draft was evaluated and found to be lacking. Please address this feedback:\n"
            f"Critique Feedback: {feedback}\n\n"
        )

    prompt += f"User Query: {query}\n\nAnswer:"

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY env variable is not set.")

    try:
        # Using Llama 3.3 70B for high quality synthesis and constraint-following
        llm = ChatGroq(
            api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )
        response = llm.invoke(prompt)
        answer = response.content.strip()
        
        logger.info("Successfully generated answer draft.")
        return {"draft": answer}
    except Exception as e:
        logger.error(f"Error in Writer Agent LLM call: {e}", exc_info=True)
        # Fallback to Llama 3.1 8B in case of rate limit/context issues
        try:
            logger.info("Retrying Writer LLM with llama-3.1-8b-instant...")
            llm_fallback = ChatGroq(
                api_key=api_key,
                model="llama-3.1-8b-instant",
                temperature=0.3
            )
            response = llm_fallback.invoke(prompt)
            return {"draft": response.content.strip()}
        except Exception as fallback_err:
            logger.error(f"Fallback Writer LLM also failed: {fallback_err}", exc_info=True)
            return {"draft": f"Error generating answer: {e}"}
