import os
import logging
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from state import PlanExecuteState

logger = logging.getLogger(__name__)

class EvaluationResultSchema(BaseModel):
    faithfulness: int = Field(
        ..., 
        description="Faithfulness score (1-5). 1 = Answer invents facts not in retrieved context. 5 = Every claim is directly supported by retrieved context.", 
        ge=1, 
        le=5
    )
    relevance: int = Field(
        ..., 
        description="Relevance score (1-5). 1 = Answer is off-topic/misses query. 5 = Answer fully and directly addresses original query.", 
        ge=1, 
        le=5
    )
    feedback: str = Field(
        ..., 
        description="A concise, actionable critique feedback sentence detailing what is missing or incorrect and how to improve it."
    )

def evaluator_node(state: PlanExecuteState) -> dict:
    """Evaluator agent node that critiques the draft against the retrieved context and original query."""
    logger.info("Evaluator Agent invoked.")

    query = state.get("original_query", "")
    aggregated_context = state.get("aggregated_context", [])
    draft = state.get("draft", "")
    iteration = state.get("iteration", 0)

    logger.info(f"Evaluating draft on iteration {iteration}...")

    # Format the aggregated documents for comparison
    docs_formatted = ""
    for idx, doc in enumerate(aggregated_context):
        docs_formatted += f"--- Document [Doc {idx + 1}] ---\n{doc}\n\n"

    prompt = (
        "You are an expert quality evaluation and critique agent.\n"
        "Your task is to score the draft answer for Faithfulness and Relevance, and provide clear feedback.\n\n"
        "Scoring Rubric:\n"
        "- Faithfulness (1 to 5):\n"
        "  - 1: Answer invents facts not present in the retrieved chunks.\n"
        "  - 5: Every single claim in the answer is directly supported by a cited document chunk.\n"
        "- Relevance (1 to 5):\n"
        "  - 1: Answer is off-topic or fails to address the user's question.\n"
        "  - 5: Answer fully, directly, and comprehensively addresses the original user query.\n\n"
        f"Original User Query: {query}\n\n"
        f"Retrieved Documents Context:\n{docs_formatted}\n"
        f"Draft Answer to Evaluate:\n{draft}\n\n"
        "Evaluate the draft answer and output your scores and feedback."
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY env variable is not set.")

    try:
        # Using Llama 3.3 70B for strict evaluation capabilities
        llm = ChatGroq(
            api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        structured_llm = llm.with_structured_output(EvaluationResultSchema)
        result: EvaluationResultSchema = structured_llm.invoke(prompt)
        
        eval_dict = result.model_dump()
        logger.info(f"Evaluator scores -> Faithfulness: {eval_dict['faithfulness']}, Relevance: {eval_dict['relevance']}. Feedback: '{eval_dict['feedback']}'")
        
        return {
            "evaluation": eval_dict,
            "iteration": iteration + 1
        }
    except Exception as e:
        logger.error(f"Error in Evaluator LLM call: {e}", exc_info=True)
        # Fallback evaluation score to avoid failing the pipeline (assume moderate quality)
        fallback_eval = {
            "faithfulness": 3,
            "relevance": 3,
            "feedback": "Fallback evaluation triggered due to LLM error."
        }
        logger.warning(f"Falling back to basic evaluation: {fallback_eval}")
        return {
            "evaluation": fallback_eval,
            "iteration": iteration + 1
        }
