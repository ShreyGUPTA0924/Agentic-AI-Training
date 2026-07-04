import os
import logging
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from state import PlanExecuteState, SubTask

logger = logging.getLogger(__name__)

class SubTaskSchema(BaseModel):
    id: str = Field(..., description="Unique sub-task identifier, e.g. sub_0, sub_1")
    question: str = Field(..., description="A focused sub-question targeting a specific aspect of the original query.")

class Plan(BaseModel):
    sub_tasks: list[SubTaskSchema] = Field(..., description="List of 2 to 4 distinct sub-tasks to fetch documents for.")
    reasoning: str = Field(..., description="Reasoning for how these sub-tasks address the main query.")

def planner_node(state: PlanExecuteState) -> dict:
    """Planner node that decomposes the query or refines it based on critique feedback."""
    logger.info("Planner Agent invoked.")
    
    iteration = state.get("iteration", 0)
    original_query = state.get("original_query", "")
    evaluation = state.get("evaluation", {})
    feedback = evaluation.get("feedback", "") if evaluation else ""

    logger.info(f"Iteration: {iteration}, Query: '{original_query}', Feedback: '{feedback}'")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY env variable is not set.")

    # Using Llama 3.3 70B for better planning capabilities
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )
    
    structured_llm = llm.with_structured_output(Plan)

    if iteration == 0 or not feedback:
        # First iteration planning
        prompt = (
            "You are a master planning agent.\n"
            "Your task is to decompose the following complex query into 2 to 4 distinct, focused sub-questions "
            "for research and document retrieval.\n"
            "Each sub-question must be specific, targeted, and answerable by searching a technical knowledge base.\n\n"
            f"Original Query: {original_query}\n\n"
            "Decompose this query into sub-questions."
        )
    else:
        # Replanning iteration incorporating evaluator feedback
        prompt = (
            "You are a master planning agent.\n"
            "A previous attempt to answer the user query failed. You must revise the planning process.\n"
            "Review the original query and the feedback from the evaluation system, then generate a new set of "
            "2 to 4 sub-questions. Focus on addressing the specific issues mentioned in the feedback.\n\n"
            f"Original Query: {original_query}\n"
            f"Critique Feedback: {feedback}\n\n"
            "Generate revised sub-questions to improve the final answer."
        )

    try:
        plan: Plan = structured_llm.invoke(prompt)
        # Convert schema format to state SubTask TypedDict format
        sub_tasks_dict = [{"id": task.id, "question": task.question} for task in plan.sub_tasks]
        logger.info(f"Planner successfully generated {len(sub_tasks_dict)} sub-tasks: {sub_tasks_dict}")
        
        return {
            "sub_tasks": sub_tasks_dict,
            "retrieved_docs": [],          # reset for new run
            "aggregated_context": [],      # reset for new run
            "draft": ""                    # reset for new run
        }
    except Exception as e:
        logger.error(f"Error in Planner LLM call: {e}", exc_info=True)
        # Fallback in case of failures: split query into two basic sub-questions
        fallback_tasks = [
            {"id": "sub_0", "question": f"Overview of {original_query}"},
            {"id": "sub_1", "question": f"Specific details or implementations of {original_query}"}
        ]
        logger.warning(f"Falling back to basic tasks: {fallback_tasks}")
        return {
            "sub_tasks": fallback_tasks,
            "retrieved_docs": [],
            "aggregated_context": [],
            "draft": ""
        }
