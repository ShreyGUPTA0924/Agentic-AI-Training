import os
import logging
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from state import PipelineState

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

class RouteDecision(BaseModel):
    """Structured output schema for routing decisions."""
    next: Literal["researcher", "writer", "end"] = Field(
        ..., 
        description="The next agent node to run: 'researcher' for RAG search, 'writer' to draft the answer, or 'end' to terminate."
    )
    reasoning: str = Field(
        ..., 
        description="Detailed, step-by-step reasoning explaining the routing decision."
    )

def get_supervisor_llm() -> ChatGroq:
    """Initialize the ChatGroq model for structured routing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    # Using Llama 3.1 8B model for fast structured routing
    return ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0.0
    )

def supervisor_node(state: PipelineState) -> dict:
    """Supervisor agent node that determines the next step in the RAG pipeline."""
    logger.info("Supervisor Agent invoked.")

    query = state.get("query", "")
    retrieved_docs = state.get("retrieved_docs", [])
    approved = state.get("approved", False)
    draft = state.get("draft", "")

    logger.info(f"Supervisor State Check -> docs: {len(retrieved_docs)}, approved: {approved}, draft_exists: {bool(draft)}")

    # Prompt constructing current state context
    prompt = (
        "You are the Supervisor Agent in a Multi-Agent Retrieval-Augmented Generation (RAG) system.\n"
        "Your task is to review the current pipeline state and decide which node to transition to next.\n\n"
        "Rules:\n"
        "1. If retrieved_docs is empty (no chunks have been fetched yet), you MUST choose 'researcher'.\n"
        "2. If retrieved_docs exist but approved is False (context not approved), you MUST choose 'researcher' to query again.\n"
        "3. If approved is True and draft is empty, you MUST choose 'writer' to generate the answer.\n"
        "4. If draft exists (answer already generated), you MUST choose 'end'.\n\n"
        f"Current Pipeline State:\n"
        f"- Original Query: '{query}'\n"
        f"- Number of retrieved docs: {len(retrieved_docs)}\n"
        f"- Approved by user: {approved}\n"
        f"- Draft response exists: {bool(draft)}\n"
    )

    try:
        llm = get_supervisor_llm()
        structured_llm = llm.with_structured_output(RouteDecision)
        decision: RouteDecision = structured_llm.invoke(prompt)

        logger.info(f"Supervisor Decision: {decision.next} | Reasoning: {decision.reasoning}")
        return {"next_agent": decision.next}
    except Exception as e:
        logger.error(f"Error in Supervisor Agent LLM call: {e}", exc_info=True)
        # Fallback rules in case of API issues to keep the system resilient
        fallback_next = "researcher"
        if len(retrieved_docs) > 0:
            if approved and not draft:
                fallback_next = "writer"
            elif draft:
                fallback_next = "end"
            else:
                fallback_next = "researcher"
        logger.info(f"Supervisor Fallback Decision: {fallback_next} (due to error)")
        return {"next_agent": fallback_next}
