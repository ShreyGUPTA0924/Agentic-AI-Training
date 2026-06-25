from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages

class PipelineState(TypedDict):
    """Shared state of the Multi-Agent RAG pipeline."""
    messages: Annotated[list, add_messages]  # conversation history
    query: str                             # user's original question
    retrieved_docs: List[str]               # chunks from vector store
    approved: bool                         # did user approve the context?
    draft: str                             # Writer's drafted answer
    final_answer: str                      # polished final output
    next_agent: str                        # Supervisor routing decision
