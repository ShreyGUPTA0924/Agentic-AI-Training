from pydantic import BaseModel, Field
from typing import Any, Optional

class ResearchRequest(BaseModel):
    query: str = Field(..., description="The original query to research.")
    thread_id: str = Field(..., description="The unique thread ID for tracking checkpoints.")
    max_iterations: Optional[int] = Field(3, description="Maximum iterations for self-reflection critique loops.")

class ResearchStateResponse(BaseModel):
    status: str = Field(..., description="Pipeline execution status, e.g. complete, running, or error.")
    state: Optional[dict[str, Any]] = Field(None, description="The latest PlanExecuteState values of the thread.")
