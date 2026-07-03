import operator
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class SubTask(TypedDict):
    id: str          # e.g. "sub_0", "sub_1"
    question: str    # the focused sub-question

class EvaluationResult(TypedDict):
    faithfulness: int  # 1–5: are claims supported by retrieved context?
    relevance: int     # 1–5: does the answer address the original query?
    feedback: str      # one sentence of actionable critique

class PlanExecuteState(TypedDict):
    messages:           Annotated[list, add_messages]
    original_query:     str               # user's original question (never modified)
    sub_tasks:          list[SubTask]     # planner's decomposition
    retrieved_docs:     Annotated[list, operator.add]  # fan-in accumulator
    aggregated_context: list[str]         # deduplicated chunks ready for writer
    draft:              str               # writer's current draft
    final_answer:       str               # approved final output
    evaluation:         EvaluationResult  # latest evaluator scores
    iteration:          int               # reflection loop counter (starts at 0)
    max_iterations:     int               # hard cap (default 3)
    current_sub_task:   SubTask           # the sub-task assigned to a specific parallel worker branch
