from typing import Annotated, List, TypedDict, Optional
from langgraph.graph.message import add_messages

class SchemaPage(TypedDict):
    table_name: str
    content: str  # DDL + Samples
    attention_score: float

class AgentState(TypedDict):
    # Standard conversation history
    messages: Annotated[list, add_messages]
    # Paged Attention RAG: The subset of schema currently being 'attended'
    active_pages: List[SchemaPage]
    # The 'RAISE' logical plan (steps before SQL)
    logical_plan: Optional[str]
    current_sql: Optional[str]
    # Iteration tracking for self-correction
    iteration_count: int
    error_log: Optional[str]