from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Tracks the chat history
    messages: Annotated[list, add_messages]
    # The refined schema subset relevant to the current question
    schema_context: str
    # The current iteration of the query plan
    query_plan: List[str]
    # The generated SQL
    current_sql: str
    # Results from intermediate execution
    execution_result: Optional[str]
    # Number of refinement loops performed
    iteration_count: int
    # Quality Assurance feedback
    qa_feedback: Optional[str]