from retrieval.attention_rag import PagedAttentionManager
from .prompts import PLANNER_PROMPT, GENERATOR_PROMPT

def attention_node(state: AgentState):
    """RAISE Phase 0: Update active schema pages in the Attention Window."""
    query = state["messages"][-1].content
    manager = PagedAttentionManager()
    top_tables = manager.update_attention(query)
    
    # We would call the MCP tool here to get the actual text
    return {"active_pages": [{"table_name": t, "content": "..."} for t in top_tables]}

def planner_node(state: AgentState):
    """RAISE Phase 1: Create the Logical Execution Plan (LEP)."""
    # Uses 'active_pages' to define JOIN paths
    return {"logical_plan": "1. Join Sales and Products. 2. Filter by Category."}

def sql_generator_node(state: AgentState):
    """RAISE Phase 2: Translate LEP to SQL."""
    # Takes logical_plan + active_pages -> SQL
    return {"current_sql": "SELECT * FROM sales JOIN products ..."}

def qa_node(state: AgentState):
    """RAISE Phase 3: Validation & Refinement."""
    # If execute_and_validate returns row_count=0, it triggers a retry
    if "error" in state["error_log"] or not state.get("data"):
        return {"iteration_count": state["iteration_count"] + 1}
    return {"status": "complete"}