from langgraph.graph import StateGraph, END
from agents.schema_reasoner import reason_schema_node
from agents.query_planner import plan_query_node
from agents.sql_generator import generate_sql_node
from agents.qa_validator import qa_node
from state import AgentState

def create_bi_agent():
    workflow = StateGraph(AgentState)

    # 1. Reason over schema: Use MCP to find relevant tables
    workflow.add_node("schema_reasoning", reason_schema_node)
    
    # 2. Planning: Break question into logical steps (e.g., Join A to B, filter by C)
    workflow.add_node("query_planning", plan_query_node)
    
    # 3. Generation: Write the actual SQL
    workflow.add_node("sql_generation", generate_sql_node)
    
    # 4. QA & Validation: Execute against MCP and check for errors/empty sets
    workflow.add_node("qa_validation", qa_node)

    # Define the hierarchical flow
    workflow.set_entry_point("schema_reasoning")
    workflow.add_edge("schema_reasoning", "query_planning")
    workflow.add_edge("query_planning", "sql_generation")
    workflow.add_edge("sql_generation", "qa_validation")

    # The Logic Loop: If QA fails, go back to SQL generation or planning
    workflow.add_conditional_edges(
        "qa_validation",
        lambda state: "refined" if state["qa_feedback"] == "PASS" else "retry",
        {
            "refined": END,
            "retry": "sql_generation" # Loop back to fix the query
        }
    )

    return workflow.compile()