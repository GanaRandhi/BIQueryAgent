"""
graph/workflow.py
──────────────────
LangGraph workflow assembly — wires all agents into a directed graph.

Graph topology
--------------
All nodes route through the central supervisor (hierarchical pattern).
The supervisor's conditional edge selects the next node based on state.

    ┌─────────────┐
    │  __start__  │
    └──────┬──────┘
           ▼
    ┌─────────────┐     conditional edge (supervisor)
    │  supervisor  │────────────────────────────────────────┐
    └─────────────┘                                         │
           │ routes to one of:                              │
           ├──► schema_ingestion ──────────────────────────►│
           ├──► schema_reasoning ──────────────────────────►│
           ├──► query_planning   ──────────────────────────►│
           ├──► await_clarification ────────────────────────►│
           ├──► refinement       ──────────────────────────►│
           ├──► qa_review        ──────────────────────────►│
           ├──► validation       ──────────────────────────►│
           └──► __end__

Key LangGraph concepts used
---------------------------
• StateGraph          : Graph whose edges carry a shared typed state dict
• add_node()          : Registers a callable as a graph node
• add_conditional_edges(): Routes based on a function that returns a node name
• add_edge()          : Unconditional edge from node → supervisor
• MemorySaver         : In-process checkpointer for state persistence
• stream()            : Iterate node outputs for real-time display
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents import (
    QAAgent,
    QueryPlanningAgent,
    RefinementAgent,
    SchemaIngestionAgent,
    SchemaReasoningAgent,
    ValidationAgent,
)
from config.settings import get_settings
from database.connector import DatabaseConnector
from graph.state import BIQueryState
from graph.supervisor import (
    NODE_AWAIT_CLARIFICATION,
    NODE_END,
    NODE_QA_REVIEW,
    NODE_QUERY_PLANNING,
    NODE_REFINEMENT,
    NODE_SCHEMA_INGESTION,
    NODE_SCHEMA_REASONING,
    NODE_VALIDATION,
    SupervisorAgent,
)
from mcp_server.client import MCPClient


# ── clarification node ────────────────────────────────────────────────────────

def await_clarification_node(state: BIQueryState) -> dict:
    """
    Pause node that waits for the user to answer a clarification question.

    In a real application this would suspend and resume via a websocket or
    webhook.  Here we read the answer from the MCP server (where the
    application layer would have written it) or prompt via CLI.

    Returns a state update that clears pending_clarification so the
    supervisor stops routing here once the answer arrives.
    """
    question = state.get("pending_clarification") or "Please clarify your question."

    # ── In CLI/demo mode: prompt the user directly ────────────────────────────
    print(f"\n🤔 Clarification needed:\n   {question}")
    user_answer = input("   Your answer: ").strip()

    # ── Store the answer in MCP so the planning agent can read it ─────────────
    mcp = MCPClient()
    mcp.add_clarification(
        session_id=state["session_id"],
        question=question,
        answer=user_answer,
    )

    from langchain_core.messages import HumanMessage
    return {
        "pending_clarification": None,      # Clear the pending question
        "candidate_sql": None,              # Force re-planning with the answer
        "messages": [HumanMessage(content=user_answer)],
    }


# ── workflow builder ──────────────────────────────────────────────────────────

def build_workflow(
    connector: DatabaseConnector | None = None,
    mcp_client: MCPClient | None = None,
) -> StateGraph:
    """
    Construct and compile the BI Query Agent LangGraph workflow.

    Parameters
    ----------
    connector  : Optional pre-built DatabaseConnector (useful for tests).
    mcp_client : Optional pre-built MCPClient (useful for tests).

    Returns
    -------
    Compiled LangGraph CompiledGraph ready for invoke() / stream().
    """
    db_conn = connector or DatabaseConnector()
    mcp = mcp_client or MCPClient()

    # ── Instantiate all agents ────────────────────────────────────────────────
    supervisor          = SupervisorAgent()
    schema_ingestion    = SchemaIngestionAgent(connector=db_conn, mcp_client=mcp)
    schema_reasoning    = SchemaReasoningAgent(mcp_client=mcp)
    query_planning      = QueryPlanningAgent(mcp_client=mcp)
    refinement          = RefinementAgent(connector=db_conn, mcp_client=mcp)
    qa_agent            = QAAgent(mcp_client=mcp)
    validation          = ValidationAgent(connector=db_conn, mcp_client=mcp)

    # ── Build the StateGraph ──────────────────────────────────────────────────
    graph = StateGraph(BIQueryState)

    # Register nodes: name → callable
    # Each callable receives the full state dict and returns a partial update.
    graph.add_node("supervisor",            supervisor)
    graph.add_node(NODE_SCHEMA_INGESTION,   schema_ingestion)
    graph.add_node(NODE_SCHEMA_REASONING,   schema_reasoning)
    graph.add_node(NODE_QUERY_PLANNING,     query_planning)
    graph.add_node(NODE_AWAIT_CLARIFICATION, await_clarification_node)
    graph.add_node(NODE_REFINEMENT,         refinement)
    graph.add_node(NODE_QA_REVIEW,          qa_agent)
    graph.add_node(NODE_VALIDATION,         validation)

    # ── Entry point: START → supervisor ──────────────────────────────────────
    graph.add_edge(START, "supervisor")

    # ── Supervisor conditional edge ───────────────────────────────────────────
    # The supervisor function returns a string (node name or END).
    graph.add_conditional_edges(
        "supervisor",
        supervisor,                     # routing function
        {                               # map: return value → node name
            NODE_SCHEMA_INGESTION:    NODE_SCHEMA_INGESTION,
            NODE_SCHEMA_REASONING:    NODE_SCHEMA_REASONING,
            NODE_QUERY_PLANNING:      NODE_QUERY_PLANNING,
            NODE_AWAIT_CLARIFICATION: NODE_AWAIT_CLARIFICATION,
            NODE_REFINEMENT:          NODE_REFINEMENT,
            NODE_QA_REVIEW:           NODE_QA_REVIEW,
            NODE_VALIDATION:          NODE_VALIDATION,
            NODE_END:                 END,
        },
    )

    # ── All worker nodes route back to supervisor after completion ────────────
    for node_name in [
        NODE_SCHEMA_INGESTION,
        NODE_SCHEMA_REASONING,
        NODE_QUERY_PLANNING,
        NODE_AWAIT_CLARIFICATION,
        NODE_REFINEMENT,
        NODE_QA_REVIEW,
        NODE_VALIDATION,
    ]:
        graph.add_edge(node_name, "supervisor")

    # ── Compile with in-memory checkpointer ──────────────────────────────────
    # The MemorySaver enables state persistence and resumption across
    # clarification pauses.
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled


def run_workflow(
    user_question: str,
    db_name: str,
    session_id: str,
    connector: DatabaseConnector | None = None,
    mcp_client: MCPClient | None = None,
    stream_output: bool = True,
) -> BIQueryState:
    """
    High-level runner: build workflow, initialise state, invoke graph.

    Parameters
    ----------
    user_question : Natural-language question from the business user.
    db_name       : Target database identifier.
    session_id    : Unique session ID (create via mcp_client.create_session()).
    connector     : Optional pre-built database connector.
    mcp_client    : Optional pre-built MCP client.
    stream_output : If True, print each node's output as it runs.

    Returns
    -------
    Final BIQueryState after the workflow completes.
    """
    from graph.state import initial_state
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    workflow = build_workflow(connector=connector, mcp_client=mcp_client)

    state = initial_state(
        user_question=user_question,
        db_name=db_name,
        session_id=session_id,
    )

    config = {"configurable": {"thread_id": session_id}}

    if stream_output:
        console.print(Panel(
            f"[bold cyan]BI Query Agent[/bold cyan]\n"
            f"Question: [italic]{user_question}[/italic]",
            title="Starting Workflow",
        ))

        final_state = state
        for step in workflow.stream(state, config=config):
            for node_name, node_output in step.items():
                if stream_output and node_name != "supervisor":
                    console.print(f"[green]✓ {node_name}[/green]", end=" ")
                    if isinstance(node_output, dict):
                        keys = [k for k in node_output if node_output[k] is not None]
                        console.print(f"→ updated: {keys}")
                # Merge step output into final_state for return
                if isinstance(node_output, dict):
                    final_state = {**final_state, **node_output}
        return final_state
    else:
        return workflow.invoke(state, config=config)
