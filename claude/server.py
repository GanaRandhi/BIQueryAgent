"""
mcp_server/server.py
─────────────────────
FastMCP Server — the "shared memory bus" for all BI Query Agent workflows.

Architecture role
-----------------
In a multi-agent LangGraph system, agents need a neutral place to:
  • Persist and retrieve schema maps
  • Store partial SQL query plans
  • Log refinement iterations (audit trail)
  • Share QA findings between the QA agent and the planning agent
  • Cache expensive LLM outputs for reuse

FastMCP gives us typed tools (Python functions decorated with @mcp.tool)
that are auto-exposed as MCP protocol endpoints. LangGraph agents call these
tools via the MCP client adapter rather than communicating directly, which
keeps agents decoupled and the workflow observable.

Storage strategy
----------------
We use an in-process dict (MemoryStore) for simplicity.  In production you
would swap this for Redis, DynamoDB, or Postgres with minimal changes because
all persistence goes through the tool functions below — no agent touches
storage directly.

Run this server:
  python -m mcp_server.server          (stdio transport — default)
  MCP_TRANSPORT=sse python -m mcp_server.server   (SSE/HTTP transport)
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastmcp import FastMCP

# ── in-memory store (swap for Redis etc. in production) ───────────────────────

class MemoryStore:
    """
    Thread-safe key/value + list store shared across all MCP tool calls.
    In production replace the dicts with Redis pipelines.
    """

    def __init__(self) -> None:
        self._kv: dict[str, Any] = {}          # key → any JSON-serialisable value
        self._lists: dict[str, list[Any]] = {}  # key → append-only list

    # ── key-value ─────────────────────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        self._kv[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._kv.get(key, default)

    def delete(self, key: str) -> bool:
        return self._kv.pop(key, None) is not None

    def keys(self, prefix: str = "") -> list[str]:
        return [k for k in self._kv if k.startswith(prefix)]

    # ── append-only lists ─────────────────────────────────────────────────────

    def append(self, key: str, value: Any) -> int:
        if key not in self._lists:
            self._lists[key] = []
        self._lists[key].append(value)
        return len(self._lists[key])

    def list_get(self, key: str) -> list[Any]:
        return list(self._lists.get(key, []))


_store = MemoryStore()

# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="bi-query-agent-server",
    instructions=(
        "MCP server for the BI Query Agent. "
        "Provides tools for schema storage, query plan management, "
        "refinement history, and QA findings shared between agents."
    ),
)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL GROUP 1 — Schema Storage
# Agents: SchemaIngestionAgent, SchemaReasoningAgent, QueryPlanningAgent
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def store_schema_map(db_name: str, schema_json: str) -> dict:
    """
    Persist a JSON-serialised SchemaMap produced by the SchemaExtractor.

    Called by: SchemaIngestionAgent after extraction completes.
    Read by  : SchemaReasoningAgent, QueryPlanningAgent, ValidationAgent.

    Parameters
    ----------
    db_name     : Logical database identifier (used as storage key).
    schema_json : JSON string of SchemaMap.to_dict().
    """
    key = f"schema::{db_name}"
    _store.set(key, json.loads(schema_json))
    _store.set(f"schema_indexed_at::{db_name}", time.time())
    return {"status": "stored", "db_name": db_name, "key": key}


@mcp.tool()
def get_schema_map(db_name: str) -> dict:
    """
    Retrieve the previously stored SchemaMap for a database.

    Returns {"found": False} if no schema has been ingested yet.
    """
    key = f"schema::{db_name}"
    data = _store.get(key)
    if data is None:
        return {"found": False, "db_name": db_name}
    return {"found": True, "db_name": db_name, "schema": data}


@mcp.tool()
def store_reasoning_notes(session_id: str, notes: str) -> dict:
    """
    Store the SchemaReasoningAgent's analysis of entity relationships,
    valid join paths, and entity-to-column mappings.

    Called by: SchemaReasoningAgent.
    Read by  : QueryPlanningAgent (to avoid invalid joins).
    """
    key = f"reasoning::{session_id}"
    _store.set(key, {"notes": notes, "created_at": time.time()})
    return {"status": "stored", "session_id": session_id}


@mcp.tool()
def get_reasoning_notes(session_id: str) -> dict:
    """Retrieve schema reasoning notes for a session."""
    key = f"reasoning::{session_id}"
    data = _store.get(key)
    if data is None:
        return {"found": False, "session_id": session_id}
    return {"found": True, "session_id": session_id, **data}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL GROUP 2 — Query Plan Management
# Agents: QueryPlanningAgent, RefinementAgent
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def store_query_plan(session_id: str, plan: dict) -> dict:
    """
    Persist the current query plan (candidate SQL + rationale).

    `plan` structure expected:
    {
        "sql"        : "SELECT ...",
        "rationale"  : "Explanation of chosen joins and filters",
        "assumptions": ["assumption 1", "assumption 2"],
        "iteration"  : 0
    }

    Called by: QueryPlanningAgent and RefinementAgent on each iteration.
    """
    key = f"query_plan::{session_id}"
    plan["stored_at"] = time.time()
    _store.set(key, plan)

    # Also append to the iteration history
    _store.append(f"query_plan_history::{session_id}", plan)

    return {"status": "stored", "session_id": session_id, "iteration": plan.get("iteration")}


@mcp.tool()
def get_query_plan(session_id: str) -> dict:
    """Retrieve the most recent query plan for a session."""
    key = f"query_plan::{session_id}"
    data = _store.get(key)
    if data is None:
        return {"found": False, "session_id": session_id}
    return {"found": True, "session_id": session_id, "plan": data}


@mcp.tool()
def get_query_plan_history(session_id: str) -> dict:
    """
    Return all previous query plan iterations for a session.
    Used by the RefinementAgent to avoid revisiting failed approaches.
    """
    key = f"query_plan_history::{session_id}"
    history = _store.list_get(key)
    return {"session_id": session_id, "history": history, "count": len(history)}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL GROUP 3 — Refinement & Execution Tracking
# Agents: RefinementAgent
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def log_execution_result(
    session_id: str,
    sql: str,
    row_count: int,
    columns: list[str],
    sample_rows: list[list],
    error: str | None = None,
) -> dict:
    """
    Record the result of executing an intermediate SQL query.

    The RefinementAgent inspects these results to decide whether to:
    - Adjust filters (result has too many / too few rows)
    - Fix aggregations (wrong grouping keys)
    - Ask the user a clarification question

    Parameters
    ----------
    session_id  : Unique session identifier.
    sql         : The SQL that was executed.
    row_count   : Number of rows returned.
    columns     : Column names in the result.
    sample_rows : First N rows as nested lists.
    error       : Error message if execution failed, else None.
    """
    entry = {
        "sql": sql,
        "row_count": row_count,
        "columns": columns,
        "sample_rows": sample_rows,
        "error": error,
        "timestamp": time.time(),
    }
    idx = _store.append(f"executions::{session_id}", entry)
    return {"status": "logged", "execution_index": idx - 1}


@mcp.tool()
def get_execution_history(session_id: str) -> dict:
    """
    Return all execution results for a session.
    Allows the RefinementAgent to see the full iteration trail.
    """
    history = _store.list_get(f"executions::{session_id}")
    return {"session_id": session_id, "executions": history, "count": len(history)}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL GROUP 4 — Clarification Dialogue
# Agents: QueryPlanningAgent (asks), user-facing layer (answers)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def store_clarification(
    session_id: str,
    question: str,
    answer: str | None = None,
) -> dict:
    """
    Record a clarification question posed to the user, and optionally
    its answer.  Answers are filled in by the orchestration layer when
    the user responds.

    Called by: QueryPlanningAgent when intent is ambiguous.
    """
    key = f"clarification::{session_id}"
    existing = _store.get(key) or []
    existing.append({
        "question": question,
        "answer": answer,
        "timestamp": time.time(),
    })
    _store.set(key, existing)
    return {"status": "stored", "total_clarifications": len(existing)}


@mcp.tool()
def get_clarifications(session_id: str) -> dict:
    """Retrieve all clarification Q&A pairs for a session."""
    key = f"clarification::{session_id}"
    data = _store.get(key) or []
    return {"session_id": session_id, "clarifications": data}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL GROUP 5 — QA Findings
# Agents: QAAgent writes, ValidationAgent and supervisor reads
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def store_qa_finding(
    session_id: str,
    finding_type: str,
    severity: str,
    description: str,
    suggested_fix: str | None = None,
) -> dict:
    """
    Record a quality-assurance finding from the QAAgent.

    Parameters
    ----------
    session_id    : Session identifier.
    finding_type  : Category e.g. "sql_syntax", "logic_error", "performance".
    severity      : "info" | "warning" | "error" | "critical".
    description   : Human-readable description of the issue.
    suggested_fix : Optional SQL or strategy to fix the issue.

    Called by: QAAgent during every validation pass.
    Read by  : ValidationAgent (blocks execution on critical findings),
               Supervisor (decides whether to re-route to refinement).
    """
    entry = {
        "finding_type": finding_type,
        "severity": severity,
        "description": description,
        "suggested_fix": suggested_fix,
        "timestamp": time.time(),
    }
    idx = _store.append(f"qa_findings::{session_id}", entry)
    return {"status": "stored", "finding_index": idx - 1}


@mcp.tool()
def get_qa_findings(session_id: str) -> dict:
    """Retrieve all QA findings for a session."""
    findings = _store.list_get(f"qa_findings::{session_id}")
    critical = [f for f in findings if f["severity"] == "critical"]
    return {
        "session_id": session_id,
        "findings": findings,
        "total": len(findings),
        "critical_count": len(critical),
        "has_blockers": len(critical) > 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL GROUP 6 — Final Answer
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def store_final_answer(
    session_id: str,
    sql: str,
    explanation: str,
    result_summary: str,
) -> dict:
    """
    Persist the validated, final SQL and its plain-English explanation.

    Called by: ValidationAgent after all checks pass.
    Read by  : Application / user-facing layer to display results.
    """
    key = f"final_answer::{session_id}"
    _store.set(key, {
        "sql": sql,
        "explanation": explanation,
        "result_summary": result_summary,
        "completed_at": time.time(),
    })
    return {"status": "stored", "session_id": session_id}


@mcp.tool()
def get_final_answer(session_id: str) -> dict:
    """Retrieve the final answer for a completed session."""
    key = f"final_answer::{session_id}"
    data = _store.get(key)
    if data is None:
        return {"found": False, "session_id": session_id}
    return {"found": True, "session_id": session_id, **data}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL GROUP 7 — Session Lifecycle
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def create_session(user_question: str, db_name: str) -> dict:
    """
    Create a new query session and return a unique session_id.

    Called by: main entrypoint / supervisor before starting the workflow.
    """
    import uuid
    session_id = str(uuid.uuid4())
    _store.set(f"session::{session_id}", {
        "user_question": user_question,
        "db_name": db_name,
        "status": "created",
        "created_at": time.time(),
    })
    return {"session_id": session_id, "status": "created"}


@mcp.tool()
def update_session_status(session_id: str, status: str) -> dict:
    """
    Update the lifecycle status of a session.
    Valid statuses: created → schema_ready → planning → refining → validating → done | failed
    """
    key = f"session::{session_id}"
    data = _store.get(key)
    if data is None:
        return {"error": f"Session {session_id!r} not found"}
    data["status"] = status
    data["updated_at"] = time.time()
    _store.set(key, data)
    return {"session_id": session_id, "status": status}


@mcp.tool()
def get_session(session_id: str) -> dict:
    """Retrieve full session metadata."""
    key = f"session::{session_id}"
    data = _store.get(key)
    if data is None:
        return {"found": False, "session_id": session_id}
    return {"found": True, "session_id": session_id, **data}


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config.settings import get_settings
    settings = get_settings()

    if settings.mcp_transport == "sse":
        # HTTP/SSE mode — exposes the server as an HTTP endpoint
        mcp.run(transport="sse", host=settings.mcp_host, port=settings.mcp_port)
    else:
        # stdio mode — default for local LangGraph integration
        mcp.run(transport="stdio")
