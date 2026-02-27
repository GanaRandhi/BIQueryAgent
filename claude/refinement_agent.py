"""
agents/refinement_agent.py
────────────────────────────
Iterative Refinement Agent — Architecture Layer 4.

Responsibilities
----------------
• Execute partial/candidate SQL queries to inspect result shape.
• Analyse result shape (columns, types, row count, sample values).
• Modify filters, aggregations, or joins based on what the data shows.
• Repeat until the query stabilises or max iterations are reached.

This implements the "iterative refinement loop" from RAISE (arxiv 2506.01273)
§4.2 and AskDB (arxiv 2511.16131) §5 — both papers show that executing
intermediate queries and adjusting based on actual results dramatically
reduces hallucinations and wrong answers.

Stabilisation criterion
-----------------------
The loop stabilises when one of:
a) The LLM judges the result shape is correct (no_change flag).
b) Two consecutive iterations produce identical SQL.
c) max_refinement_iterations is reached.
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config.settings import get_settings
from database.connector import DatabaseConnector
from graph.state import BIQueryState
from mcp_server.client import MCPClient

logger = structlog.get_logger(__name__)


# ── output schema ─────────────────────────────────────────────────────────────

class RefinementDecision(BaseModel):
    """What the refinement agent decides to do after inspecting a result."""
    no_change: bool = Field(
        description="True if the current SQL is correct and doesn't need further refinement"
    )
    refined_sql: str | None = Field(
        default=None,
        description="The improved SQL when no_change=False"
    )
    change_summary: str = Field(
        description="What was changed and why (or why no change was needed)"
    )
    issues_found: list[str] = Field(
        default_factory=list,
        description="List of issues identified in the current result"
    )


# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a data quality expert reviewing SQL query results.
Your task is to examine the result of a SQL query and decide if it correctly
answers the user's business question, or if the query needs to be refined.

Common issues to look for:
- Wrong row count (too many = missing filter; too few = over-filtering)
- NULL values where there shouldn't be any
- Incorrect aggregation (SUM vs COUNT, wrong GROUP BY keys)
- Missing joins (values that should be joined are ID numbers instead)
- Date range issues (wrong period, missing date filter)
- Duplicates caused by unintended cartesian products

Output JSON matching the RefinementDecision schema.
"""

USER_PROMPT_TEMPLATE = """# Original User Question
{user_question}

# Schema Context
{schema_context}

# Current SQL
```sql
{candidate_sql}
```

# Execution Result
- Row count: {row_count}
- Columns: {columns}
- Sample rows (first {sample_count}):
{sample_rows}

# Previous Iteration Issues
{previous_issues}

# Task
Analyse whether this result correctly answers the user's question.
If yes, set no_change=true.
If no, provide refined_sql with the fix and explain what you changed.

Output JSON:
{{
  "no_change": bool,
  "refined_sql": string | null,
  "change_summary": string,
  "issues_found": [string]
}}
"""


# ── agent ─────────────────────────────────────────────────────────────────────

class RefinementAgent:
    """
    LangGraph node: refinement

    Input state fields consumed
    ---------------------------
    - candidate_sql, user_question, schema_context, session_id,
      refinement_iteration

    Output state fields written
    ---------------------------
    - candidate_sql          : Updated SQL (or same if no change)
    - refinement_iteration   : Incremented
    - last_execution_result  : Raw execution result dict
    - refinement_notes       : What changed and why
    """

    def __init__(
        self,
        connector: DatabaseConnector | None = None,
        mcp_client: MCPClient | None = None,
    ) -> None:
        settings = get_settings()
        self._llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.0,
            openai_api_key=settings.openai_api_key,
        ).bind(response_format={"type": "json_object"})
        self._connector = connector or DatabaseConnector()
        self._mcp = mcp_client or MCPClient()
        self._settings = settings
        self._parser = JsonOutputParser(pydantic_object=RefinementDecision)

    def __call__(self, state: BIQueryState) -> dict:
        session_id = state["session_id"]
        iteration = state.get("refinement_iteration", 0)
        candidate_sql = state.get("candidate_sql") or ""

        logger.info(
            "RefinementAgent: iteration",
            session_id=session_id,
            iteration=iteration,
        )

        # ── Step 1: Execute the candidate SQL ─────────────────────────────────
        exec_result = self._execute_safely(session_id, candidate_sql)

        # ── Step 2: Fetch previous issues from execution history ───────────────
        history = self._mcp.get_executions(session_id)
        previous_issues = self._summarise_history(history.get("executions", []))

        # ── Step 3: Ask LLM to evaluate result and refine ─────────────────────
        sample_rows_text = self._format_sample_rows(exec_result)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT_TEMPLATE.format(
                user_question=state["user_question"],
                schema_context=state.get("schema_context") or "Not available",
                candidate_sql=candidate_sql,
                row_count=exec_result.get("row_count", "error"),
                columns=exec_result.get("columns", []),
                sample_count=self._settings.max_preview_rows,
                sample_rows=sample_rows_text,
                previous_issues=previous_issues or "None",
            )),
        ]

        raw_response = self._llm.invoke(messages)
        decision_dict = self._parser.parse(raw_response.content)
        decision = RefinementDecision(**decision_dict) if isinstance(decision_dict, dict) else decision_dict

        # ── Step 4: Determine new SQL ─────────────────────────────────────────
        new_sql = candidate_sql if decision.no_change else (decision.refined_sql or candidate_sql)

        # ── Step 5: Persist refined plan to MCP ───────────────────────────────
        plan_record = {
            "sql": new_sql,
            "rationale": decision.change_summary,
            "assumptions": state.get("query_assumptions", []),
            "iteration": iteration + 1,
        }
        self._mcp.store_plan(session_id, plan_record)
        self._mcp.update_status(session_id, "refining")

        logger.info(
            "RefinementAgent: decision",
            no_change=decision.no_change,
            issues=len(decision.issues_found),
        )

        return {
            "candidate_sql": new_sql,
            "refinement_iteration": iteration + 1,
            "last_execution_result": exec_result,
            "refinement_notes": decision.change_summary,
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _execute_safely(self, session_id: str, sql: str) -> dict:
        """
        Execute SQL and return a dict with result metadata.
        Errors are captured (not raised) so the refinement loop can continue.
        """
        try:
            df = self._connector.execute_query(
                sql,
                max_rows=self._settings.max_preview_rows,
            )
            sample_rows = df.head(5).values.tolist()
            result = {
                "row_count": len(df),
                "columns": list(df.columns),
                "sample_rows": sample_rows,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            result = {
                "row_count": 0,
                "columns": [],
                "sample_rows": [],
                "error": str(exc),
            }

        # Log to MCP for audit trail and history inspection
        self._mcp.log_execution(
            session_id=session_id,
            sql=sql,
            row_count=result["row_count"],
            columns=result["columns"],
            sample_rows=result["sample_rows"],
            error=result.get("error"),
        )
        return result

    @staticmethod
    def _format_sample_rows(exec_result: dict) -> str:
        """Format sample rows for the LLM prompt."""
        if exec_result.get("error"):
            return f"ERROR: {exec_result['error']}"
        rows = exec_result.get("sample_rows", [])
        cols = exec_result.get("columns", [])
        if not rows:
            return "(no rows returned)"
        header = " | ".join(str(c) for c in cols)
        sep = "-" * len(header)
        data_lines = [" | ".join(str(v) for v in row) for row in rows]
        return "\n".join([header, sep] + data_lines)

    @staticmethod
    def _summarise_history(executions: list[dict]) -> str:
        """Produce a brief summary of previous execution errors/issues."""
        errors = [e["error"] for e in executions if e.get("error")]
        if not errors:
            return ""
        return "Previous errors:\n" + "\n".join(f"- {e}" for e in errors[-3:])
