"""
graph/supervisor.py
────────────────────
Hierarchical Supervisor — orchestrates the multi-agent BI workflow.

Architecture role
-----------------
The supervisor is the "brain" of the LangGraph workflow.  It doesn't do
SQL work itself; instead, it:

1. Decides which agent runs next based on the current state.
2. Detects when the refinement loop has stabilised or failed.
3. Routes clarification questions back to the planning stage.
4. Routes failed validations back to refinement.
5. Enforces global safety guards (max iterations, error escalation).

Hierarchical process
--------------------
The supervisor itself is a LangGraph node.  The full graph is:

  [supervisor] → [schema_ingestion] → [supervisor]
              → [schema_reasoning]  → [supervisor]
              → [query_planning]    → [supervisor]
              → [await_clarification] → [supervisor]
              → [refinement]        → [supervisor]
              → [qa_review]         → [supervisor]
              → [validation]        → END

This hierarchical pattern (all nodes route through a central supervisor)
allows the supervisor to inspect state after every step and re-route if
something unexpected happened.
"""

from __future__ import annotations

import structlog

from config.settings import get_settings
from graph.state import BIQueryState

logger = structlog.get_logger(__name__)

# ── routing constants ─────────────────────────────────────────────────────────
# These strings must match the node names registered in workflow.py

NODE_SCHEMA_INGESTION    = "schema_ingestion"
NODE_SCHEMA_REASONING    = "schema_reasoning"
NODE_QUERY_PLANNING      = "query_planning"
NODE_AWAIT_CLARIFICATION = "await_clarification"
NODE_REFINEMENT          = "refinement"
NODE_QA_REVIEW           = "qa_review"
NODE_VALIDATION          = "validation"
NODE_END                 = "__end__"


class SupervisorAgent:
    """
    LangGraph node: supervisor

    The supervisor examines the current BIQueryState and returns the name
    of the next node to execute.  It is a pure routing function — it does
    NOT modify the state.

    Routing logic (in priority order)
    ----------------------------------
    1. If there is an unrecoverable error → END
    2. If validation has passed → END
    3. If schema context is missing → schema_ingestion
    4. If reasoning notes are missing → schema_reasoning
    5. If candidate SQL is missing:
        a. If clarification is pending → await_clarification
        b. Else → query_planning
    6. If clarification is still pending (waiting for user) → await_clarification
    7. If within refinement budget → refinement
    8. If refinement budget exhausted → qa_review
    9. If QA passed → validation
    10. If QA failed (critical findings) → refinement (one more chance)
    11. Default → qa_review
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def __call__(self, state: BIQueryState) -> str:
        """
        Return the name of the next node.
        Called by LangGraph's conditional edge router.
        """
        next_node = self._decide(state)
        logger.info(
            "Supervisor: routing",
            session_id=state.get("session_id"),
            next=next_node,
            iteration=state.get("refinement_iteration", 0),
        )
        return next_node

    def _decide(self, state: BIQueryState) -> str:  # noqa: C901 (complexity ok here)
        # ── Guard: catastrophic error ─────────────────────────────────────────
        if state.get("error_message") and not state.get("validation_passed"):
            error = state["error_message"] or ""
            # If QA blocked + we've already done multiple refinements → give up
            if "Blocked by critical QA" in error and state.get("refinement_iteration", 0) >= 2:
                logger.error("Supervisor: unrecoverable error, ending workflow")
                return NODE_END

        # ── Terminal: workflow completed successfully ──────────────────────────
        if state.get("validation_passed"):
            return NODE_END

        # ── Stage 1: Schema not yet ingested ──────────────────────────────────
        if not state.get("schema_context"):
            return NODE_SCHEMA_INGESTION

        # ── Stage 2: Schema reasoning not done ───────────────────────────────
        if not state.get("reasoning_notes"):
            return NODE_SCHEMA_REASONING

        # ── Stage 3: No SQL yet → planning or clarification ──────────────────
        if not state.get("candidate_sql"):
            if state.get("pending_clarification"):
                return NODE_AWAIT_CLARIFICATION
            return NODE_QUERY_PLANNING

        # ── Stage 4: Clarification still pending ─────────────────────────────
        if state.get("pending_clarification"):
            return NODE_AWAIT_CLARIFICATION

        # ── Stage 5: Refinement budget check ─────────────────────────────────
        iteration = state.get("refinement_iteration", 0)
        max_iterations = self._settings.max_refinement_iterations

        if iteration < max_iterations:
            # Check if the last refinement iteration resulted in no change
            # (stabilised) by looking at execution history length
            exec_count = self._count_executions(state)
            if exec_count > 0 and iteration > 0:
                # We have at least one execution; check if we should go to QA
                last_result = state.get("last_execution_result") or {}
                if not last_result.get("error") and last_result.get("row_count", 0) > 0:
                    # Results look reasonable — proceed to QA
                    if not self._qa_needed(state):
                        return NODE_QA_REVIEW
            return NODE_REFINEMENT

        # ── Stage 6: Budget exhausted → QA ───────────────────────────────────
        if not self._qa_completed(state):
            return NODE_QA_REVIEW

        # ── Stage 7: QA has findings — check severity ─────────────────────────
        if self._has_critical_findings(state):
            if iteration < max_iterations + 2:
                # Give the refinement agent one more attempt
                return NODE_REFINEMENT
            return NODE_END

        # ── Stage 8: QA passed → validate and execute ────────────────────────
        return NODE_VALIDATION

    # ── helper predicates ─────────────────────────────────────────────────────

    @staticmethod
    def _count_executions(state: BIQueryState) -> int:
        """Number of intermediate SQL executions so far."""
        # We infer this from the refinement_iteration counter
        return state.get("refinement_iteration", 0)

    @staticmethod
    def _qa_needed(state: BIQueryState) -> bool:
        """True if QA has not yet run this iteration."""
        # QA has run if qa_findings is non-empty OR validation was attempted
        return not state.get("qa_findings") and not state.get("validation_passed")

    @staticmethod
    def _qa_completed(state: BIQueryState) -> bool:
        """True if QA has produced at least one finding or passed cleanly."""
        # If qa_findings is populated (even empty list from QA run), QA ran.
        # We use validation_passed as proxy: if it's False but was attempted,
        # an error_message will be set.
        return bool(state.get("qa_findings")) or state.get("validation_passed", False)

    @staticmethod
    def _has_critical_findings(state: BIQueryState) -> bool:
        """True if any QA finding is critical or error level."""
        return any(
            f.get("severity") in ("critical", "error")
            for f in state.get("qa_findings", [])
        )
