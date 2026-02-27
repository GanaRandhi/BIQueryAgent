# 🧠 BI Query Agent

> An AI-powered Business Intelligence agent that reasons over database schemas, iterates on intent through dialogue, and generates SQL incrementally — treating querying as a conversation, not a translation task.

Built with **LangGraph**, **LangChain**, **FastMCP**, and **ChromaDB**.
Inspired by [RAISE](https://arxiv.org/abs/2506.01273) and [AskDB](https://arxiv.org/abs/2511.16131).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LangGraph Workflow                           │
│                                                                     │
│   ┌──────────────┐                                                  │
│   │  Supervisor  │◄──────── routes all agents (hierarchical)       │
│   └──────┬───────┘                                                  │
│          │                                                           │
│    ┌─────▼──────────┐    ┌─────────────────┐    ┌───────────────┐  │
│    │ Schema         │    │ Schema          │    │ Query         │  │
│    │ Ingestion      │───►│ Reasoning       │───►│ Planning      │  │
│    │ Agent          │    │ Agent           │    │ Agent         │  │
│    └────────────────┘    └─────────────────┘    └───────┬───────┘  │
│                                                          │          │
│    ┌─────────────────┐    ┌──────────────┐    ┌────────▼────────┐  │
│    │ Validation &    │◄───│ QA           │◄───│ Refinement     │  │
│    │ Execution       │    │ Agent        │    │ Agent (loop)   │  │
│    │ Agent           │    └──────────────┘    └────────────────┘  │
│    └────────┬────────┘                                             │
│             │                                                       │
│         Final Answer                                                │
└─────────────┼───────────────────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │   FastMCP Server    │  ← shared memory bus for all agents
    │                     │
    │  • Schema store     │
    │  • Query plans      │
    │  • Execution logs   │
    │  • QA findings      │
    │  • Session state    │
    └─────────────────────┘
```

### Five Architecture Layers

| Layer | Agent | Role |
|-------|-------|------|
| 1 | **SchemaIngestionAgent** | Extract tables, columns, FKs, sample values from DB |
| 2 | **SchemaReasoningAgent** | Build entity relationship model, identify valid joins |
| 3 | **QueryPlanningAgent** | Translate questions → SQL plans; ask clarifications |
| 4 | **RefinementAgent** | Execute partial queries, inspect results, adjust SQL |
| 5 | **ValidationAgent** | Gate-keep QA findings, execute final SQL, explain results |

### Cross-cutting concerns

- **QAAgent**: Reviews SQL for anti-patterns, security issues, and logical errors
- **SupervisorAgent**: Hierarchical orchestrator; routes between all agents
- **MCPClient / FastMCP Server**: Shared state bus (schema, plans, findings, sessions)

---

## Project Structure

```
bi_query_agent/
├── main.py                          # CLI entrypoint (typer)
├── requirements.txt
├── pyproject.toml
├── .env.example
│
├── config/
│   ├── __init__.py
│   └── settings.py                  # Pydantic-settings configuration
│
├── database/
│   ├── __init__.py
│   ├── connector.py                 # SQLAlchemy engine + safe query execution
│   ├── schema_extractor.py          # Schema ingestion: tables, columns, FKs, samples
│   └── seed.py                      # Sample e-commerce SQLite database
│
├── retrieval/
│   ├── __init__.py
│   └── schema_retriever.py          # ChromaDB indexer + semantic schema retrieval
│
├── mcp_server/
│   ├── __init__.py
│   ├── server.py                    # FastMCP server with 18 typed tools
│   └── client.py                    # MCPClient facade used by all agents
│
├── agents/
│   ├── __init__.py
│   ├── schema_ingestion_agent.py    # Layer 1: DB → SchemaMap → MCP + ChromaDB
│   ├── schema_reasoning_agent.py    # Layer 2: schema → entity model
│   ├── query_planning_agent.py      # Layer 3: question → SQL plan + clarifications
│   ├── refinement_agent.py          # Layer 4: iterative SQL refinement loop
│   ├── qa_agent.py                  # QA: SQL review + auto-correction
│   └── validation_agent.py          # Layer 5: final execution + explanation
│
├── graph/
│   ├── __init__.py
│   ├── state.py                     # BIQueryState TypedDict shared across all nodes
│   ├── supervisor.py                # Hierarchical routing supervisor
│   └── workflow.py                  # LangGraph StateGraph assembly
│
└── tests/
    ├── __init__.py
    └── test_integration.py          # Integration tests (DB + MCP + agents)
```

---

## Quick Start

### 1. Install dependencies

```bash
cd bi_query_agent
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env:
#   OPENAI_API_KEY=sk-...
#   DB_URL=sqlite:///./sample.db      # or your PostgreSQL/MySQL URL
```

### 3. Seed the sample database

```bash
python main.py seed
```

### 4. Ingest & index the schema

```bash
python main.py ingest
```

### 5. Ask a question!

```bash
python main.py query "What are the top 10 customers by total order value this year?"
```

### Run the demo

```bash
python main.py demo
```

---

## How it works

### Dialogue mode (clarification)

When the user's question is ambiguous, the agent asks focused clarification questions:

```
User:  "Show me sales by region"
Agent: "Which time period would you like — last month, last quarter, or YTD?
        Also, which column defines 'region' — customers.country or a separate table?"
User:  "Last quarter, use customers.country"
Agent: [generates correct SQL with date filter and country grouping]
```

### Iterative refinement

The agent doesn't generate SQL in one shot. It:
1. Writes a candidate query
2. Executes it to inspect the result shape
3. Adjusts if: too many rows, NULLs where unexpected, wrong aggregation, etc.
4. Repeats until stable (default: up to 5 iterations)

### Schema retrieval

With hundreds of tables, injecting the full schema into every prompt would exceed
context limits. Instead, the agent uses ChromaDB semantic search to retrieve only
the 5–6 most relevant table descriptions per question.

---

## MCP Server Tools

The FastMCP server exposes **18 typed tools** grouped by function:

| Group | Tools |
|-------|-------|
| Schema | `store_schema_map`, `get_schema_map`, `store_reasoning_notes`, `get_reasoning_notes` |
| Query Plans | `store_query_plan`, `get_query_plan`, `get_query_plan_history` |
| Executions | `log_execution_result`, `get_execution_history` |
| Clarification | `store_clarification`, `get_clarifications` |
| QA | `store_qa_finding`, `get_qa_findings` |
| Final Answer | `store_final_answer`, `get_final_answer` |
| Sessions | `create_session`, `update_session_status`, `get_session` |

To run the MCP server standalone (SSE transport for remote agents):

```bash
MCP_TRANSPORT=sse python main.py serve
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | LLM API key |
| `OPENAI_MODEL` | `gpt-4o` | Model to use |
| `DB_URL` | `sqlite:///./sample.db` | Database connection URL |
| `MAX_REFINEMENT_ITERATIONS` | `5` | Max SQL refinement loops |
| `MAX_CLARIFICATION_ROUNDS` | `3` | Max clarification questions |
| `SQL_EXECUTION_TIMEOUT` | `30` | Seconds before SQL is killed |
| `MAX_PREVIEW_ROWS` | `20` | Rows returned in preview queries |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `sse` |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use a real SQLite database but mock all LLM calls for deterministic, fast execution.

---

## References

- [RAISE: Reasoning-Augmented Iterative SQL Engine](https://arxiv.org/abs/2506.01273)
- [AskDB: Schema-Aware Dialogue for Database Querying](https://arxiv.org/abs/2511.16131)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FastMCP](https://github.com/jlowin/fastmcp)
