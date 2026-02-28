# BI Query Agent

## 📂 Folder Structure - Gemini

```
bi_query_agent/
├── mcp/
│   └── database_server.py    # FastMCP Server: DB access & Schema Paging
├── retrieval/
│   └── attention_rag.py     # Paged Attention: Memory & Block Management
├── agents/
│   ├── prompts.py           # RAISE-aligned System Prompts
│   ├── nodes.py             # LangGraph Node implementations
│   └── planner.py           # Logical Execution Plan (LEP) logic
├── state.py                 # LangGraph State & Type definitions
├── graph.py                 # The State Machine (The "Brain")
└── main.py                  # Entry Point
```
