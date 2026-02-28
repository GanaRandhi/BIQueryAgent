# BI Query Agent

## 📂 Folder Structure - Gemini
```
bi_query_agent/
├── retrieval/
│   └── paged_rag.py         # Logic for "Page" indexing and retrieval
├── agents/
│   ├── schema_reasoner.py   # Uses RAISE logic to identify "Pages"
│   ├── sql_generator.py     # Incremental SQL construction node
│   └── prompts.py           # Centralized System Prompts (RAISE-aligned)
├── mcp_server/
│   └── database_mcp.py      # Extended to support "Page" fetching
├── state.py
└── graph.py
```
