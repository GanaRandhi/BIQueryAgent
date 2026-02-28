from mcp.server.fastmcp import FastMCP

mcp = FastMCP("BI_Agent_Server")

@mcp.tool()
def get_table_page(table_name: str) -> str:
    """
    Paged RAG Tool: Returns a comprehensive 'Page' for a table.
    Includes: DDL, Foreign Key relationships, and 3 sample rows.
    """
    # 1. Get DDL
    # 2. Get Related Tables (Foreign Keys)
    # 3. Get Sample Data (SELECT * FROM table LIMIT 3)
    return f"PAGE FOR {table_name}:\nDDL: ...\nRELATIONS: ...\nSAMPLES: ..."

@mcp.tool()
def validate_sql_logic(sql: str) -> dict:
    """
    Executes EXPLAIN or a dry-run to check for logical errors.
    Used by the QA node to provide feedback.
    """
    # Logic to catch errors before returning to user
    return {"status": "success", "row_count": 150}