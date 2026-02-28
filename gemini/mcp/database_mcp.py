from mcp.server.fastmcp import FastMCP

mcp = FastMCP("BI_Data_Engine")

@mcp.tool()
def fetch_schema_pages(table_names: List[str]) -> str:
    """Returns detailed DDL and sample rows for specific tables."""
    # In a real scenario, this queries the DB metadata
    pages = []
    for table in table_names:
        pages.append(f"PAGE: {table}\nDDL: CREATE TABLE {table} (...)\nSAMPLES: [...]")
    return "\n---\n".join(pages)

@mcp.tool()
def execute_and_validate(sql: str) -> dict:
    """Executes SQL and returns results or structured error for RAISE refinement."""
    # Logic to catch empty sets or syntax errors
    return {"status": "success", "data": [], "row_count": 0}