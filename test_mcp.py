import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp():
    server_params = StdioServerParameters(
        command=".venv/Scripts/python",
        args=["shipai/mcp_servers/github_server.py"],
    )
    print("Starting client...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Session initialized.")
            
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")
            
            print("Calling search_repos for 'langgraph'...")
            try:
                result = await session.call_tool("search_repos", {"query": "langgraph"})
                print(f"search_repos Result:\n{result.content}")
            except Exception as e:
                print(f"Error calling tool: {e}")

if __name__ == "__main__":
    asyncio.run(test_mcp())
