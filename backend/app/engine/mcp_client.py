import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import tool

class MCPConnectionManager:
    """Manages persistent connections to local MCP servers for a LangGraph node."""
    
    def __init__(self, python_path: str = ".venv/Scripts/python"):
        self.python_path = python_path
        self._contexts = {}
        self._sessions = {}
        
    async def connect(self, server_name: str, script_path: str):
        """Connects to an MCP server and stores the session."""
        server_params = StdioServerParameters(
            command=self.python_path,
            args=[script_path],
        )
        # We manually manage the context to keep it open during the node's execution
        cm_stdio = stdio_client(server_params)
        read, write = await cm_stdio.__aenter__()
        
        cm_session = ClientSession(read, write)
        session = await cm_session.__aenter__()
        await session.initialize()
        
        self._contexts[server_name] = (cm_stdio, cm_session)
        self._sessions[server_name] = session
        return session
        
    async def disconnect_all(self):
        """Closes all active MCP sessions."""
        for name, (cm_stdio, cm_session) in self._contexts.items():
            try:
                await cm_session.__aexit__(None, None, None)
                await cm_stdio.__aexit__(None, None, None)
            except:
                pass
        self._contexts.clear()
        self._sessions.clear()
        
    def get_session(self, server_name: str) -> ClientSession:
        return self._sessions.get(server_name)

    async def create_langchain_tools(self, server_name: str):
        """Creates LangChain-compatible tools that route to the active MCP session."""
        session = self.get_session(server_name)
        if not session:
            raise ValueError(f"No active session for {server_name}")
            
        mcp_tools = await session.list_tools()
        lc_tools = []
        
        for t in mcp_tools.tools:
            # Create a closure to capture the session and tool name
            def make_func(t_name=t.name):
                async def mcp_tool_wrapper(**kwargs):
                    try:
                        result = await session.call_tool(t_name, kwargs)
                        return result.content
                    except Exception as e:
                        return f"Error executing {t_name}: {e}"
                mcp_tool_wrapper.__name__ = t_name
                mcp_tool_wrapper.__doc__ = t.description
                return mcp_tool_wrapper
                
            func = make_func()
            lc_tools.append(tool(func))
            
        return lc_tools

    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect_all()
        
    async def call_tool(self, server_name: str, tool_name: str, **kwargs):
        """Calls a tool directly on the specified server."""
        session = self.get_session(server_name)
        if not session:
            raise ValueError(f"No active session for {server_name}")
            
        result = await session.call_tool(tool_name, kwargs)
        if result and result.content:
            text = result.content[0].text
            try:
                return json.loads(text)
            except:
                return text
        return None
