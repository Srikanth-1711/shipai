import asyncio
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from app.engine.orchestrator import stream_graph
from uuid import uuid4

async def main():
    session_id = str(uuid4())
    user_input = "I need a RAG system for 300 employees"
    print("Starting stream...")
    try:
        async for event in stream_graph(user_input, session_id):
            print(event.to_json())
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
