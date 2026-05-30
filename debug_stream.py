import asyncio
import sys
import os

# Add backend directory to sys.path so we can import app
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from app.engine.orchestrator import stream_graph

async def main():
    print("Starting stream_graph test...")
    try:
        async for event in stream_graph("i want a agentic rag project for my team !", "test_session_123"):
            print(f"EVENT: {event.type.value} -> {event.payload}")
            sys.stdout.flush()
        print("stream_graph finished successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
