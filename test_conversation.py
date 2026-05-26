import asyncio
import sys
import os

# Add backend directory to sys.path so we can import app
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from app.engine.orchestrator import stream_graph

async def run_dialogue():
    session_id = "demo_session_999"
    
    # Dialogue steps
    turns = [
        "i want a agentic rag project for my team !",
        "our data is pure logs very huge data with 3 lakhs of lines",
        "every day regression will be start running while we sleep and mrng mail comes to each person who starts regression in that mail using regex they parse fail and errors only in gmail, but using AI we need RCA from that log file while huge logs data already there in auto nfs path with read only access",
        "every day data of logs runs and get stored in autopaths around 3lakhs of lines ~ 200+ engineers will use it logs come from router cisco router internal electonic parts like cpu npu etc diags testign kinda logs they maintain",
        "24 unique platforms. Around 100 different logs for one day, they maintain 3 months data only and each log 4mb to 100mb max",
        "unstructured log only stored in nfs mount auto paths"
    ]
    
    for idx, turn in enumerate(turns):
        print(f"\n\n--- USER TURN {idx+1}: {turn} ---")
        sys.stdout.flush()
        
        last_event = None
        async for event in stream_graph(turn, session_id):
            if event.type == "token":
                print(event.payload["text"], end="")
                sys.stdout.flush()
            elif event.type in ("requirement", "decision", "complete"):
                print(f"\n[EVENT] {event.type.value} -> {event.payload}")
                sys.stdout.flush()
                last_event = event
                
        # If it reached complete with a project_path, we are done!
        if last_event and last_event.type == "complete" and last_event.payload.get("project_path"):
            print("\n\n🎉 SUCCESS: Architecture generated and codebase built!")
            sys.stdout.flush()
            break

if __name__ == "__main__":
    asyncio.run(run_dialogue())
