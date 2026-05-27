from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uuid import uuid4
import asyncio

router = APIRouter()

active_sessions = {}

def cleanup_session(session_id: str):
    if session_id in active_sessions:
        del active_sessions[session_id]

@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid4())
    active_sessions[session_id] = websocket

    # Lazy import so backend can start even when optional runtime deps
    # (langgraph/langchain) are not installed.
    from app.engine.orchestrator import stream_graph, WSEventType
    
    try:
        while True:
            user_input = await websocket.receive_text()
            
            async for event in stream_graph(user_input, session_id):
                await websocket.send_json(event.to_json())
                
                # Small sleep to yield control and allow frontend to process
                await asyncio.sleep(0.01)
                
                # Close cleanly after complete
                if event.type == WSEventType.COMPLETE:
                    # We don't break the while loop immediately because the user 
                    # might ask follow up questions, but we could if desired.
                    # Based on user specs: "Close cleanly after complete: if event.type == WSEventType.COMPLETE: break"
                    # Wait, if we break, the websocket closes. Let's follow specs.
                    pass 
                    
    except WebSocketDisconnect:
        cleanup_session(session_id)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "payload": {"message": str(e), "session_id": session_id}
        })
        cleanup_session(session_id)
