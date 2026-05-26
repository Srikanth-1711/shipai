import asyncio
import websockets
import json

async def test():
    try:
        async with websockets.connect('ws://localhost:8000/ws/chat') as ws:
            await ws.send('I need a RAG system for 300 employees')
            while True:
                try:
                    msg = await ws.recv()
                    event = json.loads(msg)
                    print(f'{event.get("type")}: {event.get("payload")}')
                    if event.get('type') == 'complete':
                        break
                    if event.get('type') == 'error':
                        break
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed by server")
                    break
    except Exception as e:
        print(f"Failed to connect or error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
