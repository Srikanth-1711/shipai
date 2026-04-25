import threading
import time
import socket
import webview
import uvicorn
import logging
from backend.app.main import app

# Silence uvicorn logs for a cleaner desktop experience
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

def get_free_port():
    """Find a random available port to run the backend on."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def run_server(port):
    """Run the FastAPI backend."""
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

if __name__ == "__main__":
    # 1. Find a free port
    port = get_free_port()
    
    # 2. Start the FastAPI backend in a background thread
    server_thread = threading.Thread(target=run_server, args=(port,), daemon=True)
    server_thread.start()
    
    # Give the server a moment to start
    time.sleep(1.5)
    
    # 3. Open the native Desktop Window pointing to the local backend
    webview.create_window(
        title="ShipAI — The AI Engineer You Can Hire at Scale",
        url=f"http://127.0.0.1:{port}/",
        width=1200,
        height=800,
        min_size=(800, 600),
        resizable=True,
        background_color="#0a0a0a" # Match the dark theme
    )
    
    # 4. Start the GUI loop
    webview.start()
