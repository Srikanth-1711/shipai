import os
import sys
import subprocess
import platform

def build_app():
    """Build the ShipAI Desktop Executable using PyInstaller."""
    print("Starting PyInstaller build for ShipAI...")
    
    # Platform specific path separator for PyInstaller --add-data
    sep = ";" if platform.system() == "Windows" else ":"
    
    # We need to bundle the frontend folder, and the backend/app package
    command = [
        sys.executable,
        "-m", "PyInstaller",
        "--name=ShipAI",
        "--onefile",
        "--windowed",  # No console window
        f"--add-data=frontend{sep}frontend",
        "--hidden-import=uvicorn.logging",
        "--hidden-import=uvicorn.loops",
        "--hidden-import=uvicorn.loops.auto",
        "--hidden-import=uvicorn.protocols",
        "--hidden-import=uvicorn.protocols.http",
        "--hidden-import=uvicorn.protocols.http.auto",
        "--hidden-import=uvicorn.protocols.websockets",
        "--hidden-import=uvicorn.protocols.websockets.auto",
        "--hidden-import=uvicorn.lifespan",
        "--hidden-import=uvicorn.lifespan.on",
        "desktop.py"
    ]
    
    # Run the build
    process = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)
    
    if process.returncode == 0:
        print("\nBuild successful! You can find ShipAI.exe in the 'dist' folder.")
    else:
        print("\nBuild failed. Check the logs above.")

if __name__ == "__main__":
    build_app()
