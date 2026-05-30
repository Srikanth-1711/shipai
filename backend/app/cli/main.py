import os
import sys
import json
import psutil
import httpx
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

console = Console()

# Theme colors
COLORS = {
    "discovery": "blue",
    "architect": "purple",
    "builder": "green",
    "explainer": "cyan",
    "system": "red",
    "success": "green bold"
}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_error(msg: str, solution: str = None):
    text = f"[{COLORS['system']}]{msg}[/]"
    if solution:
        text += f"\n{solution}"
    console.print(Panel(text, title="Error", border_style=COLORS['system']))

def print_success(msg: str):
    console.print(f"[{COLORS['success']}]{msg}[/]")

def detect_hardware() -> dict:
    """Detects basic hardware info for model recommendations."""
    hw = {}
    try:
        hw['cpu_cores'] = psutil.cpu_count(logical=False)
        hw['ram_gb'] = round(psutil.virtual_memory().total / (1024**3))
        
        # Simple GPU check (Windows/Linux)
        # In a real app we'd use pynvml or similar, but for now we'll do a basic check
        has_gpu = False
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                has_gpu = True
                hw['gpu_vram'] = int(result.stdout.strip().split('\n')[0])
        except:
            pass
        
        hw['has_gpu'] = has_gpu
    except:
        hw['cpu_cores'] = 4
        hw['ram_gb'] = 8
        hw['has_gpu'] = False
    return hw

def recommend_model(hw: dict) -> str:
    if not hw.get('has_gpu'):
        return "qwen2.5:1.5b"
    
    vram = hw.get('gpu_vram', 0)
    if vram <= 4096: # 4GB e.g. GTX 1650
        return "qwen2.5:3b"
    elif vram <= 12288: # 10-12GB e.g. RTX 3080
        return "qwen2.5:7b"
    else: # 24GB e.g. RTX 4090
        return "qwen2.5:14b"

def init_flow():
    clear_screen()
    console.print(Panel("[bold cyan]ShipAI[/] - The AI Architect", border_style="cyan"))
    
    with console.status("[blue]Detecting system capabilities...[/]", spinner="dots"):
        hw = detect_hardware()
    
    console.print("\n[bold]System Detected:[/]")
    console.print(f"  CPU: {hw.get('cpu_cores')} cores")
    console.print(f"  RAM: {hw.get('ram_gb')} GB")
    console.print(f"  GPU: {'Yes (' + str(hw.get('gpu_vram', 0)) + 'MB VRAM)' if hw.get('has_gpu') else 'No'}")
    
    model = recommend_model(hw)
    console.print(f"\n[bold]Recommended Model:[/] [green]{model}[/]")
    
    # Check Ollama
    try:
        resp = httpx.get("http://localhost:11434/api/version", timeout=2)
        if resp.status_code == 200:
            console.print("[green]✓ Ollama is running[/]")
    except Exception:
        print_error("Ollama isn't running.", "Start it with: [bold]ollama serve[/bold]")
        sys.exit(1)
        
    # Check if model exists
    try:
        resp = httpx.get("http://localhost:11434/api/tags")
        models = [m['name'] for m in resp.json().get('models', [])]
        if model not in models and f"{model}:latest" not in models:
            print_error("Model not pulled.", f"Run: [bold]ollama pull {model}[/bold]")
            # In a real app we might prompt to pull it here with a progress bar
    except Exception as e:
        console.print(f"[red]Error checking models: {e}[/]")
        
    # Check Knowledge Base
    kb_path = Path("knowledge_base/chroma_db")
    if not kb_path.exists():
        print_error("Knowledge base empty.", "Run: [bold]shipai ingest --fast[/bold]")
    else:
        console.print("[green]✓ Knowledge base found[/]")

def print_requirements_table(requirements: dict):
    table = Table(title="System Requirements", border_style=COLORS["discovery"])
    table.add_column("Field", style=COLORS["discovery"])
    table.add_column("Value", style="white")
    table.add_column("Confidence", style=COLORS["success"])

    for field, value in requirements.items():
        if value:
            table.add_row(str(field), str(value), "✓ confirmed")

    console.print(table)

def print_architecture_decision(config_json: dict):
    config_display = Syntax(
        json.dumps(config_json, indent=2),
        "json",
        theme="monokai",
        background_color="default"
    )
    console.print(Panel(config_display, 
        title=f"[{COLORS['architect']}]Architecture Decision[/]",
        border_style=COLORS['architect']
    ))

def print_welcome_back(user_name: str, last_project: str, completion_pct: int):
    console.print(Panel(
        f"Welcome back [bold]{user_name}[/bold]\n"
        f"Last project: [{COLORS['discovery']}]{last_project}[/]\n"  
        f"Requirements: [{COLORS['success']}]{completion_pct}% complete[/]",
        title="ShipAI",
        border_style=COLORS["discovery"]
    ))

def simulate_generation(files_to_generate: list):
    """Simulate file generation with a Rich progress bar."""
    import time
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[{COLORS['builder']}]{{task.description}}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating project files...", total=len(files_to_generate))
        for file in files_to_generate:
            progress.update(task, description=f"Writing {file}...")
            time.sleep(0.5) # Simulate work
            progress.advance(task)

if __name__ == "__main__":
    init_flow()
