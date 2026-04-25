"""
ShipAI CLI -- The command-line interface
Usage:
    python -m shipai install          # First-time setup
    python -m shipai create <template> # Generate a project
    python -m shipai models           # List available models
    python -m shipai hardware         # Check hardware
    python -m shipai activate <key>   # Activate license
    python -m shipai serve            # Start the backend server
    python -m shipai status           # Check system status
"""
import sys
import os

# Add parent to path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        print_banner()
        print_help()
        return

    command = args[0].lower()

    if command == "install":
        cmd_install()
    elif command == "create":
        template = args[1] if len(args) > 1 else None
        cmd_create(template)
    elif command == "models":
        cmd_models()
    elif command == "hardware":
        cmd_hardware()
    elif command == "activate":
        key = args[1] if len(args) > 1 else None
        cmd_activate(key)
    elif command == "serve":
        cmd_serve()
    elif command == "status":
        cmd_status()
    elif command == "templates":
        cmd_templates()
    elif command == "advisor":
        description = " ".join(args[1:]) if len(args) > 1 else None
        cmd_advisor(description)
    else:
        print(f"Unknown command: {command}")
        print_help()


def print_banner():
    print("""
 ____  _     _          _    ___
/ ___|| |__ (_)_ __    / \\  |_ _|
\\___ \\| '_ \\| | '_ \\ / _ \\  | |
 ___) | | | | | |_) / ___ \\ | |
|____/|_| |_|_| .__/_/   \\_\\___|
              |_|

  The AI Engineer You Can Hire at Scale
  Ship AI Products in Minutes, Not Months
""")


def print_help():
    print("""
COMMANDS:
  install              First-time setup (check hardware, setup Ollama, pull models)
  create <template>    Generate a new AI project from template
  templates            List available project templates
  models               List locally available LLM models
  hardware             Check system hardware and get model recommendations
  advisor <desc>       Get AI architecture recommendation for your use case
  activate <key>       Activate a license key
  serve                Start the ShipAI backend server
  status               Check system status

EXAMPLES:
  python -m shipai install
  python -m shipai create rag_chatbot
  python -m shipai advisor "I need a customer support chatbot"
  python -m shipai activate SK-P-abc123-xyz789
""")


def cmd_install():
    """First-time setup flow."""
    print_banner()
    print("[1/4] Checking hardware...")

    from app.services.hardware_checker import check_hardware
    hw = check_hardware()

    print(f"  OS:     {hw.os_name} {hw.os_version}")
    print(f"  CPU:    {hw.cpu_name} ({hw.cpu_cores_physical} cores)")
    print(f"  RAM:    {hw.ram_total_gb} GB ({hw.ram_available_gb} GB available)")
    if hw.gpu:
        print(f"  GPU:    {hw.gpu.name} ({hw.gpu.vram_total_mb} MB VRAM)")
    else:
        print("  GPU:    None detected")
    print(f"  Disk:   {hw.disk_free_gb} GB free")
    print(f"  Tier:   {hw.tier_label}")
    print()

    print("[2/4] Checking Ollama...")
    import subprocess
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"  Ollama: Installed ({result.stdout.strip()})")
        else:
            print("  Ollama: Not found. Please install from https://ollama.ai")
            return
    except FileNotFoundError:
        print("  Ollama: Not found. Please install from https://ollama.ai")
        return

    print()
    print("[3/4] Checking available models...")
    import asyncio
    from app.services.ollama_service import ollama_service

    models = asyncio.run(ollama_service.list_models())
    if models:
        print(f"  Found {len(models)} model(s):")
        for m in models:
            print(f"    - {m['name']} ({m['size_gb']} GB, {m['parameters']})")
    else:
        print("  No models found. Pulling recommended model...")
        recommended = hw.recommended_models[0] if hw.recommended_models else "tinyllama"
        print(f"  Pulling {recommended}...")
        result = asyncio.run(ollama_service.pull_model(recommended))
        print(f"  {result.get('status', 'done')}")

    print()
    print("[4/4] Setup complete!")
    print()
    print("  Recommended models for your hardware:")
    for m in hw.recommended_models:
        print(f"    - {m}")
    print()
    print("  Next steps:")
    print("    python -m shipai serve        # Start the platform")
    print("    python -m shipai templates    # See available templates")
    print("    python -m shipai create rag_chatbot  # Create your first project!")
    print()


def cmd_create(template: str = None):
    """Generate a new project from template."""
    if not template:
        print("Usage: python -m shipai create <template>")
        print()
        cmd_templates()
        return

    print(f"Creating project from template: {template}")
    print()

    # Get project name
    project_name = input("Project name: ").strip()
    if not project_name:
        project_name = f"my_{template}_project"

    import asyncio
    from app.services.template_engine import generate_project
    from app.services.license_service import get_current_license

    license_info = get_current_license()

    config = {"project_name": project_name}

    # Template-specific config
    if template == "rag_chatbot":
        system_prompt = input("System prompt (Enter for default): ").strip()
        if system_prompt:
            config["system_prompt"] = system_prompt

    print()
    print(f"Generating {template} project '{project_name}'...")
    result = asyncio.run(generate_project(
        template_id=template,
        config=config,
        tier=license_info.tier,
    ))

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print()
    print(f"  Project: {result['project_name']}")
    print(f"  Template: {result['template']}")
    print(f"  Files: {result['files_generated']}")
    print(f"  Infra patterns: {result['infra_patterns_included']}")
    print(f"  Location: {result['output_dir']}")
    print()
    print("  Next steps:")
    for step in result.get("next_steps", []):
        print(f"    {step}")
    print()


def cmd_models():
    """List available models."""
    import asyncio
    from app.services.ollama_service import ollama_service

    print("Local Models:")
    print("-" * 60)

    models = asyncio.run(ollama_service.list_models())
    if not models:
        print("  No models found. Pull one with: ollama pull tinyllama")
        return

    for m in models:
        print(f"  {m['name']:<25} {m['size_gb']:>6} GB  {m['parameters']:>6}  {m['quantization']}")
    print()


def cmd_hardware():
    """Check system hardware."""
    from app.services.hardware_checker import check_hardware
    hw = check_hardware()

    print("System Hardware:")
    print("-" * 60)
    print(f"  OS:           {hw.os_name} {hw.os_version} ({hw.architecture})")
    print(f"  CPU:          {hw.cpu_name}")
    print(f"  CPU Cores:    {hw.cpu_cores_physical} physical / {hw.cpu_cores_logical} logical")
    print(f"  RAM:          {hw.ram_total_gb} GB total / {hw.ram_available_gb} GB available ({hw.ram_used_percent}% used)")
    if hw.gpu:
        print(f"  GPU:          {hw.gpu.name}")
        print(f"  VRAM:         {hw.gpu.vram_total_mb} MB total / {hw.gpu.vram_free_mb} MB free")
        print(f"  CUDA:         {hw.gpu.cuda_version}")
        print(f"  Driver:       {hw.gpu.driver_version}")
    else:
        print("  GPU:          None detected (CPU-only mode)")
    print(f"  Disk:         {hw.disk_free_gb} GB free / {hw.disk_total_gb} GB total")
    print()
    print(f"  Hardware Tier: {hw.tier_label}")
    print()
    print("  Recommended Models:")
    for m in hw.recommended_models:
        print(f"    - {m}")
    print()


def cmd_activate(key: str = None):
    """Activate a license key."""
    if not key:
        key = input("Enter license key: ").strip()
    if not key:
        print("No key provided.")
        return

    from app.services.license_service import activate_license
    result = activate_license(key)

    if result["status"] == "activated":
        print(f"  License activated successfully!")
        print(f"  Tier: {result['tier']}")
        print(f"  Expires: {result['expires_at']}")
        print(f"  Days remaining: {result['days_remaining']}")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")


def cmd_serve():
    """Start the ShipAI backend server."""
    print_banner()
    print("Starting ShipAI backend server...")
    print("API docs: http://localhost:8000/docs")
    print()

    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


def cmd_status():
    """Check system status."""
    import asyncio
    from app.services.ollama_service import ollama_service
    from app.services.hardware_checker import check_hardware
    from app.services.license_service import get_current_license

    print("ShipAI Status:")
    print("-" * 60)

    # Ollama
    status = asyncio.run(ollama_service.health_check())
    if status["status"] == "healthy":
        print(f"  Ollama:    Running (v{status.get('version', '?')})")
    else:
        print(f"  Ollama:    Not running")

    # Models
    models = asyncio.run(ollama_service.list_models())
    print(f"  Models:    {len(models)} available")

    # Hardware
    hw = check_hardware()
    print(f"  Hardware:  {hw.tier_label}")
    print(f"  RAM:       {hw.ram_available_gb} GB free / {hw.ram_total_gb} GB total")

    # License
    lic = get_current_license()
    print(f"  License:   {lic.tier.upper()} tier")

    print()


def cmd_templates():
    """List available templates."""
    from app.services.template_engine import list_templates
    from app.services.license_service import get_current_license

    lic = get_current_license()
    templates = list_templates()

    print("Available Templates:")
    print("-" * 60)
    for t in templates:
        locked = "" if t["id"] in lic.features.get("templates", []) else " [LOCKED]"
        print(f"  {t['icon']}  {t['name']:<25} {t['tier']:>10} tier{locked}")
        print(f"     {t['description']}")
        print(f"     Setup: {t['estimated_setup_time']} | Min hardware: {t['min_hardware']}")
        print()


def cmd_advisor(description: str = None):
    """Get AI architecture recommendation."""
    if not description:
        description = input("Describe your use case: ").strip()
    if not description:
        print("No description provided.")
        return

    print()
    print("Analyzing your use case...")
    print()

    import asyncio
    from app.services.ai_advisor import analyze_use_case
    from app.services.hardware_checker import check_hardware

    hw = check_hardware()
    result = asyncio.run(analyze_use_case(description, hw.hardware_tier))

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # Show quick matches first
    if result.get("quick_matches"):
        print("Quick Matches:")
        for match in result["quick_matches"]:
            print(f"  - {match['pattern']} ({match['category']}) - Relevance: {match['relevance_score']}")
        print()

    # Show LLM analysis
    print("AI Advisor Analysis:")
    print("-" * 60)
    print(result.get("analysis", "No analysis available"))
    print()
    print(f"(Model: {result.get('model_used', '?')}, Time: {result.get('inference_time_ms', 0):.0f}ms)")
    print()


if __name__ == "__main__":
    main()
