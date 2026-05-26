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
    python -m shipai config           # Manage ShipAI config
    python -m shipai insights         # View architectural decision insights
    python -m shipai plan             # Detect env + assign models → ~/.shipai/model_plan.json
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
    elif command == "config":
        cmd_config(args[1:])
    elif command == "insights":
        cmd_insights()
    elif command == "plan":
        cmd_plan()
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
  install              Setup Fleet: detect env, negotiate models, pull, save plan
  create <template>    Generate a new AI project from template
  templates            List available project templates
  models               List locally available LLM models
  hardware             Check system hardware and get model recommendations
  advisor <desc>       Get AI architecture recommendation for your use case
  activate <key>       Activate a license key
  serve                Start the ShipAI backend server
  status               Check system status
  config               Set configuration values (e.g. --github-token)
  insights             View intelligence loop insights and optimize patterns
  plan                 Detect hardware/runtimes and write ~/.shipai/model_plan.json

EXAMPLES:
  python -m shipai install
  python -m shipai create rag_chatbot
  python -m shipai advisor "I need a customer support chatbot"
  python -m shipai activate SK-P-abc123-xyz789
  python -m shipai plan
""")


def _print_plan_summary(plan, pulled=None):
    from app.install.model_plan import PLAN_PATH

    print(f"\nPlan saved: {PLAN_PATH}\n")
    for node, asn in plan.node_assignments.items():
        flag = "INSTALLED" if asn.status == "installed" else asn.status.upper()
        print(f"  {node:<16} {asn.model or '-':<28} [{flag}, {asn.confidence}]")
    if plan.embedding:
        print(f"  {'embeddings':<16} {plan.embedding.model:<28} [installed]")
    if pulled:
        print(f"\nDownloaded this run: {', '.join(pulled)}")
    if plan.models_to_download:
        print("\nStill needed:")
        for m in plan.models_to_download:
            print(f"  ollama pull {m}")
    if plan.warnings:
        print("\nWarnings:")
        for w in plan.warnings:
            print(f"  ! {w}")
    print()


def _run_setup_fleet_cli(auto_pull: bool):
    import asyncio
    from app.setup.fleet import run_setup_fleet

    def on_progress(agent: str, message: str) -> None:
        print(f"  [{agent}] {message}")

    print("ShipAI Setup Fleet\n")
    print("  Scout -> Librarian -> Negotiator -> Acquirer -> Validator\n")
    result = asyncio.run(run_setup_fleet(auto_pull=auto_pull, on_progress=on_progress))
    if result.plan:
        _print_plan_summary(result.plan, pulled=result.state.pulled_models)
    if result.state.success:
        print("Setup complete. Next: python -m shipai serve")
    else:
        print("Setup finished with warnings — check plan and pull any missing models.")
    print()


def cmd_plan():
    """Detect environment and write model plan (no auto-download)."""
    print("Negotiating models (no download)...")
    print("(Installed = live Ollama API. Matrix scores = estimates until benchmarked.)\n")
    _run_setup_fleet_cli(auto_pull=False)


def cmd_install():
    """First-time setup — full Setup Fleet including model downloads."""
    print_banner()
    _run_setup_fleet_cli(auto_pull=True)


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


def cmd_insights():
    """View intelligence loop insights and pattern optimization."""
    import asyncio
    import json
    from app.db.feedback_db import count_records, overall_rate, get_top_corrections
    from app.engine.pattern_optimizer import optimize_patterns
    
    print("ShipAI Architectural Insights")
    print("=" * 60)
    print(f"Total Sessions Logged: {count_records()}")
    print(f"Overall Acceptance Rate: {overall_rate() * 100:.1f}%\n")
    
    top = get_top_corrections()
    if top:
        print("Top User Corrections:")
        for k, v in top.items():
            print(f"  - {k} ({v} times)")
        print()
    else:
        print("No user corrections logged yet.\n")
        
    print("Running Pattern Optimizer...")
    print("(This takes a moment as it analyzes weak decisions against L4 engineering principles)\n")
    
    suggestions = asyncio.run(optimize_patterns())
    if isinstance(suggestions, str):
        print(suggestions)
    else:
        for s in suggestions:
            print(f"Weak Decision Detected: {s['component']} ({s['chosen']})")
            print(f"User Preferred: {s['preferred']} (Acceptance Rate: {s['acceptance_rate']*100:.1f}%)")
            print("Suggested Rule Update:")
            print(s['suggestion'])
            print("-" * 60)
            
if __name__ == "__main__":
    main()
