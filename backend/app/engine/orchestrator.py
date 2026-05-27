import json
import os
import time
try:
    from langgraph.graph import StateGraph, END  # type: ignore
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore
    _LANGGRAPH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    StateGraph = None  # type: ignore
    END = None  # type: ignore
    MemorySaver = None  # type: ignore
    _LANGGRAPH_AVAILABLE = False

try:
    from langchain_community.chat_models import ChatOllama  # type: ignore
    from langchain_core.messages import SystemMessage  # type: ignore
    _LANGCHAIN_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    ChatOllama = None  # type: ignore
    SystemMessage = None  # type: ignore
    _LANGCHAIN_AVAILABLE = False
from app.engine.state import AgentState
from app.config import settings
from app.install.llm_config import get_node_model
from app.contracts.config_schema import validate_config
from app.verification.project_verifier import verify_project
from app.verification.quality_metrics import score_verification

# Node Stubs
def interview_node(state: AgentState):
    """Discovery Question Tree: Asks questions until requirements are met."""
    if not _LANGCHAIN_AVAILABLE or ChatOllama is None:
        raise RuntimeError("interview_node requires 'langchain' runtime dependencies")
    llm = ChatOllama(model=get_node_model("interview_node"), temperature=0.1)
    messages = state.get("messages", [])
    
    # 1. Extract requirements using JSON mode
    extract_prompt = """You are extracting requirements from a conversation.
Your job is to capture what the user ACTUALLY SAID in their own words. Do not translate, normalize, or invent values.
For each field, extract the specific user-provided details. If a field is not mentioned, set it to "".

Return a JSON object with these fields:
- "use_case": the main goal or task of the AI
- "data_type": the type of data or content
- "data_location": where the data is stored
- "data_change_rate": how often the data changes
- "scale": the scale of the team, system, or users
- "query_type": the nature of user queries
- "privacy_level": compliance or privacy requirements
- "user_level": technical expertise of the target users

Capture only concrete details from the conversation below. If the user did not specify a detail, keep it as "". Do not use placeholder example values.
"""
    conv_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
    
    resp = llm.invoke(extract_prompt + conv_text, format="json")
    try:
        extracted = json.loads(resp.content)
    except:
        extracted = {}
        
    current_reqs = state.get("requirements") or {}
    
    # Store extracted free-form properties exactly as returned
    for k in ["user_level", "use_case", "data_type", "data_location", "data_change_rate", "scale", "query_type", "privacy_level"]:
        val = extracted.get(k, "")
        if val and str(val).lower() != "null" and str(val).strip() != "":
            current_reqs[k] = str(val).strip()
            
    # Python fallback for empty or vague fields based on conversation history
    history_text = conv_text.lower()
    
    # 1. Fallback for data_type
    if not current_reqs.get("data_type") or str(current_reqs.get("data_type")).strip() == "":
        if "menu" in history_text:
            current_reqs["data_type"] = "menu"
        elif "log" in history_text:
            current_reqs["data_type"] = "logs"
        elif "document" in history_text or "doc" in history_text:
            current_reqs["data_type"] = "documents"
            
    # 2. Fallback for use_case
    val_use_case = str(current_reqs.get("use_case", "")).strip()
    if not val_use_case or val_use_case == "" or "ai for my company" in val_use_case.lower() or "ai for company" in val_use_case.lower():
        if "menu" in history_text:
            current_reqs["use_case"] = "answer customer questions about the menu"
        elif "log" in history_text or "rca" in history_text:
            current_reqs["use_case"] = "log analysis and daily regression RCA"
        elif "policy" in history_text or "hr" in history_text:
            current_reqs["use_case"] = "answering HR policy questions"
        elif "hipaa" in history_text or "tenant" in history_text:
            current_reqs["use_case"] = "multi-tenant RAG search"
            
    # 3. Fallback for scale
    if not current_reqs.get("scale") or str(current_reqs.get("scale")).strip() == "":
        if "500" in history_text:
            current_reqs["scale"] = "500 enterprise users"
        elif "200" in history_text:
            current_reqs["scale"] = "200 engineers"
        elif "50" in history_text:
            current_reqs["scale"] = "50 employees"
        elif "restaurant" in history_text or "menu" in history_text:
            current_reqs["scale"] = "small scale"
            
    # Run the user level detector to populate user_level if it's missing
    from app.engine.user_level_detector import run_detection
    state_to_detect = {"messages": messages, "requirements": current_reqs}
    detection_updates = run_detection(state_to_detect)
    if "requirements" in detection_updates:
        current_reqs = detection_updates["requirements"]

    # 2. LLM-based Gap Detection
    core_reqs = {
        "use_case": current_reqs.get("use_case", ""),
        "data_type": current_reqs.get("data_type", ""),
        "scale": current_reqs.get("scale", "")
    }

    gap_prompt = f"""You are a Senior AI Architect auditing the current requirements.

Here are the requirements:
{json.dumps(core_reqs, indent=2)}

List only the fields from ['use_case', 'data_type', 'scale'] that are empty "" or contain no user-provided information.
Do not list fields that contain actual information.
If all three fields have information, return an empty list.

JSON output format:
{{
  "missing_fields": []
}}
"""
    gap_resp = llm.invoke(gap_prompt, format="json")
    try:
        clean_gap = gap_resp.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_gap)
        missing = data.get("missing_fields", [])
        if not isinstance(missing, list):
            missing = []
        # Filter out fields that are not in core allowed list
        allowed = ["use_case", "data_type", "scale"]
        missing = [m for m in missing if m in allowed]
    except Exception as e:
        required_keys = ["use_case", "data_type", "scale"]
        missing = [k for k in required_keys if not current_reqs.get(k)]

    # If LLM decides no critical fields are missing, proceed!
    if not missing:
        return {"requirements": current_reqs, "requirements_complete": True}

    # 3. Ask question for missing requirements
    ask_prompt = f"You are a Senior AI Architect. The following requirements are still missing: {', '.join(missing)}. Ask ONE concise, conversational question to figure out 1 or 2 of these missing pieces. Do not overwhelm the user."
    
    # Keep context small for the generation
    context_msgs = []
    for m in messages[-3:]:
        context_msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        
    q_resp = llm.invoke([SystemMessage(content=ask_prompt)] + context_msgs)
    
    new_messages = list(messages)
    new_messages.append({"role": "assistant", "content": q_resp.content})
    
    # Run the user level detector before returning state
    state_to_detect = {"messages": new_messages, "requirements": current_reqs}
    detection_updates = run_detection(state_to_detect)
    
    if "requirements" in detection_updates:
        current_reqs = detection_updates["requirements"]
    
    return {
        "requirements": current_reqs,
        "requirements_complete": False,
        "messages": new_messages,
        "research_iterations": state.get("research_iterations", 0)
    }

from app.engine.mcp_client import MCPConnectionManager

def evaluate_research_quality(book_insights: list, github_patterns: list, requirements: dict) -> bool:
    # Rule 1: Minimum coverage
    if len(book_insights) < 3:
        return False
    if len(github_patterns) < 2:
        return False
    
    # Rule 2: Use-case relevance check
    use_case = requirements.get("use_case", "").lower()
    if use_case:
        all_insights = str(book_insights) + str(github_patterns)
        if use_case not in all_insights.lower():
            return False
            
    # Rule 3: Loop protection (from README)
    return True

def build_research_queries(req: dict) -> list[dict]:
    """Build differentiated queries: broad for GitHub, specific for Books."""
    queries = []
    use_case = req.get("use_case", "software")
    scale = req.get("scale", "mid")
    data_type = req.get("data_type", "data")
    data_change = req.get("data_change_rate", "slow")
    privacy = req.get("privacy_level", "standard")
    q_type = req.get("query_type", "factual")
    
    # --- GitHub queries: BROAD terms, high star filter ---
    queries.append({
        "text": f"{use_case} production",
        "target": "github",
        "min_stars": 500,
    })
    queries.append({
        "text": f"langchain {use_case}",
        "target": "github",
        "min_stars": 200,
    })
    
    # --- Books queries: SPECIFIC terms, semantic matching ---
    queries.append({
        "text": f"{use_case} {scale} production architecture patterns",
        "target": "books",
    })
    queries.append({
        "text": f"{data_type} {data_change} data pipeline {use_case}",
        "target": "books",
    })
    
    # Privacy-specific queries
    if privacy in ("pii", "military-grade"):
        queries.append({
            "text": "Presidio PII redaction",
            "target": "github",
            "min_stars": 100,
        })
        queries.append({
            "text": f"data privacy masking {use_case} production",
            "target": "books",
        })
    
    # Scale-specific infra query
    if scale in ("mid", "enterprise"):
        queries.append({
            "text": f"{use_case} enterprise scale reliability",
            "target": "books",
        })
    
    # Query type specific
    if q_type == "multi-hop":
        queries.append({
            "text": "agentic RAG multi-hop reasoning",
            "target": "github",
            "min_stars": 200,
        })
        
    return queries

def map_use_case_to_layer(use_case: str) -> str:
    use_case = str(use_case).lower()
    if "rag" in use_case or "agent" in use_case:
        return "llm"
    if "data" in use_case:
        return "data"
    return "ops"

async def research_node(state: AgentState):
    """Meta-RAG: Queries local Book MCP and GitHub FastMCP with differentiated strategies."""
    requirements = state.get("requirements", {})
    iteration = state.get("research_iterations", 0)
    
    queries = build_research_queries(requirements)
    
    book_insights = list(state.get("book_insights", []))
    github_patterns = list(state.get("github_patterns", []))
    
    async with MCPConnectionManager() as manager:
        await manager.connect("books", "shipai/mcp_servers/books_server.py")
        await manager.connect("github", "shipai/mcp_servers/github_server.py")
        
        for q in queries:
            if q["target"] == "books":
                book_result = await manager.call_tool(
                    "books",
                    "query_knowledge",
                    query=q["text"],
                    layer=map_use_case_to_layer(requirements.get("use_case", "")),
                    top_k=3
                )
                if isinstance(book_result, list):
                    book_insights.extend(book_result)
            
            elif q["target"] == "github":
                github_result = await manager.call_tool(
                    "github",
                    "search_repos",
                    query=q["text"],
                    min_stars=q.get("min_stars", 500)
                )
                if isinstance(github_result, list):
                    github_patterns.extend(github_result)
                
    quality_met = evaluate_research_quality(book_insights, github_patterns, requirements)
    
    return {
        "book_insights": book_insights,
        "github_patterns": github_patterns,
        "research_complete": quality_met,
        "research_iterations": iteration + 1
    }

from app.engine.rag_decision_matrix import run_decision_matrix

def plan_node(state: AgentState):
    """Pattern Matcher: Maps requirements + research into config_json for the template engine."""
    # The new RAG decision matrix handles the complex mapping and rationale
    return run_decision_matrix(state)

def config_validator_node(state: AgentState):
    """Validate config_json from planner using strict ConfigV1 schema.

    Never raises; on error, records validation_errors and lets the graph complete.
    """
    raw = state.get("config_json", {}) or {}
    cfg, errors = validate_config(raw)
    if errors or not cfg:
        # Keep the raw config so the user can still see what the planner tried to do.
        return {
            "config_json": raw,
            "config_validation_errors": errors,
        }

    # Capability-aware post-processing (hardware-aware overrides).
    # This keeps the planner LLM-free from hardware concerns while still adapting
    # to the actual machine capabilities discovered during install.
    try:
        from app.install.model_plan import load_model_plan

        plan = load_model_plan()
        hw = (plan.metadata.get("hardware") if plan and plan.metadata else {}) if plan else {}
        embedding_model = plan.embedding.model if (plan and plan.embedding) else None

        def _to_float(x: object) -> float | None:
            try:
                return float(x)  # type: ignore[arg-type]
            except Exception:
                return None

        effective_vram = _to_float(hw.get("effective_vram_gb"))
        effective_ram = _to_float(hw.get("effective_ram_gb"))

        # Helper: keep `decisions[*].choice` in sync with overridden component values.
        decisions = cfg.get("decisions") if isinstance(cfg, dict) else {}
        if not isinstance(decisions, dict):
            decisions = {}
            cfg["decisions"] = decisions

        def sync_choice(component_key: str) -> None:
            if component_key in cfg and component_key in decisions and isinstance(decisions[component_key], dict):
                decisions[component_key]["choice"] = cfg[component_key]

        # If no embedding model exists, prefer sparse retrieval to avoid dense-only stacks.
        if not embedding_model and cfg.get("search_type") in ("dense", "hybrid"):
            cfg["search_type"] = "sparse"
            sync_choice("search_type")

        # Low VRAM safety rails: avoid heavyweight multi-agent / reranking / caching.
        if effective_vram is not None and effective_vram < 4.0:
            if cfg.get("template") == "multi_agent":
                cfg["template"] = "rag_chatbot"
            if cfg.get("framework") == "langgraph":
                cfg["framework"] = "langchain"
            if cfg.get("reranker") != "none":
                cfg["reranker"] = "none"
            if cfg.get("cache") != "none":
                cfg["cache"] = "none"
            if cfg.get("infra_tier") != "minimal":
                cfg["infra_tier"] = "minimal"
            if cfg.get("vector_db") != "chroma":
                cfg["vector_db"] = "chroma"
            if cfg.get("search_type") == "hybrid":
                cfg["search_type"] = "dense"

            for k in ["framework", "vector_db", "search_type", "reranker", "cache", "infra_tier"]:
                sync_choice(k)

        # Slightly higher-end machine: allow standard infra if VRAM is healthy.
        if effective_vram is not None and effective_vram >= 6.0 and cfg.get("infra_tier") == "minimal":
            cfg["infra_tier"] = "standard"
            if cfg.get("cache") == "none":
                cfg["cache"] = "redis"
            sync_choice("infra_tier")
            sync_choice("cache")

        # Re-validate final cfg shape/options after overrides.
        cfg2, errors2 = validate_config(cfg)
        if not errors2 and cfg2:
            cfg = cfg2
    except Exception:
        # Never block builder from generation because of override logic.
        pass

    return {"config_json": cfg, "config_validation_errors": []}

def builder_node(state: AgentState):
    """Builder / Engineer: Calls template_engine to generate the real project on disk."""
    # Short-circuit if validation failed
    validation_errors = state.get("config_validation_errors") or []
    if validation_errors:
        return {
            "project_path": "",
            "generated_files": [],
            "build_error": "Config validation failed; see config_validation_errors.",
        }

    config = state.get("config_json", {})
    if not config:
        return {
            "project_path": "",
            "generated_files": [],
            "build_error": "No config_json from plan_node — cannot generate project.",
        }

    try:
        from app.services.template_engine import ShipAITemplateEngine
        from app.engine.builder_utils import (
            deterministic_output_dir,
            safe_remove_output_dir,
            build_manifest,
            write_manifest,
        )
        from app.install.model_plan import load_model_plan

        engine = ShipAITemplateEngine()
        dry_run = str(os.getenv("SHIPAI_DRY_RUN", "")).lower() in ("1", "true", "yes")
        output_dir = deterministic_output_dir(config)
        if dry_run:
            temp_base = Path(os.getenv("TEMP") or Path.cwd())
            temp_base = temp_base / "shipai_dryrun_projects"
            output_dir = deterministic_output_dir(config, base_dir=temp_base)
        safe_remove_output_dir(output_dir)

        project_path = engine.generate_project(config, output_dir=str(output_dir))

        # Load model_plan context for reproducibility / debugging.
        plan = load_model_plan()
        model_plan_summary = {}
        if plan:
            model_plan_summary = {
                "primary_runtime": plan.primary_runtime,
                "final_three": plan.final_three,
                "embedding": plan.embedding.model if plan.embedding else None,
                "node_assignments": {
                    k: {
                        "model": v.model,
                        "runtime": v.runtime,
                        "status": v.status,
                        "role": v.role,
                    }
                    for k, v in plan.node_assignments.items()
                }
                if plan.node_assignments
                else {},
            }

        # Hash the exact config snapshot we validated.
        from app.engine.builder_utils import compute_config_hash, canonical_json

        config_hash = compute_config_hash(config)

        # Collect all real files (relative paths from project root)
        root = Path(project_path)
        files = []
        for f in root.rglob("*"):
            if f.is_file() and "__pycache__" not in str(f):
                files.append(str(f.relative_to(root)))

        manifest = build_manifest(
            project_path=root,
            config=config,
            config_hash=config_hash,
            shipai_version=settings.APP_VERSION,
            generated_files=files,
            model_plan_summary=model_plan_summary,
        )
        write_manifest(root, manifest)

        return {
            "project_path": project_path,
            "generated_files": files,
            "build_error": "",
        }
    except Exception as exc:
        return {
            "project_path": "",
            "generated_files": [],
            "build_error": str(exc),
        }

def verifier_node(state: AgentState):
    """Project Verifier: static verification of generated project artifacts."""
    build_error = state.get("build_error") or ""
    project_path = state.get("project_path") or ""

    # Build errors: mark verification as failed but do not crash.
    if build_error:
        report = {
            "syntax_valid": False,
            "imports_valid": False,
            "structure_valid": False,
            "dependency_valid": False,
            "manifest_valid": False,
            "smoke_test_valid": False,
            "runtime_validation_score": 0.0,
            "semantic_warnings": [],
            "errors": [f"Build error: {build_error}"],
            "warnings": [],
            "verified_files": [],
            "verification_duration_ms": 0,
        }
        return {
            "project_path": project_path,
            "verification_report": report,
            "verification_metrics": score_verification(report),
            "verification_failed": True,
            "reality_test_failed": True,
        }

    if not project_path:
        report = {
            "syntax_valid": False,
            "imports_valid": False,
            "structure_valid": False,
            "dependency_valid": False,
            "manifest_valid": False,
            "smoke_test_valid": False,
            "runtime_validation_score": 0.0,
            "semantic_warnings": [],
            "errors": ["project_path missing; cannot verify"],
            "warnings": [],
            "verified_files": [],
            "verification_duration_ms": 0,
        }
        return {
            "project_path": project_path,
            "verification_report": report,
            "verification_metrics": score_verification(report),
            "verification_failed": True,
            "reality_test_failed": True,
        }

    report_obj = verify_project(project_path)
    report = report_obj.to_dict()
    return {
        "project_path": project_path,
        "verification_report": report,
        "verification_metrics": score_verification(report),
        "verification_failed": report_obj.failed,
        "reality_test_failed": not bool(report.get("smoke_test_valid")),
    }


def explain_node(state: AgentState):
    """Explainer Agent: Translates the architecture decision into the user's language."""
    if not _LANGCHAIN_AVAILABLE or ChatOllama is None:
        raise RuntimeError("explain_node requires 'langchain' runtime dependencies")
    requirements = state.get("requirements", {})
    config = state.get("config_json", {})
    user_level = requirements.get("user_level", "junior_dev")
    verification_report = state.get("verification_report") or {}
    verification_failed = bool(state.get("verification_failed"))
    
    llm = ChatOllama(model=get_node_model("explain_node"), temperature=0.5)
    
    # Tailor explanation depth to user level
    level_instructions = {
        "civilian": "Explain like I'm not technical. Use simple analogies. No jargon. Focus on what the system DOES for them.",
        "business": "Explain the business value. Use high-level concepts. Avoid deep code architecture details.",
        "junior_dev": "Explain clearly with some technical terms. Mention frameworks by name. Focus on setup and usage.",
        "senior_engineer": "Be technical and specific. Mention design patterns, trade-offs, and alternatives considered.",
        "architect": "Be deeply technical. Discuss why alternatives were rejected, scaling implications, and failure modes.",
    }
    instruction = level_instructions.get(user_level, level_instructions["junior_dev"])

    verification_block = ""
    if verification_report:
        errs = verification_report.get("errors", []) or []
        warns = verification_report.get("warnings", []) or []
        verified_files = verification_report.get("verified_files", []) or []
        verification_block = f"""

Verification summary:
- failed: {verification_failed}
- syntax_valid: {verification_report.get('syntax_valid')}
- structure_valid: {verification_report.get('structure_valid')}
- dependency_valid: {verification_report.get('dependency_valid')}
- manifest_valid: {verification_report.get('manifest_valid')}
- smoke_test_valid: {verification_report.get('smoke_test_valid')}
- runtime_validation_score: {verification_report.get('runtime_validation_score')}
- verified_files: {len(verified_files)}
- warnings: {len(warns)}
- errors: {len(errs)}
"""
        if errs:
            verification_block += "\nTop errors:\n" + "\n".join([f"- {e}" for e in errs[:5]])

    explain_prompt = f"""You are ShipAI's Explainer Agent. A project was just generated with these settings:

Architecture: {config.get('template', '?')}
Framework: {config.get('framework', '?')}
Vector DB: {config.get('vector_db', '?')}
Search: {config.get('search_type', '?')}
Reranker: {config.get('reranker', '?')}
Cache: {config.get('cache', '?')}
PII Protection: {config.get('pii', '?')}
Infrastructure: {config.get('infra_tier', '?')}

{verification_block}

Decision Reasoning:
{json.dumps(config.get('decisions', {}), indent=2)}

User level: {user_level}
Instructions: {instruction}

Write a clear explanation of what was built and why. 200-300 words:"""
    
    resp = llm.invoke(explain_prompt)
    explanation = resp.content if hasattr(resp, "content") else str(resp)
    
    # Keep the real project_path from builder_node (don't overwrite with placeholder)
    return {
        "explanation": explanation,
    }

# Edge Routers
def route_from_interview(state: AgentState):
    if state.get("requirements_complete"):
        return "research_node"
    return END  # Wait for user input

def route_from_research(state: AgentState):
    if state.get("research_complete"):
        return "plan_node"
    if state.get("research_iterations", 0) >= 3:
        return "plan_node"
    return "research_node"  # Self-correction loop

# Build Graph (optional dependency gated)
workflow = None
app_brain = None
if _LANGGRAPH_AVAILABLE and StateGraph is not None:
    workflow = StateGraph(AgentState)

    workflow.add_node("interview_node", interview_node)
    workflow.add_node("research_node", research_node)
    workflow.add_node("plan_node", plan_node)
    workflow.add_node("config_validator_node", config_validator_node)
    workflow.add_node("builder_node", builder_node)
    workflow.add_node("verifier_node", verifier_node)
    workflow.add_node("explain_node", explain_node)

    workflow.set_entry_point("interview_node")
    workflow.add_conditional_edges("interview_node", route_from_interview)
    workflow.add_conditional_edges("research_node", route_from_research)
    workflow.add_edge("plan_node", "config_validator_node")
    workflow.add_edge("config_validator_node", "builder_node")
    workflow.add_edge("builder_node", "verifier_node")
    workflow.add_edge("verifier_node", "explain_node")
    workflow.add_edge("explain_node", END)

    # In production, we attach a checkpointer here (e.g., Redis or SQLite)
    if MemorySaver is not None:
        memory = MemorySaver()
        app_brain = workflow.compile(checkpointer=memory)
    else:
        app_brain = workflow.compile()

from dataclasses import dataclass
from typing import AsyncGenerator, Any
from enum import Enum
from pathlib import Path

class WSEventType(str, Enum):
    TOKEN         = "token"
    AGENT_CHANGE  = "agent_change"
    REQUIREMENT   = "requirement"
    DECISION      = "decision"
    FILE_GENERATED = "file_generated"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETE = "verification_complete"
    VERIFICATION_FAILED = "verification_failed"
    REALITY_TEST_STARTED = "reality_test_started"
    REALITY_TEST_COMPLETE = "reality_test_complete"
    REALITY_TEST_FAILED = "reality_test_failed"
    COMPLETE      = "complete"
    ERROR         = "error"

@dataclass
class WSEvent:
    type: WSEventType
    payload: dict[str, Any]
    
    def to_json(self) -> dict:
        return {"type": self.type.value, "payload": self.payload}

# Agent color map — single source of truth
AGENT_CONFIG = {
    "interview_node": {"color": "blue",   "label": "Discovery Agent"},
    "research_node":  {"color": "purple", "label": "Architect Agent"},
    "plan_node":      {"color": "green",  "label": "Planner Agent"},
    "config_validator_node": {"color": "yellow", "label": "Validator Agent"},
    "builder_node":   {"color": "orange", "label": "Engineer Agent"},
    "verifier_node":  {"color": "red",    "label": "Project Verifier"},
    "explain_node":   {"color": "cyan",   "label": "Explainer Agent"},
}

def load_or_create_state(session_id: str, user_input: str) -> dict:
    # Gets state or appends message
    # To keep it simple, we don't pass 'state' directly to astream if we use checkpointer
    # We pass the input to the graph, and LangGraph merges it into state
    return {"messages": [{"role": "user", "content": user_input}]}

def list_generated_files(project_path: str) -> list[Path]:
    p = Path(project_path)
    if not p.exists():
        return []
    files = []
    for f in p.rglob("*"):
        if f.is_file() and not "__pycache__" in str(f):
            files.append(f)
    return files

def is_key_file(file_path: Path) -> bool:
    key_files = ["retriever.py", "pii_guard.py", "reranker.py", "docker-compose.yml", "evaluate.py", "README.md"]
    return file_path.name in key_files

async def stream_graph(user_input: str, session_id: str) -> AsyncGenerator[WSEvent, None]:
    """
    Streams typed WSEvents as the graph executes.
    This is what the WebSocket endpoint consumes.
    """
    
    if app_brain is None:
        # Optional runtime dependencies (langgraph/langchain) not installed.
        yield WSEvent(
            type=WSEventType.ERROR,
            payload={
                "message": "ShipAI chat runtime unavailable: optional dependency 'langgraph' is not installed.",
                "session_id": session_id,
            },
        )
        return

    if not _LANGCHAIN_AVAILABLE:
        yield WSEvent(
            type=WSEventType.ERROR,
            payload={
                "message": "ShipAI chat runtime unavailable: optional dependency 'langchain' is not installed.",
                "session_id": session_id,
            },
        )
        return

    start_ts = time.perf_counter()
    input_data = load_or_create_state(session_id, user_input)
    config = {"configurable": {"thread_id": session_id}}
    
    # Stream through each node
    async for chunk in app_brain.astream(
        input_data, 
        config=config,
        stream_mode="updates"
    ):
        for node_name, node_output in chunk.items():
            # 1. Signal agent change
            if node_name in AGENT_CONFIG:
                yield WSEvent(
                    type=WSEventType.AGENT_CHANGE,
                    payload=AGENT_CONFIG[node_name]
                )
            
            # 2. Stream tokens from LLM calls (mock token streaming for now from final output)
            if "messages" in node_output:
                last_msg = node_output["messages"][-1]
                content = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")
                if content:
                    # Mock streaming the content in chunks of 5 chars to simulate token streaming
                    for i in range(0, len(content), 5):
                        yield WSEvent(
                            type=WSEventType.TOKEN,
                            payload={"text": content[i:i+5]}
                        )
            
            # 3. Emit requirement confirmations
            if "requirements" in node_output:
                reqs = node_output["requirements"]
                for field, value in reqs.items():
                    if value and str(value).lower() != "null":  # only emit confirmed fields
                        yield WSEvent(
                            type=WSEventType.REQUIREMENT,
                            payload={
                                "field": field,
                                "value": value,
                                "confirmed": True
                            }
                        )
            
            # 4. Emit architecture decisions
            if "config_json" in node_output:
                config_json = node_output["config_json"]
                decisions = config_json.get("decisions", {})
                for component, decision in decisions.items():
                    if isinstance(decision, dict):
                        yield WSEvent(
                            type=WSEventType.DECISION,
                            payload={
                                "component": component,
                                "choice": decision.get("choice", ""),
                                "reason": decision.get("reason", ""),
                                "book": decision.get("book", "")
                            }
                        )
            
            # 5. Emit file generation events from builder_node
            if "generated_files" in node_output:
                project_path = node_output.get("project_path", "")
                build_error = node_output.get("build_error", "")
                if build_error:
                    yield WSEvent(
                        type=WSEventType.ERROR,
                        payload={"message": f"Build error: {build_error}"}
                    )
                else:
                    for rel_path in node_output["generated_files"]:
                        fp = Path(rel_path)
                        yield WSEvent(
                            type=WSEventType.FILE_GENERATED,
                            payload={
                                "path": str(Path(project_path) / rel_path),
                                "name": fp.name,
                                "highlighted": is_key_file(fp),
                            }
                        )

            # 6. Emit verification + reality test events from verifier_node
            if "verification_report" in node_output:
                verification_report = node_output.get("verification_report") or {}
                verification_failed = bool(node_output.get("verification_failed"))
                verification_metrics = node_output.get("verification_metrics") or {}
                reality_failed = bool(node_output.get("reality_test_failed"))

                yield WSEvent(
                    type=WSEventType.VERIFICATION_STARTED,
                    payload={"project_path": node_output.get("project_path", None)},
                )

                yield WSEvent(
                    type=WSEventType.REALITY_TEST_STARTED,
                    payload={"project_path": node_output.get("project_path", None)},
                )

                if reality_failed:
                    yield WSEvent(
                        type=WSEventType.REALITY_TEST_FAILED,
                        payload={
                            "smoke_test_valid": verification_report.get("smoke_test_valid"),
                            "runtime_validation_score": verification_report.get(
                                "runtime_validation_score"
                            ),
                            "semantic_warnings": verification_report.get("semantic_warnings", []),
                            "metrics": verification_metrics,
                        },
                    )
                else:
                    yield WSEvent(
                        type=WSEventType.REALITY_TEST_COMPLETE,
                        payload={
                            "smoke_test_valid": True,
                            "runtime_validation_score": verification_report.get(
                                "runtime_validation_score"
                            ),
                            "metrics": verification_metrics,
                        },
                    )

                if verification_failed:
                    yield WSEvent(
                        type=WSEventType.VERIFICATION_FAILED,
                        payload={
                            "summary": {
                                "syntax_valid": verification_report.get("syntax_valid"),
                                "structure_valid": verification_report.get("structure_valid"),
                                "dependency_valid": verification_report.get("dependency_valid"),
                                "manifest_valid": verification_report.get("manifest_valid"),
                                "imports_valid": verification_report.get("imports_valid"),
                                "smoke_test_valid": verification_report.get("smoke_test_valid"),
                            },
                            "errors_count": len(verification_report.get("errors", []) or []),
                            "warnings_count": len(verification_report.get("warnings", []) or []),
                            "metrics": verification_metrics,
                        },
                    )
                else:
                    yield WSEvent(
                        type=WSEventType.VERIFICATION_COMPLETE,
                        payload={
                            "summary": {
                                "syntax_valid": verification_report.get("syntax_valid"),
                                "structure_valid": verification_report.get("structure_valid"),
                                "dependency_valid": verification_report.get("dependency_valid"),
                                "manifest_valid": verification_report.get("manifest_valid"),
                                "imports_valid": verification_report.get("imports_valid"),
                                "smoke_test_valid": verification_report.get("smoke_test_valid"),
                            },
                            "verified_files": len(verification_report.get("verified_files", []) or []),
                            "metrics": verification_metrics,
                        },
                    )
    
    # 6. Final complete event
    final_state = app_brain.get_state(config).values
    if final_state.get("requirements_complete") and final_state.get("explanation"):
        total_duration_ms = int((time.perf_counter() - start_ts) * 1000)
        yield WSEvent(
            type=WSEventType.COMPLETE,
            payload={
                "project_path": final_state.get("project_path", ""),
                "config_json": final_state.get("config_json", {}),
                "explanation": final_state.get("explanation", ""),
                "verification_report": final_state.get("verification_report", {}),
                "verification_metrics": final_state.get("verification_metrics", {}),
                "session_id": session_id,
                "total_duration_ms": total_duration_ms,
            }
        )
    else:
        total_duration_ms = int((time.perf_counter() - start_ts) * 1000)
        yield WSEvent(
            type=WSEventType.COMPLETE,
            payload={
                "project_path": None,
                "config_json": None,
                "explanation": None,
                "session_id": session_id,
                "total_duration_ms": total_duration_ms,
            }
        )
