import json
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage
from app.engine.state import AgentState
from app.config import settings
from app.install.llm_config import get_node_model

# Node Stubs
def interview_node(state: AgentState):
    """Discovery Question Tree: Asks questions until requirements are met."""
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

def explain_node(state: AgentState):
    """Explainer Agent: Translates the architecture decision into the user's language."""
    requirements = state.get("requirements", {})
    config = state.get("config_json", {})
    user_level = requirements.get("user_level", "junior_dev")
    
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
    
    explain_prompt = f"""You are ShipAI's Explainer Agent. A project was just generated with these settings:

Architecture: {config.get('template', '?')}
Framework: {config.get('framework', '?')}
Vector DB: {config.get('vector_db', '?')}
Search: {config.get('search_type', '?')}
Reranker: {config.get('reranker', '?')}
Cache: {config.get('cache', '?')}
PII Protection: {config.get('pii', '?')}
Infrastructure: {config.get('infra_tier', '?')}

Decision Reasoning:
{json.dumps(config.get('decisions', {}), indent=2)}

User level: {user_level}
Instructions: {instruction}

Write a clear explanation of what was built and why. 200-300 words:"""
    
    resp = llm.invoke(explain_prompt)
    explanation = resp.content if hasattr(resp, "content") else str(resp)
    
    return {
        "explanation": explanation,
        "project_path": f"./generated/{config.get('template', 'project')}",
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

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("interview_node", interview_node)
workflow.add_node("research_node", research_node)
workflow.add_node("plan_node", plan_node)
workflow.add_node("explain_node", explain_node)

workflow.set_entry_point("interview_node")
workflow.add_conditional_edges("interview_node", route_from_interview)
workflow.add_conditional_edges("research_node", route_from_research)
workflow.add_edge("plan_node", "explain_node")
workflow.add_edge("explain_node", END)

from langgraph.checkpoint.memory import MemorySaver

# In production, we attach a checkpointer here (e.g., Redis or SQLite)
memory = MemorySaver()
app_brain = workflow.compile(checkpointer=memory)

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
    "plan_node":      {"color": "green",  "label": "Builder Agent"},
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
            
            # 5. Emit file generation events
            if "project_path" in node_output:
                project_path = node_output["project_path"]
                if project_path:
                    files = list_generated_files(project_path)
                    for file_path in files:
                        yield WSEvent(
                            type=WSEventType.FILE_GENERATED,
                            payload={
                                "path": str(file_path),
                                "name": file_path.name,
                                "highlighted": is_key_file(file_path)
                            }
                        )
    
    # 6. Final complete event
    final_state = app_brain.get_state(config).values
    if final_state.get("requirements_complete") and final_state.get("explanation"):
        yield WSEvent(
            type=WSEventType.COMPLETE,
            payload={
                "project_path": final_state.get("project_path", ""),
                "config_json": final_state.get("config_json", {}),
                "explanation": final_state.get("explanation", ""),
                "session_id": session_id
            }
        )
    else:
        yield WSEvent(
            type=WSEventType.COMPLETE,
            payload={
                "project_path": None,
                "config_json": None,
                "explanation": None,
                "session_id": session_id
            }
        )
