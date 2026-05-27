"""Use-case / problem statement → scoring weights (no LLM)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class IntentProfile:
    raw: str = ""
    project_type: str = "general"  # rag | agents | code | chat | general
    privacy: str = "standard"  # offline | standard
    scale: str = "solo"  # solo | team | enterprise
    require_embed: bool = False
    weight_fast: float = 0.25
    weight_medium: float = 0.35
    weight_heavy: float = 0.4
    node_weights: dict[str, float] = field(default_factory=dict)


def parse_use_case(text: str | None) -> IntentProfile:
    if not text or not text.strip():
        return IntentProfile(
            node_weights={
                "interview_node": 0.2,
                "research_node": 0.3,
                "plan_node": 0.3,
                "explain_node": 0.2,
            }
        )

    low = text.lower()
    profile = IntentProfile(raw=text.strip())

    if any(w in low for w in ("rag", "document", "pdf", "embedding", "vector", "search")):
        profile.project_type = "rag"
        profile.require_embed = True
        profile.weight_heavy = 0.35
        profile.weight_medium = 0.35
        profile.weight_fast = 0.2
    elif any(w in low for w in ("agent", "multi-agent", "langgraph", "tool")):
        profile.project_type = "agents"
        profile.weight_heavy = 0.45
        profile.weight_medium = 0.35
        profile.weight_fast = 0.2
    elif any(w in low for w in ("code", "coder", "builder", "scaffold")):
        profile.project_type = "code"
        profile.weight_medium = 0.45
        profile.weight_heavy = 0.35
        profile.weight_fast = 0.2
    elif any(w in low for w in ("chat", "chatbot", "conversation")):
        profile.project_type = "chat"
        profile.weight_fast = 0.45
        profile.weight_medium = 0.35
        profile.weight_heavy = 0.2

    if any(w in low for w in ("offline", "private", "hipaa", "on-prem")):
        profile.privacy = "offline"
    if any(w in low for w in ("enterprise", "500", "team", "200 engineer")):
        profile.scale = "enterprise"
    elif any(w in low for w in ("50", "small team", "startup")):
        profile.scale = "team"

    profile.node_weights = {
        "interview_node": profile.weight_fast,
        "explain_node": profile.weight_fast,
        "research_node": profile.weight_heavy,
        "plan_node": profile.weight_heavy if profile.project_type != "code" else profile.weight_medium,
    }
    return profile
