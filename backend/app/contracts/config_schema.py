"""Config schema for ShipAI planner → builder contract.

This is the IR between `plan_node` and `builder_node` / `template_engine`.

Important: this module must be importable even in lightweight bootstrap/test
environments where `pydantic` may not be installed. We therefore implement a
fallback validator when pydantic isn't available.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar

try:
    # Optional dependency in some lightweight test environments.
    from pydantic import BaseModel, Field, ValidationError, validator  # type: ignore

    _HAVE_PYDANTIC = True
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore
    Field = None  # type: ignore
    ValidationError = Exception  # type: ignore
    validator = None  # type: ignore
    _HAVE_PYDANTIC = False


TEMPLATES = ["rag_chatbot", "multi_agent", "data_analyzer", "fine_tuner"]
FRAMEWORKS = ["langgraph", "llamaindex", "langchain", "haystack"]
VECTOR_DBS = ["chroma", "pgvector", "milvus", "pinecone"]
SEARCH_TYPES = ["dense", "sparse", "hybrid"]
RERANKERS = ["none", "bge", "cohere"]
PII_MODES = ["none", "regex", "presidio"]
CACHES = ["none", "redis", "semantic"]
INFRA_TIERS = ["minimal", "standard", "enterprise"]

ALLOWED_TOP_KEYS = {
    "version",
    "project_type",
    "template",
    "framework",
    "vector_db",
    "search_type",
    "reranker",
    "pii",
    "cache",
    "infra_tier",
    "decisions",
}


if _HAVE_PYDANTIC:
    class ConfigV1Base(BaseModel):  # type: ignore[misc]
        """Pydantic-based schema when available."""

        version: Literal["1.0"] = "1.0"
        project_type: str = "rag_chatbot"

        template: str
        framework: str
        vector_db: str
        search_type: str
        reranker: str
        pii: str
        cache: str
        infra_tier: str

        decisions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # type: ignore[assignment]

        class Config:  # type: ignore[valid-type]
            extra = "forbid"

        @validator(
            "template",
            "framework",
            "vector_db",
            "search_type",
            "reranker",
            "pii",
            "cache",
            "infra_tier",
        )
        def _non_empty_and_allowed(cls, v: str, field):  # type: ignore[override]
            if not isinstance(v, str) or not v.strip():
                raise ValueError("must be a non-empty string")
            v = v.strip()
            allowed_map = {
                "template": TEMPLATES,
                "framework": FRAMEWORKS,
                "vector_db": VECTOR_DBS,
                "search_type": SEARCH_TYPES,
                "reranker": RERANKERS,
                "pii": PII_MODES,
                "cache": CACHES,
                "infra_tier": INFRA_TIERS,
            }
            key = field.name
            if key in allowed_map and v not in allowed_map[key]:
                raise ValueError(f"invalid option '{v}' for {key}")
            return v

    ConfigV1 = ConfigV1Base  # type: ignore
else:
    # Placeholder for type-checkers; validation runs via fallback path.
    ConfigV1 = None  # type: ignore


def _fallback_validate(raw: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    if not isinstance(raw, dict):
        return None, ["config must be a dict"]

    extra = sorted(k for k in raw.keys() if k not in ALLOWED_TOP_KEYS)
    if extra:
        return None, [f"extra fields not permitted: {', '.join(extra)}"]

    # Required inputs from planner → builder contract.
    required = ["template", "framework", "vector_db", "search_type", "reranker", "pii", "cache", "infra_tier", "decisions"]
    # `decisions` may come as {}, but must exist.
    for k in required:
        if k not in raw:
            errors.append(f"{k}: field required")

    if errors:
        return None, errors

    # Non-empty strings + allowed enum checks.
    enums = {
        "template": TEMPLATES,
        "framework": FRAMEWORKS,
        "vector_db": VECTOR_DBS,
        "search_type": SEARCH_TYPES,
        "reranker": RERANKERS,
        "pii": PII_MODES,
        "cache": CACHES,
        "infra_tier": INFRA_TIERS,
    }
    out: Dict[str, Any] = dict(raw)
    out.setdefault("project_type", "rag_chatbot")
    out.setdefault("version", "1.0")

    for k, allowed in enums.items():
        v = out.get(k)
        if not isinstance(v, str) or not v.strip():
            errors.append(f"{k}: must be a non-empty string")
            continue
        v = v.strip()
        if v not in allowed:
            errors.append(f"{k}: invalid option '{v}'")

    # Ensure decisions is dict[str, dict]
    decisions = out.get("decisions")
    if not isinstance(decisions, dict):
        errors.append("decisions: must be an object")
    else:
        for dk, dv in decisions.items():
            if not isinstance(dk, str) or not isinstance(dv, dict):
                errors.append("decisions: each decision must be a dict")
                break

    if errors:
        return None, errors
    return out, []


def validate_config(raw: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Validate a raw config dict.

    Returns: (validated_config | None, errors)
    """
    if not _HAVE_PYDANTIC:
        return _fallback_validate(raw)

    try:
        cfg = ConfigV1(**raw)  # type: ignore[misc]
        # pydantic v1 compat
        data = cfg.dict()
        data["version"] = "1.0"
        return data, []
    except Exception as exc:
        if isinstance(exc, ValidationError):  # type: ignore[truthy-function]
            errors: List[str] = []
            for err in exc.errors():
                loc = ".".join(str(p) for p in err.get("loc", []))
                msg = err.get("msg", "")
                errors.append(f"{loc}: {msg}")
            return None, errors
        return None, [f"validation error: {str(exc)}"]

