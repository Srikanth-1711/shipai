"""Tests for the planner → builder config schema and validator.

These tests validate the contract shape independently of whether `pydantic`
is available; the schema module includes a fallback validator.
"""
from __future__ import annotations

from app.contracts.config_schema import validate_config


def _minimal_config() -> dict:
    return {
        "template": "rag_chatbot",
        "framework": "langchain",
        "vector_db": "chroma",
        "search_type": "hybrid",
        "reranker": "none",
        "pii": "none",
        "cache": "none",
        "infra_tier": "minimal",
        "decisions": {},
    }


def test_config_v1_accepts_valid_minimal():
    raw = _minimal_config()
    cfg, errors = validate_config(raw)
    assert errors == []
    assert cfg["template"] == "rag_chatbot"
    assert cfg["version"] == "1.0"


def test_config_v1_rejects_missing_required_fields():
    raw = _minimal_config()
    raw.pop("template")
    cfg, errors = validate_config(raw)
    assert cfg is None
    # At least one error mentioning template being required
    assert any("template" in e for e in errors)


def test_config_v1_rejects_empty_strings():
    raw = _minimal_config()
    raw["framework"] = "   "
    cfg, errors = validate_config(raw)
    assert cfg is None
    assert any("framework" in e for e in errors)


def test_config_v1_forbids_extra_fields():
    raw = _minimal_config()
    raw["unexpected"] = "x"
    cfg, errors = validate_config(raw)
    assert cfg is None
    assert any("extra fields not permitted" in e for e in errors)

