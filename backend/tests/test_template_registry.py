from __future__ import annotations

from app.services.template_registry import REGISTRY
from app.services.template_engine import list_templates, get_template


def test_registry_lists_templates():
    templates = REGISTRY.list_templates()
    assert len(templates) >= 3
    assert any(t["id"] == "rag_chatbot" for t in templates)


def test_template_contains_versioned_metadata():
    t = REGISTRY.get_template("rag_chatbot")
    assert t is not None
    assert t["version"]
    assert "capabilities" in t
    assert "dependencies" in t


def test_template_engine_uses_registry():
    templates = list_templates()
    assert any(t["id"] == "multi_agent" for t in templates)
    details = get_template("multi_agent")
    assert details is not None
    assert details["name"] == "Multi-Agent System"

