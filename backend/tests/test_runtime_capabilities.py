from __future__ import annotations

from app.runtime.capabilities import get_runtime_capabilities, summarize_capabilities


def test_runtime_capabilities_shape():
    caps = get_runtime_capabilities()
    assert "required" in caps
    assert "optional" in caps
    assert isinstance(caps["required"], list)
    assert isinstance(caps["optional"], list)


def test_runtime_summary_fields():
    s = summarize_capabilities()
    assert "required_ok" in s
    assert "optional_available" in s
    assert "optional_total" in s
    assert "details" in s

