"""Tests for builder_node logic — isolated from langgraph/langchain deps."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


# ---- unit: pure builder logic (no langgraph import needed) ----

def _run_builder(state: dict) -> dict:
    """Inline the builder logic so tests don't import orchestrator (langgraph dep)."""
    config = state.get("config_json", {})
    if not config:
        return {
            "project_path": "",
            "generated_files": [],
            "build_error": "No config_json from plan_node - cannot generate project.",
        }
    try:
        from app.services.template_engine import ShipAITemplateEngine
        engine = ShipAITemplateEngine()
        project_path = engine.generate_project(config)
        root = Path(project_path)
        files = []
        for f in root.rglob("*"):
            if f.is_file() and "__pycache__" not in str(f):
                files.append(str(f.relative_to(root)))
        return {"project_path": project_path, "generated_files": files, "build_error": ""}
    except Exception as exc:
        return {"project_path": "", "generated_files": [], "build_error": str(exc)}


def _config():
    return {
        "template": "rag_chatbot",
        "framework": "fastapi",
        "vector_db": "chroma",
        "search_type": "hybrid",
        "reranker": "none",
        "pii": "none",
        "cache": "none",
        "infra_tier": "basic",
        "decisions": {"framework": {"choice": "fastapi", "reason": "lightweight"}},
    }


def test_builder_generates_project(tmp_path):
    fake_project = tmp_path / "rag_chatbot_test"
    fake_project.mkdir()
    (fake_project / "main.py").write_text("# main")
    (fake_project / "requirements.txt").write_text("fastapi")
    sub = fake_project / "app"
    sub.mkdir()
    (sub / "retriever.py").write_text("# retriever")

    with patch("app.services.template_engine.ShipAITemplateEngine.generate_project",
               return_value=str(fake_project)):
        result = _run_builder({"config_json": _config()})

    assert result["build_error"] == ""
    assert result["project_path"] == str(fake_project)
    names = [Path(f).name for f in result["generated_files"]]
    assert "main.py" in names
    assert "requirements.txt" in names
    assert "retriever.py" in names


def test_builder_no_config():
    result = _run_builder({"config_json": {}})
    assert result["build_error"] != ""
    assert result["project_path"] == ""
    assert result["generated_files"] == []


def test_builder_engine_exception():
    with patch("app.services.template_engine.ShipAITemplateEngine.generate_project",
               side_effect=RuntimeError("disk full")):
        result = _run_builder({"config_json": _config()})
    assert "disk full" in result["build_error"]
    assert result["generated_files"] == []


def test_builder_real_template_engine(tmp_path):
    """Integration: real template_engine generates files on disk."""
    from app.services.template_engine import ShipAITemplateEngine
    engine = ShipAITemplateEngine()
    project_path = engine.generate_project(_config(), output_dir=str(tmp_path / "proj"))
    root = Path(project_path)
    assert root.exists()
    files = list(root.rglob("*.py")) + list(root.rglob("*.txt")) + list(root.rglob("*.md"))
    assert len(files) > 0, "Template engine should write at least one file"
