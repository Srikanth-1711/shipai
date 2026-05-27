from __future__ import annotations

import json
from pathlib import Path

from app.engine.builder_utils import build_manifest, compute_config_hash, write_manifest
from app.runtime.sandbox import run_sandboxed_command
from app.services.template_engine import ShipAITemplateEngine
from app.verification.project_verifier import verify_project
from app.verification.reality_testing import (
    check_docker,
    check_frontend,
    check_python_fastapi,
    run_reality_tests,
)


def _base_config() -> dict:
    return {
        "template": "rag_chatbot",
        "framework": "langchain",
        "vector_db": "chroma",
        "search_type": "dense",
        "reranker": "none",
        "pii": "none",
        "cache": "none",
        "infra_tier": "minimal",
        "decisions": {},
    }


def _write_manifest(root: Path, config: dict) -> None:
    files = []
    for f in root.rglob("*"):
        if f.is_file() and "__pycache__" not in str(f):
            files.append(str(f.relative_to(root)))
    manifest = build_manifest(
        project_path=root,
        config=config,
        config_hash=compute_config_hash(config),
        shipai_version="test",
        generated_files=files,
        model_plan_summary={},
    )
    write_manifest(root, manifest)


def _generate_project(tmp_path: Path) -> Path:
    cfg = _base_config()
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    ShipAITemplateEngine().generate_project(cfg, output_dir=str(root))
    _write_manifest(root, cfg)
    return root


def test_reality_valid_fastapi_project(tmp_path: Path):
    root = _generate_project(tmp_path)
    report = run_reality_tests(root, _base_config(), plugins=[check_python_fastapi])
    assert report.smoke_test_valid
    assert report.runtime_validation_score >= 70.0


def test_reality_broken_fastapi_missing_app(tmp_path: Path):
    root = _generate_project(tmp_path)
    main_py = root / "backend" / "app" / "main.py"
    main_py.write_text("x = 1\n", encoding="utf-8")
    report = run_reality_tests(root, {}, plugins=[check_python_fastapi])
    assert report.smoke_test_valid is False
    assert any("FastAPI" in e for e in report.errors)


def test_reality_invalid_package_json(tmp_path: Path):
    root = tmp_path / "frontend_proj"
    frontend = root / "frontend"
    frontend.mkdir(parents=True)
    (frontend / "package.json").write_text("{ not json", encoding="utf-8")
    report = run_reality_tests(root, {}, plugins=[check_frontend])
    assert report.smoke_test_valid is False


def test_reality_malformed_dockerfile(tmp_path: Path):
    root = tmp_path / "docker_proj"
    root.mkdir()
    (root / "Dockerfile").write_text("RUN echo hi\n", encoding="utf-8")
    report = run_reality_tests(root, {}, plugins=[check_docker])
    assert report.smoke_test_valid is False
    assert any("FROM" in e for e in report.errors)


def test_sandbox_timeout_behavior():
    result = run_sandboxed_command(
        ["python", "-c", "import time; time.sleep(5)"],
        timeout_seconds=1,
    )
    assert result.timed_out or result.exit_code != 0


def test_verify_project_includes_smoke_fields(tmp_path: Path):
    root = _generate_project(tmp_path)
    report = verify_project(root)
    d = report.to_dict()
    assert "smoke_test_valid" in d
    assert "runtime_validation_score" in d
    assert "semantic_warnings" in d
