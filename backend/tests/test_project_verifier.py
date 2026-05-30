from __future__ import annotations

from pathlib import Path

from app.engine.builder_utils import (
    compute_config_hash,
    build_manifest,
    write_manifest,
)
from app.services.template_engine import ShipAITemplateEngine
from app.verification.project_verifier import verify_project


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


def _list_rel_files(root: Path) -> list[str]:
    files: list[str] = []
    for f in root.rglob("*"):
        if f.is_file() and "__pycache__" not in str(f):
            files.append(str(f.relative_to(root)))
    return files


def _write_manifest(root: Path, config: dict, *, shipai_version: str = "test") -> None:
    files = _list_rel_files(root)
    config_hash = compute_config_hash(config)
    manifest = build_manifest(
        project_path=root,
        config=config,
        config_hash=config_hash,
        shipai_version=shipai_version,
        generated_files=files,
        model_plan_summary={},
    )
    write_manifest(root, manifest)


def test_verifier_valid_project(tmp_path: Path):
    config = _base_config()
    engine = ShipAITemplateEngine()
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    engine.generate_project(config, output_dir=str(root))
    _write_manifest(root, config)

    report = verify_project(root)
    assert report.syntax_valid
    assert report.structure_valid
    assert report.dependency_valid
    assert report.manifest_valid
    assert report.smoke_test_valid
    assert report.failed is False


def test_verifier_syntax_error_project(tmp_path: Path):
    config = _base_config()
    engine = ShipAITemplateEngine()
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    engine.generate_project(config, output_dir=str(root))

    # Corrupt one generated python file (create a SyntaxError).
    target = root / "backend" / "app" / "main.py"
    text = target.read_text(encoding="utf-8", errors="replace")
    target.write_text(text + "\nthis is not valid python\n", encoding="utf-8")
    _write_manifest(root, config)  # update hashes to isolate syntax failure

    report = verify_project(root)
    assert report.syntax_valid is False
    assert any("syntax error" in e.lower() or "compile error" in e.lower() for e in report.errors)


def test_verifier_missing_manifest(tmp_path: Path):
    config = _base_config()
    engine = ShipAITemplateEngine()
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    engine.generate_project(config, output_dir=str(root))

    report = verify_project(root)
    assert report.manifest_valid is False
    assert any("manifest" in e.lower() or "required file missing" in e.lower() for e in report.errors)


def test_verifier_hash_mismatch(tmp_path: Path):
    config = _base_config()
    engine = ShipAITemplateEngine()
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    engine.generate_project(config, output_dir=str(root))
    _write_manifest(root, config)

    # Mutate a file without updating manifest hashes.
    target = root / "README.md"
    target.write_text(target.read_text(encoding="utf-8", errors="replace") + "\nCHANGED", encoding="utf-8")

    report = verify_project(root)
    assert report.manifest_valid is False
    assert any("sha256 mismatch" in e.lower() for e in report.errors)


def test_verifier_malformed_requirements(tmp_path: Path):
    config = _base_config()
    engine = ShipAITemplateEngine()
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    engine.generate_project(config, output_dir=str(root))

    (root / "requirements.txt").write_text("this is not a valid requirement line !!!!\n", encoding="utf-8")
    _write_manifest(root, config)

    report = verify_project(root)
    assert report.dependency_valid is False
    assert any("requirements.txt" in e.lower() for e in report.errors)


def test_verifier_partial_success_missing_requirements(tmp_path: Path):
    config = _base_config()
    engine = ShipAITemplateEngine()
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    engine.generate_project(config, output_dir=str(root))
    _write_manifest(root, config)

    # Remove requirements to cause structure/dependency issues.
    (root / "requirements.txt").unlink()

    report = verify_project(root)
    assert report.structure_valid is False or report.dependency_valid is False

