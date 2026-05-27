from __future__ import annotations

import json
from pathlib import Path

from app.engine.builder_utils import (
    build_manifest,
    compute_config_hash,
    deterministic_output_dir,
    safe_remove_output_dir,
    write_manifest,
)
from app.services.template_engine import ShipAITemplateEngine


def _config() -> dict:
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


def _collect_rel_files(root: Path) -> list[str]:
    files = []
    for f in root.rglob("*"):
        if f.is_file() and "__pycache__" not in str(f):
            files.append(str(f.relative_to(root)))
    files.sort()
    return files


def _generate_with_manifest(base_dir: Path, config: dict) -> tuple[Path, dict]:
    out = deterministic_output_dir(config, base_dir=base_dir)
    safe_remove_output_dir(out)
    engine = ShipAITemplateEngine()
    engine.generate_project(config, output_dir=str(out))

    files = _collect_rel_files(out)
    manifest = build_manifest(
        project_path=out,
        config=config,
        config_hash=compute_config_hash(config),
        shipai_version="test",
        generated_files=files,
        model_plan_summary={},
    )
    write_manifest(out, manifest)
    return out, manifest


def test_same_config_same_output_dir(tmp_path: Path):
    cfg = _config()
    a = deterministic_output_dir(cfg, base_dir=tmp_path)
    b = deterministic_output_dir(cfg, base_dir=tmp_path)
    assert a == b


def test_snapshot_reproducibility_same_hashes(tmp_path: Path):
    cfg = _config()
    _, m1 = _generate_with_manifest(tmp_path, cfg)
    _, m2 = _generate_with_manifest(tmp_path, cfg)

    # Ignore timestamp, compare stable parts.
    keys = ["config_hash", "config_snapshot", "generated_files", "shipai_version", "manifest_version"]
    s1 = {k: m1.get(k) for k in keys}
    s2 = {k: m2.get(k) for k in keys}
    assert s1 == s2


def test_manifest_lists_existing_files(tmp_path: Path):
    cfg = _config()
    root, manifest = _generate_with_manifest(tmp_path, cfg)
    for rel in manifest["generated_files"].keys():
        assert (root / rel).exists()

