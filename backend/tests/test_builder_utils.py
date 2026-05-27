from __future__ import annotations

import json
from pathlib import Path

from app.engine.builder_utils import (
    canonical_json,
    compute_config_hash,
    deterministic_output_dir,
    build_manifest,
)


def test_canonical_json_stable():
    a = {"b": 2, "a": 1}
    b = {"a": 1, "b": 2}
    assert canonical_json(a) == canonical_json(b)


def test_compute_config_hash_stable():
    cfg = {"template": "rag_chatbot", "infra_tier": "minimal", "decisions": {}}
    h1 = compute_config_hash(cfg)
    h2 = compute_config_hash(dict(reversed(list(cfg.items()))))
    assert h1 == h2
    assert len(h1) == 64


def test_deterministic_output_dir_uses_template_and_hash(tmp_path: Path):
    cfg = {"template": "rag_chatbot", "infra_tier": "minimal", "decisions": {}}
    out = deterministic_output_dir(cfg, base_dir=tmp_path)
    assert out.name.startswith("rag_chatbot_")
    assert out.parent == tmp_path


def test_build_manifest_includes_hashes(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir(parents=True)
    (project / "README.md").write_text("hello")
    manifest = build_manifest(
        project_path=project,
        config={"template": "rag_chatbot", "decisions": {}},
        config_hash="abc",
        shipai_version="0.0.1",
        generated_files=["README.md"],
        model_plan_summary={},
    )
    assert "generated_files" in manifest
    assert "README.md" in manifest["generated_files"]
    assert manifest["generated_files"]["README.md"]["sha256"]
    assert manifest["generated_files"]["README.md"]["size_bytes"] == len("hello")

