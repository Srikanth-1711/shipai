"""Utilities for deterministic project building and manifest generation.

This module is intentionally free of `langgraph` / `langchain` imports so it can
be unit-tested in lightweight environments.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import shutil


def canonical_json(data: Any) -> str:
    """Stable JSON for hashing (sorted keys, no whitespace)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_config_hash(config: Dict[str, Any]) -> str:
    # Hash the full config snapshot (post-validation ideally).
    return sha256_hex(canonical_json(config))


def deterministic_output_dir(config: Dict[str, Any], *, base_dir: Path | None = None) -> Path:
    base = base_dir or (Path.home() / ".shipai" / "projects")
    template = str(config.get("template") or config.get("project_type") or "project")
    h = compute_config_hash(config)[:12]
    return base / f"{template}_{h}"


def safe_remove_output_dir(path: Path) -> None:
    """Remove output dir if it looks like a ShipAI generated project."""
    if not path.exists():
        return
    # Only remove directories that contain our expected sub-structure.
    if (path / "backend").exists() and (path / "eval").exists():
        shutil.rmtree(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    *,
    project_path: Path,
    config: Dict[str, Any],
    config_hash: str,
    shipai_version: str,
    generated_files: List[str],
    model_plan_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    files = {}
    for rel_path in generated_files:
        fp = project_path / rel_path
        if fp.is_file():
            files[rel_path] = {
                "sha256": sha256_file(fp),
                "size_bytes": fp.stat().st_size,
            }

    manifest = {
        "manifest_version": "1.0",
        "shipai_version": shipai_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "config_snapshot": config,
        "model_plan_summary": model_plan_summary or {},
        "generated_files": files,
    }
    return manifest


def write_manifest(project_path: Path, manifest: Dict[str, Any]) -> None:
    (project_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

