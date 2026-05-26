"""Persist negotiated model assignments — agents read this, bootstrap writes it."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PLAN_PATH = Path.home() / ".shipai" / "model_plan.json"
SCHEMA_VERSION = "1"


@dataclass
class NodeAssignment:
    model: str
    runtime: str
    base_url: Optional[str] = None
    status: str = "installed"  # installed | needs_download | cloud_required | insufficient_hardware
    confidence: str = "installed_verified"  # installed_verified | installed_unknown | matrix_estimate
    quality_score: float = 0.0
    reason: str = ""


@dataclass
class ModelPlan:
    schema_version: str = SCHEMA_VERSION
    generated_at: str = ""
    primary_runtime: str = "none"
    runtime_base_url: Optional[str] = None
    embedding: Optional[NodeAssignment] = None
    node_assignments: Dict[str, NodeAssignment] = field(default_factory=dict)
    models_to_download: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["node_assignments"] = {
            k: asdict(v) if isinstance(v, NodeAssignment) else v
            for k, v in self.node_assignments.items()
        }
        if self.embedding:
            d["embedding"] = asdict(self.embedding)
        return d


def save_model_plan(plan: ModelPlan, path: Path | None = None) -> Path:
    path = path or PLAN_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if not plan.generated_at:
        plan.generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with path.open("w", encoding="utf-8") as f:
        json.dump(plan.to_dict(), f, indent=2)
    return path


def load_model_plan(path: Path | None = None) -> ModelPlan | None:
    path = path or PLAN_PATH
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    nodes = {}
    for k, v in raw.get("node_assignments", {}).items():
        nodes[k] = NodeAssignment(**v)
    emb = raw.get("embedding")
    return ModelPlan(
        schema_version=raw.get("schema_version", "1"),
        generated_at=raw.get("generated_at", ""),
        primary_runtime=raw.get("primary_runtime", "none"),
        runtime_base_url=raw.get("runtime_base_url"),
        embedding=NodeAssignment(**emb) if emb else None,
        node_assignments=nodes,
        models_to_download=list(raw.get("models_to_download", [])),
        warnings=list(raw.get("warnings", [])),
        metadata=dict(raw.get("metadata", {})),
    )
