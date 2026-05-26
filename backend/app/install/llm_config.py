"""Read per-node models from model_plan.json (written by bootstrap, read by agents)."""
from __future__ import annotations

import os

from app.install.model_plan import load_model_plan

_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")
_DEFAULT_EMBED = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


def get_node_model(node: str) -> str:
    plan = load_model_plan()
    if plan and node in plan.node_assignments:
        asn = plan.node_assignments[node]
        if asn.status == "installed" and asn.model:
            return asn.model
    return _DEFAULT_MODEL


def get_embedding_model() -> str:
    plan = load_model_plan()
    if plan and plan.embedding and plan.embedding.model:
        return plan.embedding.model.split(":")[0]
    return _DEFAULT_EMBED
