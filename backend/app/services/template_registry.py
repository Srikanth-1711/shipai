"""Versioned template registry for ShipAI templates/plugins."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class TemplateCapability:
    key: str
    supported_values: List[str] = field(default_factory=list)


@dataclass
class TemplateMetadata:
    template_id: str
    name: str
    description: str
    category: str
    icon: str
    tier: str
    version: str = "1.0.0"
    min_shipai_version: str = "0.1.0"
    inputs: List[str] = field(default_factory=list)
    capabilities: List[TemplateCapability] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    compatible_with: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["id"] = self.template_id
        return d


class TemplateRegistry:
    def __init__(self) -> None:
        self._templates: Dict[str, TemplateMetadata] = {}
        self._load_builtin_templates()

    def _register(self, meta: TemplateMetadata) -> None:
        self._templates[meta.template_id] = meta

    def _load_builtin_templates(self) -> None:
        common_caps = [
            TemplateCapability("framework", ["langgraph", "llamaindex", "langchain", "haystack"]),
            TemplateCapability("vector_db", ["chroma", "pgvector", "milvus", "pinecone"]),
            TemplateCapability("search_type", ["dense", "sparse", "hybrid"]),
            TemplateCapability("reranker", ["none", "bge", "cohere"]),
            TemplateCapability("cache", ["none", "redis", "semantic"]),
            TemplateCapability("pii", ["none", "regex", "presidio"]),
            TemplateCapability("infra_tier", ["minimal", "standard", "enterprise"]),
        ]

        self._register(
            TemplateMetadata(
                template_id="rag_chatbot",
                name="RAG Chatbot",
                description="Production-ready chatbot powered by RAG.",
                category="rag",
                icon="📚",
                tier="free",
                inputs=["project_name"],
                capabilities=common_caps,
                dependencies=["fastapi", "uvicorn"],
                compatible_with={"deployment": ["local", "docker"]},
            )
        )
        self._register(
            TemplateMetadata(
                template_id="multi_agent",
                name="Multi-Agent System",
                description="Multi-agent orchestration flow.",
                category="agent",
                icon="🤖",
                tier="starter",
                inputs=["project_name"],
                capabilities=common_caps,
                dependencies=["langgraph", "langchain"],
                compatible_with={"deployment": ["local", "docker"]},
            )
        )
        self._register(
            TemplateMetadata(
                template_id="data_analyzer",
                name="Data Analyzer",
                description="Log analysis and metrics RAG analyzer.",
                category="data",
                icon="📊",
                tier="pro",
                inputs=["project_name"],
                capabilities=common_caps,
                dependencies=["pandas", "fastapi"],
                compatible_with={"deployment": ["local", "docker"]},
            )
        )

    def list_templates(self, tier: str | None = None) -> List[dict]:
        out = [meta.to_dict() for meta in self._templates.values()]
        if tier:
            out = [t for t in out if t.get("tier") == tier]
        return out

    def get_template(self, template_id: str) -> Optional[dict]:
        meta = self._templates.get(template_id)
        return meta.to_dict() if meta else None

    def exists(self, template_id: str) -> bool:
        return template_id in self._templates


REGISTRY = TemplateRegistry()

