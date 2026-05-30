"""
Model negotiator — match discovered models to agent roles using
hardware-constrained scoring.  Matrix scores are estimates unless
live benchmarks have been merged (see benchmark_fetcher.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.install.canonical import normalize_canonical_id
from app.install.capability_fetcher import (
    get_model_entry,
    get_node_requirements,
    rank_catalog_for_hardware,
)
from app.install.model_plan import ModelPlan, NodeAssignment, save_model_plan
from app.install.types import DiscoveredModel, EnvironmentReport, HardwareProfile, LLMRuntime

# Default chat-model node names — passed in from the application layer.
# The install module does NOT depend on these names; they are provided as a
# convenience default.  The application (engine/orchestrator.py, setup/fleet.py)
# owns the canonical list and should pass it explicitly.
DEFAULT_CHAT_NODES: tuple[str, ...] = (
    "interview_node", "research_node", "plan_node", "explain_node",
)
# Backward-compat alias (deprecated — callers should use DEFAULT_CHAT_NODES
# or pass nodes explicitly).
CHAT_NODES = DEFAULT_CHAT_NODES

UNKNOWN_MODEL_BASELINE = {
    "json_reliable": 0.5,
    "reasoning_score": 0.5,
    "code_generation": 0.5,
}


@dataclass
class NegotiationResult:
    node_assignments: Dict[str, NodeAssignment] = field(default_factory=dict)
    models_to_download: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    embedding: Optional[NodeAssignment] = None
    primary_runtime: str = "none"
    runtime_base_url: Optional[str] = None


class ModelNegotiator:
    def __init__(
        self,
        report: EnvironmentReport,
        matrix: dict[str, Any],
        catalog: List[dict[str, Any]] | None = None,
    ):
        self.report = report
        self.matrix = matrix
        self.catalog = catalog or []
        self.hw = report.hardware
        self._budget_vram = self._effective_vram()
        self._budget_ram = max(
            self.hw.ram_available_gb,
            self.hw.ram_total_gb * 0.85,
        )

    def _effective_vram(self) -> float:
        if self.hw.is_apple_silicon and self.hw.unified_memory_gb:
            return max(self.hw.unified_memory_gb * 0.55, 0.5)
        if self.hw.has_gpu:
            return max(self.hw.gpu_vram_free_gb, 0.5)
        return 0.0

    def _primary_runtime(self) -> Tuple[str, Optional[str]]:
        order = ("ollama", "vllm", "lmstudio", "llamacpp", "openai", "groq", "anthropic")
        running = [r for r in self.report.runtimes if r.is_running]
        for name in order:
            for rt in running:
                if rt.name == name or (name == "ollama" and rt.name == "ollama_custom"):
                    return (
                        "ollama" if rt.name == "ollama_custom" else rt.name,
                        rt.base_url,
                    )
        return "none", None

    def _served_chat_models(self) -> List[DiscoveredModel]:
        out: List[DiscoveredModel] = []
        for item in self.report.inventory:
            if not item.is_served:
                continue
            entry = get_model_entry(self.matrix, item.name)
            if entry and entry.get("capabilities", {}).get("embeddings"):
                continue
            out.append(item)
        return out

    def _embedding_model(self) -> Optional[NodeAssignment]:
        for item in self.report.inventory:
            if not item.is_served:
                continue
            entry = get_model_entry(self.matrix, item.name)
            if entry and entry.get("capabilities", {}).get("embeddings"):
                rt, url = self._primary_runtime()
                return NodeAssignment(
                    model=item.name,
                    runtime=item.serving_runtime or rt,
                    base_url=url,
                    status="installed",
                    confidence="installed_verified",
                    quality_score=1.0,
                    reason="Embedding model detected via Ollama/runtime API",
                )
            if "embed" in item.name.lower():
                rt, url = self._primary_runtime()
                return NodeAssignment(
                    model=item.name,
                    runtime=item.serving_runtime or rt,
                    base_url=url,
                    status="installed",
                    confidence="installed_unknown",
                    quality_score=0.8,
                    reason="Name suggests embeddings; not in capability matrix",
                )
        return None

    def _caps_for_inventory_item(self, item: DiscoveredModel) -> Tuple[dict, str]:
        entry = get_model_entry(self.matrix, item.name)
        if entry:
            return entry.get("capabilities", {}), "installed_verified"
        return dict(UNKNOWN_MODEL_BASELINE), "installed_unknown"

    def score_against_requirements(
        self,
        caps: dict[str, Any],
        requirements: dict[str, Any],
        item: DiscoveredModel,
    ) -> float:
        checks = [
            ("minimum_json_reliable", "json_reliable"),
            ("minimum_reasoning", "reasoning_score"),
            ("minimum_code_generation", "code_generation"),
        ]
        score = 0.0
        for req_key, cap_key in checks:
            if req_key not in requirements:
                continue
            minimum = float(requirements[req_key])
            actual = float(caps.get(cap_key, 0))
            if actual < minimum:
                return 0.0
            score += actual

        if requirements.get("needs_tool_calling") and not caps.get("tool_calling"):
            return 0.0

        if requirements.get("needs_chain_of_thought") and not caps.get("chain_of_thought"):
            score *= 0.5

        priority = requirements.get("priority", "quality")
        if priority == "speed" and item.size_gb is not None:
            score += max(0, 0.3 - item.size_gb * 0.05)

        return score

    def score_relaxed(self, caps: dict[str, Any], requirements: dict[str, Any], item: DiscoveredModel) -> float:
        """Score installed models even if below matrix minimums (best-effort)."""
        score = 0.0
        for key in ("json_reliable", "reasoning_score", "code_generation"):
            score += float(caps.get(key, 0))
        if requirements.get("priority") == "speed" and item.size_gb is not None:
            score += max(0, 0.3 - item.size_gb * 0.05)
        return score

    def find_best_installed(self, node: str) -> Optional[NodeAssignment]:
        requirements = get_node_requirements(self.matrix, node)
        if not requirements:
            return None

        candidates: List[Tuple[float, DiscoveredModel, dict, str]] = []
        for item in self._served_chat_models():
            caps, confidence = self._caps_for_inventory_item(item)
            sc = self.score_against_requirements(caps, requirements, item)
            if sc > 0:
                candidates.append((sc, item, caps, confidence))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[0])
        sc, item, _caps, confidence = candidates[0]
        rt_name, base_url = self._primary_runtime()
        reason = (
            "Installed model matched capability matrix"
            if confidence == "installed_verified"
            else "Installed model not in matrix — run shipai benchmark for genuine scores"
        )
        return NodeAssignment(
            model=item.name,
            runtime=item.serving_runtime or rt_name,
            base_url=base_url,
            status="installed",
            confidence=confidence,
            quality_score=round(sc, 3),
            reason=reason,
        )

    def find_best_installed_relaxed(self, node: str) -> Optional[NodeAssignment]:
        """Pick best real installed model even when below matrix minimums."""
        requirements = get_node_requirements(self.matrix, node)
        candidates: List[Tuple[float, DiscoveredModel, str]] = []
        for item in self._served_chat_models():
            caps, confidence = self._caps_for_inventory_item(item)
            sc = self.score_relaxed(caps, requirements, item)
            candidates.append((sc, item, confidence))
        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[0])
        sc, item, confidence = candidates[0]
        rt_name, base_url = self._primary_runtime()
        return NodeAssignment(
            model=item.name,
            runtime=item.serving_runtime or rt_name,
            base_url=base_url,
            status="installed",
            confidence=confidence,
            quality_score=round(sc, 3),
            reason="Best installed model on this machine; below recommended matrix threshold for this node",
        )

    def find_best_downloadable(self, node: str) -> Optional[str]:
        if not self.catalog:
            return None
        ranked = rank_catalog_for_hardware(
            self.catalog,
            self.matrix,
            effective_vram_gb=self._budget_vram,
            effective_ram_gb=self.hw.ram_available_gb,
            ram_total_gb=self.hw.ram_total_gb,
            has_cuda=self.hw.has_cuda,
            top_n=15,
        )
        requirements = get_node_requirements(self.matrix, node)
        installed_ids = {normalize_canonical_id(m.name) for m in self._served_chat_models()}

        best_id: Optional[str] = None
        best_score = 0.0
        for cand in ranked:
            name = cand.get("name", "")
            if normalize_canonical_id(name) in installed_ids:
                continue
            entry = cand.get("capabilities") or get_model_entry(self.matrix, name)
            if not entry:
                continue
            caps = entry.get("capabilities", {})
            fake_item = DiscoveredModel(name=name, source="catalog")
            sc = self.score_against_requirements(caps, requirements, fake_item)
            if sc > best_score:
                best_score = sc
                best_id = entry.get("ollama_id", name)
        return best_id

    def negotiate(self) -> NegotiationResult:
        result = NegotiationResult()
        result.primary_runtime, result.runtime_base_url = self._primary_runtime()
        result.embedding = self._embedding_model()

        if result.primary_runtime == "none":
            result.warnings.append(
                "No local LLM runtime detected. Install Ollama or set OPENAI_API_KEY / GROQ_API_KEY."
            )

        for node in CHAT_NODES:
            installed = self.find_best_installed(node)
            if not installed:
                installed = self.find_best_installed_relaxed(node)
                if installed:
                    result.warnings.append(
                        f"{node}: using {installed.model} (installed; below ideal matrix threshold)"
                    )
            if installed:
                result.node_assignments[node] = installed
                continue

            downloadable = self.find_best_downloadable(node)
            if downloadable:
                result.node_assignments[node] = NodeAssignment(
                    model=downloadable,
                    runtime=result.primary_runtime if result.primary_runtime != "none" else "ollama",
                    base_url=result.runtime_base_url,
                    status="needs_download",
                    confidence="matrix_estimate",
                    quality_score=0.0,
                    reason="No installed model met requirements; suggested from capability matrix",
                )
                if downloadable not in result.models_to_download:
                    result.models_to_download.append(downloadable)
            else:
                result.node_assignments[node] = NodeAssignment(
                    model="",
                    runtime=result.primary_runtime,
                    status="insufficient_hardware",
                    confidence="matrix_estimate",
                    reason="No feasible local model; add cloud API key or upgrade hardware",
                )
                result.warnings.append(
                    f"{node}: no installed or downloadable model fits this machine"
                )

        served_count = len(self._served_chat_models())
        if served_count == 0 and result.primary_runtime != "none":
            result.warnings.append(
                "Runtime is running but no chat models are pulled. Run: ollama pull <model>"
            )

        return result


def negotiate_from_report(
    report: EnvironmentReport,
    matrix: dict[str, Any],
    catalog: List[dict[str, Any]] | None = None,
) -> NegotiationResult:
    return ModelNegotiator(report, matrix, catalog).negotiate()


async def build_and_save_model_plan(
    client=None,
    matrix: dict[str, Any] | None = None,
    catalog: List[dict[str, Any]] | None = None,
) -> ModelPlan:
    from app.install.capability_fetcher import fetch_ollama_catalog, get_capability_matrix
    from app.install.environment import build_environment_report

    report = await build_environment_report(client=client)
    if matrix is None:
        matrix = await get_capability_matrix(client=client)
    if catalog is None:
        catalog = await fetch_ollama_catalog(client=client, matrix_fallback=matrix)

    neg = negotiate_from_report(report, matrix, catalog)
    plan = ModelPlan(
        primary_runtime=neg.primary_runtime,
        runtime_base_url=neg.runtime_base_url,
        embedding=neg.embedding,
        node_assignments=neg.node_assignments,
        models_to_download=neg.models_to_download,
        warnings=neg.warnings,
        metadata={
            "inventory_count": len(report.inventory),
            "served_chat_models": [
                m.name for m in report.inventory if m.is_served and "embed" not in m.name.lower()
            ],
            "matrix_schema": matrix.get("schema_version"),
            "matrix_updated": matrix.get("updated"),
            "genuine_note": "Installed assignments come from live runtime API inventory. "
            "Matrix scores are estimates until benchmarked.",
        },
    )
    save_model_plan(plan)
    return plan
