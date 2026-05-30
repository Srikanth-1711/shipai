"""
Enhanced Model Negotiator with User Suggestions + Optional Gemini Reasoning.
Wraps deterministic scoring with optional LLM validation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.install.model_negotiator import (
    negotiate_from_report,
    DEFAULT_CHAT_NODES,
)
from app.install.model_plan import ModelPlan, NodeAssignment
from app.install.types import EnvironmentReport
from app.services.gemini_service import gemini_service

logger = logging.getLogger("shipai.enhanced_negotiator")


class EnhancedNegotiator:
    """
    Negotiator that:
    1. Uses deterministic scoring (base behavior)
    2. Optionally uses Gemini to explain choices
    3. Accepts user model suggestions
    4. Validates suggestions against hardware + capabilities
    """

    def __init__(
        self,
        report: EnvironmentReport,
        matrix: dict[str, Any],
        catalog: List[dict[str, Any]] | None = None,
        use_gemini: bool = False,
        nodes: tuple[str, ...] | None = None,
    ):
        self.report = report
        self.matrix = matrix
        self.catalog = catalog or []
        self.use_gemini = use_gemini and gemini_service.enabled
        self.nodes: tuple[str, ...] = tuple(nodes) if nodes else DEFAULT_CHAT_NODES
        self.user_suggestions: Dict[str, str] = {}

    def set_user_suggestions(self, suggestions: Dict[str, str]) -> None:
        """
        User suggests models for specific nodes.
        Format: {"interview_node": "qwen3:4b", "research_node": "deepseek-r1:7b"}
        """
        self.user_suggestions = suggestions
        logger.info(f"User suggestions registered: {suggestions}")

    async def negotiate(self) -> ModelPlan:
        """
        Enhanced negotiation:
        1. Get deterministic assignment from base negotiator
        2. Apply user suggestions if provided
        3. Optionally validate with Gemini
        4. Return enhanced plan
        """

        # Step 1: Get base deterministic assignment
        base_result = negotiate_from_report(
            self.report, self.matrix, self.catalog, nodes=self.nodes
        )

        # Step 2: Create initial plan
        plan = ModelPlan(
            primary_runtime=base_result.primary_runtime,
            runtime_base_url=base_result.runtime_base_url,
            embedding=base_result.embedding,
            node_assignments=base_result.node_assignments.copy(),
            models_to_download=list(base_result.models_to_download),
            warnings=list(base_result.warnings),
            gemini_enabled=self.use_gemini,
            metadata={"enhanced_negotiator": True},
        )

        # Step 3: Apply user suggestions
        if self.user_suggestions:
            plan = await self._apply_user_suggestions(plan)

        # Step 4: Optional Gemini reasoning
        if self.use_gemini:
            plan = await self._add_gemini_reasoning(plan)

        return plan

    async def _apply_user_suggestions(self, plan: ModelPlan) -> ModelPlan:
        """
        User suggested specific models. Validate and apply them.
        """
        hw = self.report.hardware

        for node, suggested_model in self.user_suggestions.items():
            if node not in self.nodes and node != "embedding":
                logger.warning(f"Unknown node: {node}")
                continue

            # Validate suggestion
            validation = await gemini_service.validate_user_model_suggestion(
                suggested_model=suggested_model,
                hardware_specs={
                    "ram_gb": hw.ram_total_gb,
                    "gpu": hw.gpu_name or "none",
                    "gpu_vram_gb": hw.gpu_vram_free_gb,
                },
                available_models=list(self.matrix.get("models", {}).keys()),
            ) if self.use_gemini else {"valid": True, "reasoning": "Validation skipped"}

            if validation.get("valid", True):
                # Apply user suggestion
                plan.node_assignments[node] = NodeAssignment(
                    model=suggested_model,
                    runtime=plan.primary_runtime,
                    base_url=plan.runtime_base_url,
                    status="user_suggested",
                    confidence="user_override",
                    quality_score=0.0,
                    reason=f"User suggested: {suggested_model}",
                    user_suggested=True,
                )
                plan.user_suggestions[node] = suggested_model
                logger.info(f"Applied user suggestion for {node}: {suggested_model}")
            else:
                # Reject suggestion
                concern = validation.get("concerns", ["Not suitable for hardware"])[0]
                plan.warnings.append(
                    f"User suggestion for {node} ({suggested_model}) rejected: {concern}"
                )
                logger.warning(f"Rejected user suggestion: {suggested_model} - {concern}")

        return plan

    async def _add_gemini_reasoning(self, plan: ModelPlan) -> ModelPlan:
        """
        Add Gemini explanations for each node assignment.
        Does NOT change assignments, just enriches with reasoning.
        """
        hw = self.report.hardware
        hw_specs = {
            "ram_gb": hw.ram_total_gb,
            "ram_available_gb": hw.ram_available_gb,
            "gpu": hw.gpu_name or "none",
            "gpu_vram_gb": hw.gpu_vram_free_gb,
            "effective_vram_gb": getattr(hw, "effective_vram_gb", 0),
        }

        for node, assignment in plan.node_assignments.items():
            if not assignment.model:
                continue

            # Get alternatives (other models that could work)
            alternatives = [
                m
                for m in self.matrix.get("models", {}).keys()
                if m != assignment.model
            ][:3]

            # Request Gemini explanation
            result = await gemini_service.explain_model_choice(
                hardware_specs=hw_specs,
                selected_model=assignment.model,
                alternatives=alternatives,
                use_case=node,
            )

            if result.get("explanation"):
                assignment.gemini_reasoning = result["explanation"]
                logger.info(f"Added Gemini reasoning for {node}: {assignment.model}")

        return plan


async def negotiate_enhanced(
    report: EnvironmentReport,
    matrix: dict[str, Any],
    catalog: List[dict[str, Any]] | None = None,
    user_suggestions: Dict[str, str] | None = None,
    use_gemini: bool = False,
    nodes: tuple[str, ...] | None = None,
) -> ModelPlan:
    """
    Top-level function for enhanced negotiation with user suggestions + Gemini.
    """
    negotiator = EnhancedNegotiator(
        report=report,
        matrix=matrix,
        catalog=catalog,
        use_gemini=use_gemini,
        nodes=nodes,
    )

    if user_suggestions:
        negotiator.set_user_suggestions(user_suggestions)

    return await negotiator.negotiate()
