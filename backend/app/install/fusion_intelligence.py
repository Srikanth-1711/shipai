"""
Unified fusion intelligence — CEO + web + hardware math + use-case → model plan.
Zero LLM in the decision path. Optional Gemini explain after.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import httpx

from app.install.capability_fetcher import get_capability_matrix
from app.install.ceo_policy import load_ceo_policy
from app.install.environment import build_environment_report
from app.install.intent_profile import parse_use_case
from app.install.model_plan import ModelPlan, save_model_plan
from app.install.tools.model_scorer import build_plan_from_rank, rank_models
from app.install.tools.web_intelligence import fetch_all_intelligence

logger = logging.getLogger("shipai.fusion")

ProgressCallback = Optional[Callable[[str, str], None]]


def _emit(cb: ProgressCallback, agent: str, msg: str) -> None:
    if cb:
        cb(agent, msg)
    logger.info("[%s] %s", agent, msg)


async def run_fusion_intelligence(
    *,
    use_case: str | None = None,
    offline: bool = False,
    use_gemini_explain: bool = False,
    client: httpx.AsyncClient | None = None,
    on_progress: ProgressCallback = None,
) -> ModelPlan:
    _emit(on_progress, "Scout", "Scanning hardware, runtimes, inventory...")
    report = await build_environment_report(client=client)

    hw = report.hardware
    _emit(
        on_progress,
        "Scout",
        f"GPU {hw.gpu_name or 'CPU'} | {hw.effective_vram_gb}GB VRAM budget | {hw.ram_total_gb}GB RAM",
    )

    _emit(on_progress, "Librarian", "CEO policy + matrix + web catalogs...")
    policy = load_ceo_policy()
    matrix = await get_capability_matrix(client=client)
    candidates = await fetch_all_intelligence(matrix, client=client, offline=offline)
    _emit(on_progress, "Librarian", f"{len(candidates)} candidates")

    intent = parse_use_case(use_case)
    if intent.raw:
        _emit(on_progress, "Intent", f"{intent.project_type} | embed={intent.require_embed}")

    _emit(on_progress, "Ranker", "Fusion score -> top 10 -> top 3 roles...")
    rank = rank_models(candidates, report, matrix, intent, policy)
    plan = build_plan_from_rank(rank, report, matrix, intent, gemini_enabled=use_gemini_explain)

    if not rank.top_10:
        plan.warnings.append("No feasible models - check hardware or relax CEO policy")

    for i, s in enumerate(rank.top_10[:10], 1):
        _emit(
            on_progress,
            "Ranker",
            f"  #{i} {s.model.ollama_id} score={s.final_score} role={s.recommended_role}",
        )

    _emit(
        on_progress,
        "Ranker",
        f"Final 3: fast={rank.final_three.get('fast')} | "
        f"medium={rank.final_three.get('medium')} | heavy={rank.final_three.get('heavy')}",
    )

    if use_gemini_explain:
        plan = await _add_gemini_explanations(plan)

    save_model_plan(plan)
    _emit(on_progress, "Validator", "Saved model_plan.json")
    return plan


async def _add_gemini_explanations(plan: ModelPlan) -> ModelPlan:
    try:
        from app.services.gemini_service import gemini_service
    except Exception:
        return plan

    if not gemini_service.enabled:
        plan.warnings.append("Gemini explain skipped — set SHIPAI_GEMINI_API_KEY")
        return plan

    hw_specs = plan.metadata.get("hardware", {})
    for node, asn in plan.node_assignments.items():
        if not asn.model:
            continue
        alts = [m["model"] for m in plan.top10_candidates[:4] if m.get("model") != asn.model]
        result = await gemini_service.explain_model_choice(
            hardware_specs=hw_specs,
            selected_model=asn.model,
            alternatives=alts,
            use_case=node,
        )
        if result.get("explanation"):
            asn.gemini_reasoning = result["explanation"]
    return plan
