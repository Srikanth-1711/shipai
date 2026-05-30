"""Fusion scorer — CEO + matrix + web + hardware → top 10 → top 3 roles."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from app.install.ceo_policy import ceo_boost_for_model, is_banned, load_ceo_policy
from app.install.capability_fetcher import get_model_entry
from app.install.intent_profile import IntentProfile
from app.install.model_negotiator import CHAT_NODES
from app.install.model_plan import ModelPlan, NodeAssignment
from app.install.speed_estimator import format_speed_display
from app.install.tools.feasibility_filter import FeasibilityResult, is_feasible
from app.install.tools.web_intelligence import RichModelInfo
from app.install.types import DiscoveredModel, EnvironmentReport, HardwareProfile

WEIGHTS = {
    "performance": 0.30,
    "efficiency": 0.25,
    "fit": 0.20,
    "popularity": 0.15,
    "speed": 0.10,
}


@dataclass
class ModelScore:
    model: RichModelInfo
    final_score: float
    dimensions: Dict[str, float] = field(default_factory=dict)
    feasibility: Optional[FeasibilityResult] = None
    recommended_role: str = "medium"
    installed: bool = False


@dataclass
class RankResult:
    top_10: List[ModelScore]
    top_3: List[ModelScore]
    final_three: Dict[str, str]
    node_assignments: Dict[str, NodeAssignment]
    embedding_model: Optional[str] = None
    models_to_download: List[str] = field(default_factory=list)


def _matrix_fit_score(model_id: str, matrix: dict[str, Any]) -> float:
    entry = get_model_entry(matrix, model_id)
    if not entry:
        return 0.4
    caps = entry.get("capabilities", {})
    if caps.get("embeddings"):
        return 0.0
    return min(
        1.0,
        (
            float(caps.get("json_reliable", 0.5))
            + float(caps.get("reasoning_score", 0.5))
            + float(caps.get("code_generation", 0.4))
        )
        / 2.5,
    )


def _installed_ids(report: EnvironmentReport) -> Set[str]:
    return {m.name.lower() for m in report.inventory if m.is_served}


def score_model(
    model: RichModelInfo,
    feasibility: FeasibilityResult,
    hw: HardwareProfile,
    matrix: dict[str, Any],
    policy: dict[str, Any],
    max_pulls: int,
) -> ModelScore:
    bench = min((model.benchmark_score or 0) / 100.0, 1.0)
    perf = max(bench, _matrix_fit_score(model.ollama_id, matrix))

    eff = 0.0
    if feasibility.total_vram_needed_gb > 0:
        eff = min(perf / feasibility.total_vram_needed_gb, 1.0) / 0.15
        eff = min(eff, 1.0)

    speed = min(feasibility.estimated_tps / 60.0, 1.0)
    pop = 0.0
    if max_pulls > 0 and model.pull_count > 0:
        pop = min(math.log(model.pull_count + 1) / math.log(max_pulls + 1), 1.0)

    fit = _matrix_fit_score(model.ollama_id, matrix)
    ceo = max(ceo_boost_for_model(model.ollama_id, policy), 0.0)

    final = (
        perf * WEIGHTS["performance"]
        + eff * WEIGHTS["efficiency"]
        + fit * WEIGHTS["fit"]
        + pop * WEIGHTS["popularity"]
        + speed * WEIGHTS["speed"]
        + ceo
    )

    role = "medium"
    if speed >= 0.5 and perf < 0.75:
        role = "fast"
    elif perf >= 0.8:
        role = "heavy"

    return ModelScore(
        model=model,
        final_score=round(final, 4),
        dimensions={
            "performance": round(perf, 3),
            "efficiency": round(eff, 3),
            "speed": round(speed, 3),
            "popularity": round(pop, 3),
            "fit": round(fit, 3),
            "ceo_boost": round(ceo, 3),
        },
        feasibility=feasibility,
        recommended_role=role,
    )


def _pick_embed_model(
    ranked: List[ModelScore],
    installed: Set[str],
    matrix: dict[str, Any],
    policy: dict[str, Any],
) -> Optional[str]:
    prefs = policy.get("role_preferences", {}).get("embed", {}).get("prefer", [])
    for name in installed:
        if "embed" in name.lower():
            return name
    for pref in prefs:
        for s in ranked:
            if pref.lower() in s.model.ollama_id.lower():
                return s.model.ollama_id
    for key, entry in matrix.get("models", {}).items():
        if entry.get("capabilities", {}).get("embeddings"):
            return entry.get("ollama_id", key)
    return "nomic-embed-text"


def assign_roles_from_top3(
    top3: List[ModelScore],
    intent: IntentProfile,
) -> Tuple[Dict[str, str], Dict[str, NodeAssignment]]:
    if not top3:
        return {}, {}

    if len(top3) == 1:
        by_role = {"fast": top3[0], "medium": top3[0], "heavy": top3[0]}
    elif len(top3) == 2:
        by_role = {"heavy": top3[0], "medium": top3[1], "fast": top3[1]}
    else:
        by_role = {"heavy": top3[0], "medium": top3[1], "fast": top3[2]}

    for s in top3:
        if s.recommended_role == "fast":
            by_role["fast"] = s
        elif s.recommended_role == "heavy":
            by_role["heavy"] = s
        else:
            by_role["medium"] = s

    final_three = {
        "fast": by_role["fast"].model.ollama_id,
        "medium": by_role["medium"].model.ollama_id,
        "heavy": by_role["heavy"].model.ollama_id,
    }

    node_map = {
        "interview_node": ("fast", by_role["fast"]),
        "explain_node": ("fast", by_role["fast"]),
        "research_node": ("heavy", by_role["heavy"]),
        "plan_node": ("heavy", by_role["heavy"]) if intent.project_type != "code" else ("medium", by_role["medium"]),
    }

    assignments: Dict[str, NodeAssignment] = {}
    for node, (role, sc) in node_map.items():
        status = "installed" if sc.installed else "needs_download"
        conf = "installed_verified" if sc.installed else "fusion_scored"
        # Build display output with speed/VRAM info
        tps = sc.feasibility.estimated_tps if sc.feasibility else 0.0
        tps_low = getattr(sc.feasibility, "tps_low", tps * 0.8) if sc.feasibility else 0.0
        tps_high = getattr(sc.feasibility, "tps_high", tps * 1.25) if sc.feasibility else 0.0
        vram_needed = sc.feasibility.total_vram_needed_gb if sc.feasibility else 0.0
        vram_conf = getattr(sc.feasibility, "vram_confidence", "medium") if sc.feasibility else ""
        tps_conf = getattr(sc.feasibility, "tps_confidence", "?") if sc.feasibility else "?"
        display = format_speed_display(
            model_id=sc.model.ollama_id,
            median_tps=tps,
            low_tps=tps_low,
            high_tps=tps_high,
            vram_needed_gb=vram_needed,
            vram_available_gb=0.0,  # filled by caller if known
            confidence=tps_conf,
        )
        assignments[node] = NodeAssignment(
            model=sc.model.ollama_id,
            runtime="ollama",
            status=status,
            confidence=conf,
            quality_score=sc.final_score,
            reason=f"Fusion rank - {role} slot for {node}",
            role=role,
            estimated_tps=round(tps, 1),
            vram_needed_gb=round(vram_needed, 2),
            vram_confidence=vram_conf,
            display_output=display,
        )
    return final_three, assignments


def rank_models(
    candidates: List[RichModelInfo],
    report: EnvironmentReport,
    matrix: dict[str, Any],
    intent: IntentProfile,
    policy: dict[str, Any] | None = None,
) -> RankResult:
    policy = policy or load_ceo_policy()
    hw = report.hardware
    installed = _installed_ids(report)
    policy_flagship = policy.get("flagship_multi_agent", "")

    scored: List[ModelScore] = []
    max_pulls = max((m.pull_count for m in candidates), default=1)

    for model in candidates:
        if is_banned(model.ollama_id, policy):
            continue
        entry = get_model_entry(matrix, model.ollama_id)
        if entry and entry.get("capabilities", {}).get("embeddings"):
            continue
        feas = is_feasible(model.ollama_id, hw, model.size_gb)
        if not feas.feasible:
            continue
        sc = score_model(model, feas, hw, matrix, policy, max_pulls)
        base = model.ollama_id.lower().split(":")[0]
        sc.installed = any(
            model.ollama_id.lower() in i or i.startswith(base) or base in i.split(":")[0]
            for i in installed
        )
        if policy_flagship and policy_flagship.lower() in model.ollama_id.lower():
            sc.final_score += float(policy.get("flagship_boost", 0.15))
        if sc.installed:
            sc.final_score += 0.05
        scored.append(sc)

    scored.sort(key=lambda s: -s.final_score)
    top_10 = scored[:10]
    top_3 = scored[:3]

    final_three, node_assignments = assign_roles_from_top3(top_3, intent)
    embed = _pick_embed_model(scored, installed, matrix, policy) if intent.require_embed else None

    to_dl: List[str] = []
    seen: Set[str] = set()
    for role, mid in final_three.items():
        if mid.lower() not in installed and mid not in seen:
            to_dl.append(mid)
            seen.add(mid)
    if embed and embed.lower() not in installed and embed not in seen:
        to_dl.append(embed)
        seen.add(embed)

    return RankResult(
        top_10=top_10,
        top_3=top_3,
        final_three=final_three,
        node_assignments=node_assignments,
        embedding_model=embed,
        models_to_download=to_dl,
    )


def build_plan_from_rank(
    rank: RankResult,
    report: EnvironmentReport,
    matrix: dict[str, Any],
    intent: IntentProfile,
    *,
    gemini_enabled: bool = False,
) -> ModelPlan:
    rt = next((r for r in report.runtimes if r.is_running), None)
    primary = "ollama" if rt and rt.name in ("ollama", "ollama_custom") else (rt.name if rt else "none")
    base_url = rt.base_url if rt else None

    embedding = None
    if rank.embedding_model:
        embedding = NodeAssignment(
            model=rank.embedding_model,
            runtime=primary,
            base_url=base_url,
            status="installed"
            if rank.embedding_model.lower() in _installed_ids(report)
            else "needs_download",
            confidence="fusion_scored",
            reason="Embedding model for RAG / document use-case",
            role="embed",
        )

    return ModelPlan(
        primary_runtime=primary,
        runtime_base_url=base_url,
        embedding=embedding,
        node_assignments=rank.node_assignments,
        models_to_download=rank.models_to_download,
        final_three=rank.final_three,
        top10_candidates=[
            {
                "rank": i + 1,
                "model": s.model.ollama_id,
                "score": s.final_score,
                "role": s.recommended_role,
                "installed": s.installed,
            }
            for i, s in enumerate(rank.top_10)
        ],
        use_case_profile={
            "raw": intent.raw,
            "project_type": intent.project_type,
            "require_embed": intent.require_embed,
        },
        gemini_enabled=gemini_enabled,
        metadata={
            "hardware": {
                "gpu_name": report.hardware.gpu_name,
                "gpu_vram_gb": report.hardware.gpu_vram_total_gb,
                "ram_total_gb": report.hardware.ram_total_gb,
                "ram_available_gb": report.hardware.ram_available_gb,
                "effective_vram_gb": report.hardware.effective_vram_gb,
                "max_model_params_b": report.hardware.max_model_params_b,
            },
            "matrix_version": matrix.get("schema_version"),
            "fusion": True,
        },
    )
