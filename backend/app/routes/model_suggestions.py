"""
ShipAI — Model Suggestion Routes
Endpoints for user model suggestions and Gemini reasoning.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from app.services.gemini_service import gemini_service
from app.install.model_plan import load_model_plan

router = APIRouter(prefix="/api/models", tags=["Model Selection"])


class ModelSuggestion(BaseModel):
    """User suggests a model for a specific node."""
    node: str  # interview_node, research_node, etc.
    model: str  # model name
    reason: Optional[str] = None  # Why they suggest it


class SuggestionsRequest(BaseModel):
    """Batch model suggestions."""
    suggestions: Dict[str, str]  # {node: model_name, ...}
    validate_with_gemini: bool = False


class ExplainChoiceRequest(BaseModel):
    """Request Gemini explanation for a model choice."""
    model: str
    node: str
    hardware: Optional[dict] = None  # If not provided, use current plan


@router.post("/suggest")
async def suggest_model(req: ModelSuggestion):
    """
    User suggests a model for a node.
    Returns validation result if Gemini is enabled.
    """
    plan = load_model_plan()
    if not plan:
        raise HTTPException(status_code=400, detail="No model plan found. Run 'shipai install' first.")

    # Get current hardware context from plan metadata
    hardware = plan.metadata.get("hardware", {})

    validation = await gemini_service.validate_user_model_suggestion(
        suggested_model=req.model,
        hardware_specs=hardware,
        available_models=list(plan.metadata.get("available_models", [])),
    ) if gemini_service.enabled else {"valid": True, "reasoning": "Validation skipped (Gemini not configured)"}

    return {
        "suggestion": {
            "node": req.node,
            "model": req.model,
            "reason": req.reason,
        },
        "validation": validation,
        "status": "valid" if validation.get("valid") else "invalid",
    }


@router.post("/explain")
async def explain_model_choice(req: ExplainChoiceRequest):
    """
    Get Gemini explanation for why a model is good for a node.
    """
    if not gemini_service.enabled:
        return {
            "error": "Gemini not configured",
            "explanation": None,
        }

    plan = load_model_plan()
    if not plan and not req.hardware:
        raise HTTPException(status_code=400, detail="No hardware specs provided and no model plan found")

    # Use provided hardware or get from plan
    hardware = req.hardware or plan.metadata.get("hardware", {})

    # Get alternatives
    alternatives = list(plan.metadata.get("available_models", []))[:3]

    result = await gemini_service.explain_model_choice(
        hardware_specs=hardware,
        selected_model=req.model,
        alternatives=alternatives,
        use_case=req.node,
    )

    return {
        "model": req.model,
        "node": req.node,
        "explanation": result.get("explanation"),
        "confidence": result.get("confidence"),
        "concerns": result.get("concerns", []),
    }


@router.get("/current-plan")
async def get_current_plan():
    """
    Retrieve the current model plan with all assignments and reasoning.
    """
    plan = load_model_plan()
    if not plan:
        raise HTTPException(status_code=404, detail="No model plan found. Run 'shipai install' first.")

    return {
        "schema_version": plan.schema_version,
        "primary_runtime": plan.primary_runtime,
        "final_three": plan.final_three,
        "top10_candidates": plan.top10_candidates[:10],
        "use_case_profile": plan.use_case_profile,
        "embedding": plan.embedding.model if plan.embedding else None,
        "node_assignments": {
            node: {
                "model": asn.model,
                "status": asn.status,
                "confidence": asn.confidence,
                "role": asn.role,
                "reason": asn.reason,
                "gemini_reasoning": asn.gemini_reasoning,
                "user_suggested": asn.user_suggested,
            }
            for node, asn in plan.node_assignments.items()
        },
        "user_suggestions": plan.user_suggestions,
        "warnings": plan.warnings,
        "gemini_enabled": plan.gemini_enabled,
        "generated_at": plan.generated_at,
    }


@router.post("/replan-with-suggestions")
async def replan_with_suggestions(req: SuggestionsRequest):
    """
    Re-run negotiation with user model suggestions.
    Can be integrated into setup flow or run standalone.
    """
    from app.install.environment import build_environment_report
    from app.install.capability_fetcher import get_capability_matrix, fetch_ollama_catalog
    from app.install.enhanced_negotiator import negotiate_enhanced

    # Re-build current state
    report = await build_environment_report()
    matrix = await get_capability_matrix()
    catalog = await fetch_ollama_catalog(matrix_fallback=matrix)

    # Negotiate with suggestions
    plan = await negotiate_enhanced(
        report=report,
        matrix=matrix,
        catalog=catalog,
        user_suggestions=req.suggestions,
        use_gemini=req.validate_with_gemini,
    )

    from app.install.model_plan import save_model_plan
    save_model_plan(plan)

    return {
        "status": "success",
        "suggestions_applied": len(req.suggestions),
        "node_assignments": {
            node: {
                "model": asn.model,
                "user_suggested": asn.user_suggested,
                "reason": asn.reason,
            }
            for node, asn in plan.node_assignments.items()
        },
        "warnings": plan.warnings,
    }
