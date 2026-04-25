"""
ShipAI — AI Advisor Routes
Endpoints for architecture analysis and pattern recommendations.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.ai_advisor import analyze_use_case, get_all_patterns
from app.services.hardware_checker import check_hardware

router = APIRouter(prefix="/api/advisor", tags=["AI Advisor"])


class AnalyzeRequest(BaseModel):
    description: str
    model: str | None = None


@router.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Analyze a use case and recommend the right AI architecture.
    The AI Advisor thinks like a senior engineer — it knows all RAG types,
    agent patterns, fine-tuning approaches, and when to use each.
    """
    hw = check_hardware()
    result = await analyze_use_case(
        description=req.description,
        hardware_tier=hw.hardware_tier,
        model=req.model,
    )
    return result


@router.get("/patterns")
async def list_patterns():
    """List all AI patterns the advisor knows about."""
    patterns = get_all_patterns()
    total = sum(len(v) for v in patterns.values())
    return {
        "total_patterns": total,
        "categories": patterns,
    }
