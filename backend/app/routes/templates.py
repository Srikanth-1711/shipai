"""
ShipAI — Template Routes
Endpoints for browsing templates and generating projects.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.services.template_engine import list_templates, get_template, generate_project

router = APIRouter(prefix="/api/templates", tags=["Templates"])


class GenerateRequest(BaseModel):
    template_id: str
    config: dict
    output_dir: str | None = None
    tier: str = "free"


@router.get("/")
async def get_templates(tier: str = None):
    """List all available project templates."""
    templates = list_templates(tier=tier)
    return {"templates": templates, "count": len(templates)}


@router.get("/{template_id}")
async def get_template_details(template_id: str):
    """Get full details of a specific template."""
    template = get_template(template_id)
    if not template:
        return {"error": f"Template '{template_id}' not found"}
    return template


@router.post("/generate")
async def generate(req: GenerateRequest):
    """
    Generate a complete project from a template.
    This is the core action — creates a deployable AI project.
    """
    result = await generate_project(
        template_id=req.template_id,
        config=req.config,
        output_dir=req.output_dir,
        tier=req.tier,
    )
    return result
