"""
ShipAI -- License Routes
Endpoints for license key management and tier comparison.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.license_service import (
    generate_license_key,
    validate_license_key,
    activate_license,
    get_current_license,
    check_feature_access,
    get_tier_comparison,
)

router = APIRouter(prefix="/api/license", tags=["License"])


class GenerateKeyRequest(BaseModel):
    tier: str = "free"
    email: str = ""
    duration_days: int = 365


class ActivateRequest(BaseModel):
    key: str


class FeatureCheckRequest(BaseModel):
    feature: str


@router.get("/current")
async def current_license():
    """Get the currently active license."""
    info = get_current_license()
    return {
        "tier": info.tier,
        "email": info.email,
        "is_valid": info.is_valid,
        "is_expired": info.is_expired,
        "days_remaining": info.days_remaining,
        "features": info.features,
    }


@router.get("/tiers")
async def list_tiers():
    """Compare all available pricing tiers."""
    return {"tiers": get_tier_comparison()}


@router.post("/generate")
async def generate_key(req: GenerateKeyRequest):
    """Generate a new license key (admin endpoint)."""
    return generate_license_key(
        tier=req.tier,
        email=req.email,
        duration_days=req.duration_days,
    )


@router.post("/validate")
async def validate_key(req: ActivateRequest):
    """Validate a license key without activating it."""
    info = validate_license_key(req.key)
    return {
        "key": info.key,
        "tier": info.tier,
        "is_valid": info.is_valid,
        "is_expired": info.is_expired,
        "days_remaining": info.days_remaining,
    }


@router.post("/activate")
async def activate(req: ActivateRequest):
    """Activate a license key on this machine."""
    return activate_license(req.key)


@router.post("/check-feature")
async def check_feature(req: FeatureCheckRequest):
    """Check if a specific feature is available with the current license."""
    has_access = check_feature_access(req.feature)
    return {
        "feature": req.feature,
        "has_access": has_access,
        "current_tier": get_current_license().tier,
    }
