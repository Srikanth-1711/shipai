"""
ShipAI — System & Health Routes
Endpoints for hardware detection, health checks, and system status.
"""
from fastapi import APIRouter
from app.services.hardware_checker import check_hardware
from app.services.ollama_service import ollama_service
from app.services.infra_generator import get_infra_summary
from app.config import settings
from app.runtime.capabilities import summarize_capabilities

router = APIRouter(prefix="/api/system", tags=["System"])


@router.get("/health")
async def health_check():
    """Overall system health check."""
    ollama_status = await ollama_service.health_check()
    hw = check_hardware()
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "ollama": ollama_status,
        "hardware_tier": hw.hardware_tier,
        "gpu_detected": hw.has_gpu,
    }


@router.get("/hardware")
async def get_hardware_info():
    """Full hardware detection with model recommendations."""
    hw = check_hardware()
    result = {
        "system": {
            "os": f"{hw.os_name} {hw.os_version}",
            "architecture": hw.architecture,
        },
        "cpu": {
            "name": hw.cpu_name,
            "cores_physical": hw.cpu_cores_physical,
            "cores_logical": hw.cpu_cores_logical,
            "frequency_mhz": hw.cpu_freq_mhz,
        },
        "memory": {
            "total_gb": hw.ram_total_gb,
            "available_gb": hw.ram_available_gb,
            "used_percent": hw.ram_used_percent,
        },
        "gpu": None,
        "disk": {
            "total_gb": hw.disk_total_gb,
            "free_gb": hw.disk_free_gb,
        },
        "recommendation": {
            "tier": hw.hardware_tier,
            "tier_label": hw.tier_label,
            "recommended_models": hw.recommended_models,
            "available_features": hw.available_features,
        },
    }
    if hw.gpu:
        result["gpu"] = {
            "name": hw.gpu.name,
            "vram_total_mb": hw.gpu.vram_total_mb,
            "vram_free_mb": hw.gpu.vram_free_mb,
            "driver_version": hw.gpu.driver_version,
            "cuda_version": hw.gpu.cuda_version,
        }
    return result


@router.get("/infra-patterns")
async def get_infrastructure_patterns(tier: str = "pro"):
    """Show which production infrastructure patterns are included per tier."""
    return {
        "tier": tier,
        "patterns": get_infra_summary(tier),
    }


@router.get("/runtime-capabilities")
async def get_runtime_capability_report():
    """Report optional/required dependency capabilities for graceful degradation."""
    return summarize_capabilities()
