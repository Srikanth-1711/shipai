"""VRAM/RAM/storage feasibility — uses calculated VRAM + speed estimation."""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.install.speed_estimator import estimate_tokens_per_second, infer_backend
from app.install.types import HardwareProfile
from app.install.vram_calculator import calculate_vram_gb, extract_active_params


@dataclass
class FeasibilityResult:
    feasible: bool
    fits_vram: bool = False
    fits_with_offload: bool = False
    has_storage: bool = True
    estimated_tps: float = 0.0
    total_vram_needed_gb: float = 0.0
    vram_confidence: str = "medium"
    tps_low: float = 0.0
    tps_high: float = 0.0
    tps_confidence: str = "?"
    reason: str = "ok"


def parse_params_billions(model_id: str, size_gb: float | None = None) -> float:
    """Extract parameter count in billions from model name or size."""
    low = model_id.lower()
    for pat in (
        r"(\d+(?:\.\d+)?)\s*b",
        r":(\d+(?:\.\d+)?)b",
        r"-(\d+(?:\.\d+)?)b",
    ):
        m = re.search(pat, low)
        if m:
            return float(m.group(1))
    if size_gb:
        return max(size_gb / 0.55, 0.5)
    return 3.0


def _guess_quant_from_name(model_id: str) -> str:
    """Try to extract quantization from model name, default Q4_K_M."""
    upper = model_id.upper()
    # Check for explicit quant in name
    for q in ("Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0",
              "Q3_K_M", "Q3_K_L", "Q3_K_S", "Q2_K", "IQ2_XXS", "F16", "BF16"):
        if q in upper or q.replace("_", "-") in upper:
            return q
    return "Q4_K_M"  # safe default


def is_feasible(
    model_id: str,
    hw: HardwareProfile,
    size_gb: float | None = None,
) -> FeasibilityResult:
    """
    Determine if a model can run on this hardware.

    Uses calculated VRAM (from model params + quant) instead of static JSON,
    and estimated tok/s from bandwidth + backend efficiency.
    """
    params = parse_params_billions(model_id, size_gb)
    quant = _guess_quant_from_name(model_id)

    # Extract MoE active params
    active_params, is_moe = extract_active_params(model_id, params)

    # Calculate VRAM needed
    total_needed, vram_confidence = calculate_vram_gb(
        params_b=params,
        quant=quant,
        context_length=4096,
        is_moe=is_moe,
        active_params_b=active_params if is_moe else None,
    )

    eff_vram = getattr(hw, "effective_vram_gb", 0) or 0
    eff_ram = getattr(hw, "effective_ram_gb", hw.ram_available_gb)

    # Use pooled VRAM across all GPUs if multi-GPU machine (after parallelism overhead)
    gpu_count = getattr(hw, "gpu_count", 1) or 1
    total_vram = getattr(hw, "total_vram_gb", 0) or 0
    budget_vram = total_vram if (gpu_count > 1 and total_vram > eff_vram) else eff_vram

    fits_vram = total_needed <= budget_vram if budget_vram > 0 else total_needed <= eff_ram * 0.4
    fits_offload = False
    if not fits_vram and eff_ram > 8:
        offload = total_needed - max(eff_vram, 0)
        fits_offload = offload * 1.5 <= eff_ram

    has_storage = (size_gb or total_needed) * 1.1 <= hw.disk_free_gb

    # Speed estimation using bandwidth-based model
    bandwidth = getattr(hw, "gpu_bandwidth_gb_s", 100.0) or 100.0
    gpu_vendor = getattr(hw, "gpu_vendor", "none") or "none"
    cpu_features = getattr(hw, "cpu_features", []) or []

    if fits_vram and eff_vram > 0:
        backend = infer_backend(gpu_vendor, cpu_features)
    elif fits_offload:
        # Partial offload — use CPU backend but with some GPU assist
        backend = infer_backend("none", cpu_features)
        bandwidth = 50.0  # rough estimate for partial offload
    else:
        backend = infer_backend("none", cpu_features)
        bandwidth = 30.0  # CPU-only memory bandwidth

    median_tps, low_tps, high_tps, tps_conf = estimate_tokens_per_second(
        params_b=params,
        quant=quant,
        bandwidth_gb_s=bandwidth,
        backend=backend,
        is_moe=is_moe,
        active_params_b=active_params if is_moe else None,
    )

    feasible = (fits_vram or fits_offload) and has_storage
    return FeasibilityResult(
        feasible=feasible,
        fits_vram=fits_vram,
        fits_with_offload=fits_offload,
        has_storage=has_storage,
        estimated_tps=round(median_tps, 1),
        total_vram_needed_gb=round(total_needed, 2),
        vram_confidence=vram_confidence,
        tps_low=round(low_tps, 1),
        tps_high=round(high_tps, 1),
        tps_confidence=tps_conf,
        reason="ok" if feasible else "insufficient_vram_or_storage",
    )
