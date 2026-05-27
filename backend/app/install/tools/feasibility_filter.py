"""VRAM/RAM/storage feasibility — pure math."""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.install.types import HardwareProfile


@dataclass
class FeasibilityResult:
    feasible: bool
    fits_vram: bool = False
    fits_with_offload: bool = False
    has_storage: bool = True
    estimated_tps: float = 0.0
    total_vram_needed_gb: float = 0.0
    reason: str = "ok"


def parse_params_billions(model_id: str, size_gb: float | None = None) -> float:
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


def is_feasible(
    model_id: str,
    hw: HardwareProfile,
    size_gb: float | None = None,
) -> FeasibilityResult:
    params = parse_params_billions(model_id, size_gb)
    weights_gb = params * 0.55
    kv_cache_gb = params * 0.125
    total_needed = weights_gb + kv_cache_gb

    eff_vram = getattr(hw, "effective_vram_gb", 0) or 0
    eff_ram = getattr(hw, "effective_ram_gb", hw.ram_available_gb)

    fits_vram = total_needed <= eff_vram if eff_vram > 0 else total_needed <= eff_ram * 0.4
    fits_offload = False
    if not fits_vram and eff_ram > 8:
        offload = total_needed - max(eff_vram, 0)
        fits_offload = offload * 1.5 <= eff_ram

    has_storage = (size_gb or weights_gb) * 1.1 <= hw.disk_free_gb

    if fits_vram:
        tps = min(60.0, max(5.0, (eff_vram / max(total_needed, 0.1)) * 15))
    elif fits_offload:
        tps = 3.0 + hw.cpu_cores_physical * 0.5
    else:
        tps = 0.0

    feasible = (fits_vram or fits_offload) and has_storage
    return FeasibilityResult(
        feasible=feasible,
        fits_vram=fits_vram,
        fits_with_offload=fits_offload,
        has_storage=has_storage,
        estimated_tps=round(tps, 1),
        total_vram_needed_gb=round(total_needed, 2),
        reason="ok" if feasible else "insufficient_vram_or_storage",
    )
