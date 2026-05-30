"""
Scale classifier: maps hardware topology to a scale tier + full ScaleProfile.

Pure logic — no subprocess calls. Takes GPUTopology + HardwareProfile
and produces a ScaleProfile with recommended runtime and human reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.install.gpu_topology import GPUTopology
from app.install.types import HardwareProfile
from app.install.vram_calculator import calculate_vram_gb


@dataclass
class ScaleProfile:
    """Complete scale assessment for a machine."""
    tier: str                       # "laptop"|"workstation"|"multi_gpu"|"server"|"hyperscale"|"cpu_cluster"
    gpu_count: int
    total_vram_gb: float            # raw sum
    pooled_vram_gb: float           # after parallelism overhead
    interconnect: str               # "nvlink" | "pcie_x16" | "none"
    recommended_runtime: str        # "ollama" | "vllm" | "ray_vllm" | "llamacpp" | "mlx"
    recommended_tp: int             # tensor parallelism degree
    recommended_pp: int             # pipeline parallelism degree
    max_model_size_b: float         # largest model that fits (B params)
    can_run_7b: bool
    can_run_13b: bool
    can_run_70b: bool
    can_run_405b: bool
    reasoning: str                  # human-readable explanation


# ── VRAM needed for common sizes at Q4_K_M ──────────────────────────────────
# Pre-computed to avoid repeated calls inside tight logic
_VRAM_FOR = {
    "7b":   calculate_vram_gb(7.0, "Q4_K_M")[0],
    "13b":  calculate_vram_gb(13.0, "Q4_K_M")[0],
    "70b":  calculate_vram_gb(70.0, "Q4_K_M")[0],
    "405b": calculate_vram_gb(405.0, "Q4_K_M")[0],
}


def _max_params_for_vram(pooled_vram_gb: float) -> float:
    """Estimate max model params (B) that fit in given pooled VRAM at Q4_K_M."""
    # Binary search approximation: VRAM ≈ params_b × 0.563 + 1.5 overhead
    # Invert: params_b ≈ (vram - 1.5) / 0.563
    return max(0.0, (pooled_vram_gb - 1.5) / 0.563)


def _recommend_runtime(
    tier: str,
    interconnect: str,
    gpu_count: int,
    is_apple: bool = False,
) -> str:
    """Map scale tier to recommended runtime."""
    if is_apple:
        return "ollama"  # MLX via Ollama is the best UX on Apple Silicon
    if tier == "cpu_cluster":
        return "llamacpp"
    if tier == "laptop":
        return "ollama"
    if tier == "workstation":
        return "ollama"   # vllm optional for power users
    if tier in ("multi_gpu", "server"):
        return "vllm"
    if tier == "hyperscale":
        return "ray_vllm"
    return "ollama"


def classify(
    topology: GPUTopology,
    hw: HardwareProfile,
) -> ScaleProfile:
    """
    Produce a complete ScaleProfile from topology + hardware profile.

    Encodes all scale-tier logic in one place.
    """
    tier = topology.scale_tier
    pooled = topology.pooled_vram_gb
    raw_total = topology.total_vram_gb
    gpu_count = len(topology.gpus)
    interconnect = topology.interconnect
    is_apple = hw.is_apple_silicon

    # For Apple Silicon, unified memory is the "VRAM"
    if is_apple and hw.unified_memory_gb:
        pooled = hw.unified_memory_gb * 0.55  # usable share for model
        raw_total = hw.unified_memory_gb

    # CPU-only fallback
    if gpu_count == 0 and not is_apple:
        pooled = hw.effective_ram_gb * 0.4  # ~40% RAM usable for model weights

    tp = topology.recommended_tp_degree
    pp = topology.recommended_pp_degree
    runtime = _recommend_runtime(tier, interconnect, gpu_count, is_apple)
    max_params = _max_params_for_vram(pooled)

    can_7b = pooled >= _VRAM_FOR["7b"]
    can_13b = pooled >= _VRAM_FOR["13b"]
    can_70b = pooled >= _VRAM_FOR["70b"]
    can_405b = pooled >= _VRAM_FOR["405b"]

    # Build human reasoning string
    reasoning_parts: List[str] = []
    if gpu_count == 0 and not is_apple:
        reasoning_parts.append(f"CPU-only: {hw.cpu_cores_logical} threads, {hw.ram_total_gb:.0f}GB RAM")
        reasoning_parts.append(f"Use llama.cpp for inference")
    elif is_apple:
        reasoning_parts.append(f"Apple Silicon: {hw.unified_memory_gb:.0f}GB unified memory")
        reasoning_parts.append(f"Ollama with Metal acceleration recommended")
    elif gpu_count == 1:
        g = topology.gpus[0]
        reasoning_parts.append(f"Single GPU: {g.name} ({g.vram_total_gb:.0f}GB VRAM)")
        if tier == "laptop":
            reasoning_parts.append("Ollama for simplicity — optimal for consumer hardware")
        else:
            reasoning_parts.append("Ollama or vLLM depending on throughput needs")
    else:
        reasoning_parts.append(
            f"{gpu_count}× GPUs via {interconnect} "
            f"({raw_total:.0f}GB raw → {pooled:.0f}GB pooled after {interconnect} overhead)"
        )
        if interconnect == "nvlink":
            reasoning_parts.append(f"vLLM tensor-parallel={tp} for maximum throughput")
        else:
            reasoning_parts.append(
                f"vLLM tensor-parallel={tp} (PCIe: ~60% NVLink efficiency)"
            )

    model_cap = []
    if can_7b:
        model_cap.append("7B")
    if can_13b:
        model_cap.append("13B")
    if can_70b:
        model_cap.append("70B")
    if can_405b:
        model_cap.append("405B")
    if model_cap:
        reasoning_parts.append(f"Can run: {', '.join(model_cap)} at Q4_K_M")

    return ScaleProfile(
        tier=tier,
        gpu_count=gpu_count,
        total_vram_gb=round(raw_total, 2),
        pooled_vram_gb=round(pooled, 2),
        interconnect=interconnect,
        recommended_runtime=runtime,
        recommended_tp=tp,
        recommended_pp=pp,
        max_model_size_b=round(max_params, 1),
        can_run_7b=can_7b,
        can_run_13b=can_13b,
        can_run_70b=can_70b,
        can_run_405b=can_405b,
        reasoning=" | ".join(reasoning_parts),
    )
