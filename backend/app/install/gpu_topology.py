"""
Multi-GPU topology detection: all GPUs, interconnect, VRAM pooling.

Detects ALL physical GPUs (not just the first one), determines whether
they are connected via NVLink or PCIe, and calculates the effective
pooled VRAM after parallelism communication overhead.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

from app.install.hardware_profile import GPU_BANDWIDTH_GB_S, get_bandwidth
from app.install.types import GPUInfo

logger = logging.getLogger("shipai.topology")


# ── Interconnect bandwidth constants ────────────────────────────────────────
NVLINK_BANDWIDTH = {
    "NVLink 4.0": 900.0,   # H100
    "NVLink 3.0": 600.0,   # A100, RTX 3090
    "NVLink 2.0": 300.0,   # V100
    "NVLink":     300.0,   # generic fallback
}
PCIE_BANDWIDTH = {
    "pcie_x16": 32.0,
    "pcie_x8":  16.0,
    "pcie_x4":   8.0,
}

# VRAM pooling overhead per interconnect type
POOLING_EFFICIENCY = {
    "nvlink":   0.92,   # 8% overhead for tensor parallel comms
    "pcie_x16": 0.78,   # 22% overhead — conservative for PCIe
    "pcie_x8":  0.70,
    "none":     1.00,   # single GPU
}


@dataclass
class GPUTopology:
    """Full multi-GPU topology snapshot."""
    gpus: List[GPUInfo] = field(default_factory=list)
    interconnect: str = "none"          # "nvlink" | "pcie_x16" | "pcie_x8" | "none"
    interconnect_bandwidth_gb_s: float = 0.0
    total_vram_gb: float = 0.0          # raw sum across all GPUs
    pooled_vram_gb: float = 0.0         # after parallelism overhead
    recommended_tp_degree: int = 1      # tensor parallelism
    recommended_pp_degree: int = 1      # pipeline parallelism (NVLink only)
    scale_tier: str = "laptop"
    topology_notes: str = ""


# ── NVIDIA multi-GPU detection ───────────────────────────────────────────────

def detect_all_nvidia_gpus() -> List[GPUInfo]:
    """
    Detect ALL NVIDIA GPUs via nvidia-smi.
    Previous code only parsed the first line — this parses all.
    """
    gpus: List[GPUInfo] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return gpus
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                idx = int(parts[0])
                name = parts[1]
                total_gb = float(parts[2]) / 1024
                free_gb = float(parts[3]) / 1024
                uuid = parts[4]
                gpus.append(GPUInfo(
                    index=idx,
                    name=name,
                    vram_total_gb=round(total_gb, 2),
                    vram_free_gb=round(free_gb, 2),
                    vendor="nvidia",
                    bandwidth_gb_s=get_bandwidth(name),
                    has_cuda=True,
                    uuid=uuid,
                ))
            except (ValueError, IndexError):
                continue
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return gpus


def detect_nvlink_topology(gpu_count: int) -> tuple[str, float]:
    """
    Parse nvidia-smi topo -m to determine NVLink vs PCIe interconnect.
    Returns (interconnect_type, bandwidth_gb_s).
    """
    if gpu_count < 2:
        return "none", 0.0
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return "pcie_x16", PCIE_BANDWIDTH["pcie_x16"]

        output = result.stdout
        # NVLink markers: NV1, NV2, NV3, NV4 in topology matrix
        has_nvlink = any(f"NV{n}" in output for n in ("1", "2", "3", "4", "5", "6"))
        if has_nvlink:
            # Detect NVLink generation from GPU name (heuristic)
            if "H100" in output or "H200" in output:
                bw = NVLINK_BANDWIDTH["NVLink 4.0"]
            elif "A100" in output or "RTX 30" in output or "RTX 40" in output:
                bw = NVLINK_BANDWIDTH["NVLink 3.0"]
            else:
                bw = NVLINK_BANDWIDTH["NVLink"]
            return "nvlink", bw

        # Check for PIX (PCIe x16) vs PXB (PCIe switch)
        if "PIX" in output or "PXB" in output:
            return "pcie_x16", PCIE_BANDWIDTH["pcie_x16"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "pcie_x16", PCIE_BANDWIDTH["pcie_x16"]  # conservative default


# ── AMD multi-GPU detection ──────────────────────────────────────────────────

def detect_all_amd_gpus() -> List[GPUInfo]:
    """Detect ALL AMD GPUs via rocm-smi --showallgpus."""
    gpus: List[GPUInfo] = []
    try:
        result = subprocess.run(
            ["rocm-smi", "--showallgpus", "--showmeminfo", "vram",
             "--showproductname", "--json"],
            capture_output=True, text=True, timeout=8,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return gpus
        data = json.loads(result.stdout)
        if not isinstance(data, dict):
            return gpus
        idx = 0
        for card_id, info in data.items():
            if not isinstance(info, dict):
                continue
            if "card" not in str(card_id).lower():
                continue
            try:
                vram_bytes = int(info.get("VRAM Total Memory (B)", 0))
                name = str(info.get("Card series", f"AMD GPU {idx}"))
                if vram_bytes > 0:
                    total_gb = round(vram_bytes / 1e9, 2)
                    gpus.append(GPUInfo(
                        index=idx,
                        name=name,
                        vram_total_gb=total_gb,
                        vram_free_gb=round(total_gb * 0.9, 2),
                        vendor="amd",
                        bandwidth_gb_s=get_bandwidth(name),
                        has_rocm=True,
                    ))
                    idx += 1
            except (ValueError, TypeError):
                continue
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return gpus


# ── VRAM pooling math ────────────────────────────────────────────────────────

def calculate_pooled_vram(
    gpus: List[GPUInfo],
    interconnect: str,
) -> float:
    """
    Calculate effective pooled VRAM across all GPUs after parallelism overhead.

    NVLink: 92% efficiency (8% comms overhead)
    PCIe x16: 78% efficiency (22% overhead)
    Single GPU: 100% (no overhead)
    """
    if not gpus:
        return 0.0
    raw_total = sum(g.vram_total_gb for g in gpus)
    if len(gpus) == 1:
        return round(raw_total, 2)
    efficiency = POOLING_EFFICIENCY.get(interconnect, 0.78)
    return round(raw_total * efficiency, 2)


# ── Scale tier classification ────────────────────────────────────────────────

def classify_scale_tier(
    gpus: List[GPUInfo],
    interconnect: str,
    cpu_cores: int = 1,
    ram_gb: float = 0.0,
) -> str:
    """
    Map GPU count + interconnect + VRAM to a scale tier.

    laptop      → 0-1 GPU, ≤8GB VRAM or CPU-only
    workstation → 1 GPU, >8GB VRAM
    multi_gpu   → 2-3 GPUs
    server      → 4-7 GPUs or 1-3 GPUs with NVLink
    hyperscale  → 8+ GPUs
    cpu_cluster → 0 GPUs, many cores / large RAM
    """
    if not gpus:
        if cpu_cores >= 32 or ram_gb >= 256:
            return "cpu_cluster"
        return "laptop"

    count = len(gpus)
    total_vram = sum(g.vram_total_gb for g in gpus)

    if count >= 8:
        return "hyperscale"
    if count >= 4:
        return "server"
    if count >= 2:
        if interconnect == "nvlink":
            return "server"
        return "multi_gpu"
    # Single GPU
    if total_vram <= 8:
        return "laptop"
    return "workstation"


# ── Parallelism degree recommendation ───────────────────────────────────────

def recommend_parallelism(
    gpus: List[GPUInfo],
    interconnect: str,
    model_vram_needed_gb: float = 0.0,
) -> tuple[int, int]:
    """
    Returns (tensor_parallel_degree, pipeline_parallel_degree).

    PCIe never uses pipeline parallel for inference — PCIe bandwidth is
    too low for the constant cross-stage communication PP requires.
    PCIe prefers single GPU when model fits; tensor parallel when it doesn't.
    NVLink can use TP + PP combinations.
    """
    count = len(gpus)
    if count <= 1:
        return 1, 1

    single_gpu_vram = gpus[0].vram_total_gb if gpus else 0.0

    if interconnect == "nvlink":
        # NVLink: full tensor parallel, pipeline for very large models
        if count <= 4:
            return count, 1
        else:
            # 8+ GPUs: split TP=4, PP=2 (common vLLM config)
            tp = min(count, 4)
            pp = count // tp
            return tp, pp
    else:
        # PCIe: prefer single GPU if model fits (avoids slow cross-GPU transfer)
        if model_vram_needed_gb > 0 and single_gpu_vram >= model_vram_needed_gb:
            return 1, 1  # single GPU faster than PCIe multi-GPU
        # Must span: tensor parallel only (no pipeline parallel on PCIe)
        return count, 1


# ── Main topology detector ───────────────────────────────────────────────────

def detect_gpu_topology(
    cpu_cores: int = 1,
    ram_gb: float = 0.0,
) -> GPUTopology:
    """
    Full GPU topology detection: all vendors, all GPUs, interconnect, pooling.
    """
    # Try NVIDIA first (most common in servers)
    gpus = detect_all_nvidia_gpus()
    if not gpus:
        gpus = detect_all_amd_gpus()

    if not gpus:
        return GPUTopology(
            gpus=[],
            interconnect="none",
            total_vram_gb=0.0,
            pooled_vram_gb=0.0,
            scale_tier=classify_scale_tier([], "none", cpu_cores, ram_gb),
            topology_notes="No GPU detected — CPU-only inference",
        )

    # Detect interconnect
    interconnect, ic_bw = detect_nvlink_topology(len(gpus))
    if ic_bw == 0.0 and len(gpus) > 1:
        interconnect = "pcie_x16"
        ic_bw = PCIE_BANDWIDTH["pcie_x16"]

    raw_total = sum(g.vram_total_gb for g in gpus)
    pooled = calculate_pooled_vram(gpus, interconnect)
    scale = classify_scale_tier(gpus, interconnect, cpu_cores, ram_gb)
    tp, pp = recommend_parallelism(gpus, interconnect)

    notes = f"{len(gpus)} GPU(s), {interconnect}, {raw_total:.1f}GB raw → {pooled:.1f}GB pooled"

    return GPUTopology(
        gpus=gpus,
        interconnect=interconnect,
        interconnect_bandwidth_gb_s=ic_bw,
        total_vram_gb=round(raw_total, 2),
        pooled_vram_gb=pooled,
        recommended_tp_degree=tp,
        recommended_pp_degree=pp,
        scale_tier=scale,
        topology_notes=notes,
    )
