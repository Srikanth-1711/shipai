"""
Runtime recommendation: maps ScaleProfile → exact runtime + launch args.

Produces concrete CLI commands, env vars, and explanations.
All three fixes from plan review are applied here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.install.scale_classifier import ScaleProfile
from app.install.types import HardwareProfile
from app.install.vram_calculator import calculate_vram_gb


@dataclass
class RuntimeRecommendation:
    """Concrete runtime recommendation with launch instructions."""
    runtime: str                    # "ollama" | "vllm" | "llamacpp" | "ray_vllm" | "mlx"
    launch_args: List[str] = field(default_factory=list)   # CLI args
    env_vars: Dict[str, str] = field(default_factory=dict)
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    explanation: str = ""
    install_hint: str = ""          # what to install if runtime is missing
    gpu_device_ids: List[int] = field(default_factory=list)


# ── HF → Ollama name mapping ────────────────────────────────────────────────
# Maps common HuggingFace repo patterns to Ollama model names
_HF_TO_OLLAMA: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct":    "llama3.1:8b",
    "meta-llama/llama-3.1-70b-instruct":   "llama3.1:70b",
    "meta-llama/llama-3.2-3b-instruct":    "llama3.2:3b",
    "meta-llama/llama-3.2-1b-instruct":    "llama3.2:1b",
    "meta-llama/llama-4-scout-17b-16e-instruct": "llama4:scout",
    "google/gemma-2-9b-it":                "gemma2:9b",
    "google/gemma-2-27b-it":               "gemma2:27b",
    "mistralai/mistral-7b-instruct-v0.3":  "mistral:7b",
    "mistralai/mixtral-8x7b-instruct-v0.1": "mixtral:8x7b",
    "qwen/qwen2.5-7b-instruct":            "qwen2.5:7b",
    "qwen/qwen2.5-14b-instruct":           "qwen2.5:14b",
    "qwen/qwen2.5-72b-instruct":           "qwen2.5:72b",
    "microsoft/phi-4":                     "phi4:latest",
    "microsoft/phi-3.5-mini-instruct":     "phi3.5:mini",
    "deepseek-ai/deepseek-r1":             "deepseek-r1:7b",
    "deepseek-ai/deepseek-v3":             "deepseek-v3:latest",
}

# Known GGUF community repos (TheBloke pattern + newer bartowski/lmstudio-community)
_GGUF_REPO_PATTERNS = [
    "{org}/{name}-GGUF",          # TheBloke style
    "bartowski/{name}-GGUF",      # bartowski style
    "lmstudio-community/{name}-GGUF",
]


def map_hf_to_ollama_name(hf_repo_id: str) -> Optional[str]:
    """Map a HuggingFace repo ID to an Ollama model name."""
    key = hf_repo_id.lower().strip()
    return _HF_TO_OLLAMA.get(key)


def find_gguf_repo(model_name: str) -> Optional[str]:
    """
    Heuristic: find a likely GGUF community repo for a given model name.
    Returns a HuggingFace repo ID to try downloading from.
    """
    clean = model_name.lower()
    # Strip org prefix
    if "/" in clean:
        parts = clean.split("/")
        org, name = parts[0], parts[1]
    else:
        org, name = "", clean

    # Try bartowski first (most maintained as of 2025)
    name_title = name.replace("-", "_").title().replace("_", "-")
    candidates = [
        f"bartowski/{name_title}-GGUF",
        f"TheBloke/{name_title}-GGUF",
        f"lmstudio-community/{name_title}-GGUF",
    ]
    # Return first candidate — caller validates via HF API
    return candidates[0] if candidates else None


def recommend_runtime(
    scale: ScaleProfile,
    model_id: str,
    hw: HardwareProfile,
    model_params_b: float = 0.0,
) -> RuntimeRecommendation:
    """
    Produce a concrete RuntimeRecommendation for a given scale + model.

    Applies Fix 3: PCIe multi-GPU prefers single GPU when model fits.
    """
    tier = scale.tier
    tp = scale.recommended_tp
    pp = scale.recommended_pp
    gpu_ids = [g.index for g in []] # will be filled below
    if hasattr(hw, 'all_gpus') and hw.all_gpus:
        gpu_ids = [g.index for g in hw.all_gpus]

    # ── Fix 3: PCIe single-GPU preference ───────────────────────────────────
    if (
        scale.interconnect == "pcie_x16" or scale.interconnect == "pcie_x8"
    ) and scale.gpu_count > 1 and model_params_b > 0:
        single_vram = hw.all_gpus[0].vram_total_gb if hw.all_gpus else 0.0
        vram_needed, _ = calculate_vram_gb(model_params_b, "Q4_K_M")
        if single_vram >= vram_needed:
            # Single GPU wins on PCIe for inference
            return RuntimeRecommendation(
                runtime="ollama",
                launch_args=[],
                tensor_parallel=1,
                pipeline_parallel=1,
                explanation=(
                    f"Single GPU ({single_vram:.0f}GB) is faster than "
                    f"PCIe multi-GPU for {model_id} ({vram_needed:.1f}GB needed). "
                    "Ollama on GPU 0 recommended."
                ),
                gpu_device_ids=[gpu_ids[0]] if gpu_ids else [],
                install_hint="ollama already installed or: curl -fsSL https://ollama.com/install.sh | sh",
            )

    # ── Tier-based routing ───────────────────────────────────────────────────

    # cpu_cluster must come first — gpu_count=0 would otherwise match <= 1 below
    if tier == "cpu_cluster":
        threads = max(hw.cpu_cores_physical, 1)
        return RuntimeRecommendation(
            runtime="llamacpp",
            launch_args=[
                "--model", model_id,
                "--threads", str(threads),
                "--ctx-size", "4096",
                "--host", "0.0.0.0",
                "--port", "8080",
            ],
            env_vars={"OMP_NUM_THREADS": str(threads)},
            explanation=f"CPU-only: llama.cpp with {threads} threads, no GPU acceleration",
            install_hint="pip install llama-cpp-python",
            gpu_device_ids=[],
        )

    if tier in ("laptop", "workstation") or scale.gpu_count <= 1:
        if hw.is_apple_silicon:
            return RuntimeRecommendation(
                runtime="ollama",
                launch_args=[],
                tensor_parallel=1,
                explanation="Apple Silicon: Ollama with Metal acceleration (best UX, built-in MLX backend)",
                install_hint="brew install ollama",
                gpu_device_ids=[],
            )
        return RuntimeRecommendation(
            runtime="ollama",
            launch_args=[],
            tensor_parallel=1,
            explanation=(
                f"{'Laptop' if tier == 'laptop' else 'Workstation'}: Ollama for simplicity. "
                f"{hw.gpu_vram_total_gb:.0f}GB VRAM available."
            ),
            install_hint="curl -fsSL https://ollama.com/install.sh | sh",
            gpu_device_ids=gpu_ids[:1],
        )


    if tier in ("multi_gpu", "server"):
        vllm_args = [
            "serve", model_id,
            "--tensor-parallel-size", str(tp),
            "--gpu-memory-utilization", "0.90",
            "--host", "0.0.0.0",
            "--port", "8000",
        ]
        if pp > 1:
            vllm_args += ["--pipeline-parallel-size", str(pp)]
        ic_note = "NVLink" if scale.interconnect == "nvlink" else f"PCIe (~60% NVLink efficiency)"
        return RuntimeRecommendation(
            runtime="vllm",
            launch_args=vllm_args,
            env_vars={"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in gpu_ids)},
            tensor_parallel=tp,
            pipeline_parallel=pp,
            explanation=(
                f"{scale.gpu_count} GPUs via {ic_note}. "
                f"vLLM tensor-parallel={tp}"
                + (f", pipeline-parallel={pp}" if pp > 1 else "")
                + f". Pooled VRAM: {scale.pooled_vram_gb:.0f}GB"
            ),
            install_hint="pip install vllm",
            gpu_device_ids=gpu_ids,
        )

    if tier == "hyperscale":
        vllm_args = [
            "serve", model_id,
            "--tensor-parallel-size", str(tp),
            "--pipeline-parallel-size", str(pp),
            "--distributed-executor-backend", "ray",
            "--gpu-memory-utilization", "0.92",
        ]
        return RuntimeRecommendation(
            runtime="ray_vllm",
            launch_args=vllm_args,
            env_vars={
                "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in gpu_ids),
                "RAY_BACKEND_LOG_LEVEL": "warning",
            },
            tensor_parallel=tp,
            pipeline_parallel=pp,
            explanation=(
                f"Hyperscale: {scale.gpu_count} GPUs. "
                f"Ray + vLLM distributed. TP={tp}, PP={pp}. "
                f"Pooled VRAM: {scale.pooled_vram_gb:.0f}GB"
            ),
            install_hint="pip install vllm ray",
            gpu_device_ids=gpu_ids,
        )

    # Fallback
    return RuntimeRecommendation(
        runtime="ollama",
        explanation="Default: Ollama (unrecognized scale tier)",
    )
