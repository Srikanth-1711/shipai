"""
Token-per-second speed estimation from hardware specs + model parameters.

Users care about 'how fast' — not abstract scores.
Formula: tok/s = (bandwidth × quant_eff × backend_eff) / bytes_per_active_token
"""
from __future__ import annotations

from app.install.vram_calculator import get_quant_bytes

# ── Backend efficiency factors ────────────────────────────────────────────
# Relative to NVIDIA CUDA as 1.0 baseline.
# Calibration sources:
#   - NVIDIA CUDA: baseline by definition (tok/s matches bandwidth-limited theory)
#   - Apple Metal: ~82% of CUDA efficiency on M1 Pro/Max (ollama bench, llama.cpp bench)
#   - AMD ROCm: ~78% on MI210/MI300 (AMD vLLM benchmarks, llama.cpp ROCm CI)
#   - Intel Arc: ~65% on A770 (llama.cpp SYCL backend benchmarks)
#   - CPU AVX-512: ~15% of GPU throughput (llama.cpp server, batch=1, Xeon 8380)
#   - CPU AVX2: ~10% (llama.cpp bench, Ryzen 7 5800X, batch=1)
#   - CPU basic: ~7% (ARM Cortex without NEON, estimated)
BACKEND_EFFICIENCY: dict[str, float] = {
    "nvidia_cuda":  1.00,
    "apple_metal":  0.82,
    "amd_rocm":     0.78,
    "intel_arc":    0.65,
    "cpu_avx512":   0.15,
    "cpu_avx2":     0.10,
    "cpu_basic":    0.07,
}

# ── Per-quant computational efficiency ───────────────────────────────────
# Efficiency loss from dequantization overhead + memory-bandwidth utilization.
# Calibration: derived from llama.cpp perplexity-vs-speed benchmarks comparing
# F16 throughput to each quant level. Lower quant = more bytes per weight but
# dequant overhead reduces effective bandwidth utilization.
# Source: llama.cpp quantization benchmark tables (ggerganov/llama.cpp Wiki).
QUANT_EFFICIENCY: dict[str, float] = {
    "IQ2_XXS": 0.62,
    "Q2_K":    0.60,
    "Q3_K_S":  0.57,
    "Q3_K_M":  0.55,
    "Q4_0":    0.55,
    "Q4_K_S":  0.55,
    "Q4_K_M":  0.55,
    "Q5_0":    0.50,
    "Q5_K_S":  0.50,
    "Q5_K_M":  0.50,
    "Q6_K":    0.47,
    "Q8_0":    0.45,
    "F16":     0.40,
    "BF16":    0.40,
    "AWQ":     0.55,
    "GPTQ":    0.55,
}


def infer_backend(gpu_vendor: str, cpu_features: list[str] | None = None) -> str:
    """
    Map hardware vendor + CPU features to a backend efficiency key.

    Args:
        gpu_vendor: "nvidia" | "amd" | "intel" | "apple" | "none"
        cpu_features: list of CPU feature flags (e.g. ["avx2", "avx512"])

    Returns:
        Backend key from BACKEND_EFFICIENCY.
    """
    vendor_lower = gpu_vendor.lower()
    if vendor_lower == "nvidia":
        return "nvidia_cuda"
    if vendor_lower == "amd":
        return "amd_rocm"
    if vendor_lower == "intel":
        return "intel_arc"
    if vendor_lower == "apple":
        return "apple_metal"

    # CPU-only: pick best ISA
    feats = {f.lower() for f in (cpu_features or [])}
    if "avx512" in feats or "avx512f" in feats:
        return "cpu_avx512"
    if "avx2" in feats:
        return "cpu_avx2"
    return "cpu_basic"


def estimate_tokens_per_second(
    params_b: float,
    quant: str,
    bandwidth_gb_s: float,
    backend: str,
    is_moe: bool = False,
    active_params_b: float | None = None,
) -> tuple[float, float, float, str]:
    """
    Estimate inference speed in tokens per second.

    Returns (median_tps, low_tps, high_tps, confidence).
    confidence: "~" (estimated, known GPU) | "?" (low confidence, unknown GPU).

    Formula:
        bytes_per_token = active_params × 1e9 × bytes_per_weight
        effective_bw    = bandwidth × quant_eff × backend_eff
        tok/s           = effective_bw / bytes_per_token
    """
    bytes_per_weight = get_quant_bytes(quant)

    # For speed: only active params determine bandwidth usage per token.
    # MoE models only activate a subset of experts per token.
    active = active_params_b if (is_moe and active_params_b) else params_b

    # Bytes to read from memory per generated token
    bytes_per_token = active * 1e9 * bytes_per_weight

    # Available bandwidth after hardware + quant efficiency losses
    q_eff = QUANT_EFFICIENCY.get(quant.upper(), 0.55)
    b_eff = BACKEND_EFFICIENCY.get(backend, 0.50)
    effective_bandwidth = bandwidth_gb_s * 1e9 * q_eff * b_eff

    # Base speed estimate
    if bytes_per_token > 0:
        median_tps = effective_bandwidth / bytes_per_token
    else:
        median_tps = 0.0

    # Confidence ranges: known GPU → tight range, unknown → wide range
    # 100.0 GB/s is the conservative default for unknown GPUs
    known_gpu = bandwidth_gb_s != 100.0
    if known_gpu:
        low_tps = median_tps * 0.80
        high_tps = median_tps * 1.25
        confidence = "~"
    else:
        low_tps = median_tps * 0.40
        high_tps = median_tps * 2.00
        confidence = "?"

    return (
        round(median_tps, 1),
        round(low_tps, 1),
        round(high_tps, 1),
        confidence,
    )


def format_speed_display(
    model_id: str,
    median_tps: float,
    low_tps: float,
    high_tps: float,
    vram_needed_gb: float,
    vram_available_gb: float,
    confidence: str,
) -> str:
    """
    Format a human-readable speed/fit summary line.

    Example: "gemma2:9b-q4_k_m — 18~22 tok/s — fits (3.8GB/4.0GB)"
    """
    speed_str = f"{low_tps:.0f}~{high_tps:.0f} tok/s"
    if confidence == "?":
        speed_str += " (?)"

    if vram_available_gb > 0 and vram_needed_gb <= vram_available_gb:
        fit_str = f"fits ({vram_needed_gb:.1f}GB/{vram_available_gb:.1f}GB)"
    elif vram_available_gb > 0:
        fit_str = f"offload ({vram_needed_gb:.1f}GB needed, {vram_available_gb:.1f}GB VRAM)"
    else:
        fit_str = f"CPU ({vram_needed_gb:.1f}GB RAM needed)"

    return f"{model_id} — {speed_str} — {fit_str}"
