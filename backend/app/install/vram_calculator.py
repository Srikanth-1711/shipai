"""
VRAM calculation from first principles + MoE active parameter extraction.

Replaces static min_vram_gb JSON lookups with computed VRAM from model
architecture parameters. Formula source: Ollama GPU docs and llama.cpp.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# ── Quantization bytes per weight (authoritative table) ──────────────────────
# Sources: llama.cpp ggml-quants.c, Ollama GPU docs
QUANT_BYTES: dict[str, float] = {
    "IQ1_S":   0.188,
    "IQ1_M":   0.219,
    "IQ2_XXS": 0.250,
    "IQ2_XS":  0.281,
    "IQ2_S":   0.313,
    "IQ2_M":   0.344,
    "IQ3_XXS": 0.375,
    "IQ3_S":   0.406,
    "Q2_K":    0.313,
    "Q3_K_S":  0.375,
    "Q3_K_M":  0.406,
    "Q3_K_L":  0.438,
    "Q4_0":    0.500,
    "Q4_K_S":  0.500,
    "Q4_K_M":  0.563,
    "Q5_0":    0.625,
    "Q5_K_S":  0.625,
    "Q5_K_M":  0.656,
    "Q6_K":    0.750,
    "Q8_0":    1.000,
    "F16":     2.000,
    "BF16":    2.000,
    "F32":     4.000,
    "AWQ":     0.563,
    "GPTQ":    0.563,
}

# Per-quant quality retention (1.0 = lossless F16).
# Used to adjust benchmark scores for quantized models.
QUANT_QUALITY: dict[str, float] = {
    "IQ2_XXS": 0.60,
    "Q2_K":    0.72,
    "Q3_K_S":  0.78,
    "Q3_K_M":  0.82,
    "Q4_0":    0.88,
    "Q4_K_S":  0.90,
    "Q4_K_M":  0.92,
    "Q5_K_S":  0.95,
    "Q5_K_M":  0.96,
    "Q6_K":    0.98,
    "Q8_0":    0.99,
    "F16":     1.00,
    "BF16":    1.00,
    "AWQ":     0.92,
    "GPTQ":    0.91,
}

# ── Known MoE models — active parameter lookup ──────────────────────────────
KNOWN_MOE_MODELS: dict[str, float] = {
    "mixtral-8x7b":    12.9,
    "mixtral-8x22b":   39.1,
    "qwen3-30b-a3b":    3.0,
    "qwen3-235b-a22b": 22.0,
    "deepseek-v2":     21.0,
    "deepseek-v3":     37.0,
    "llama4-scout":    17.0,
    "llama4-maverick": 17.0,
}


def get_quant_bytes(quant: str) -> float:
    """Look up bytes-per-weight for a quantization type, defaulting to Q4_K_M."""
    return QUANT_BYTES.get(quant.upper(), QUANT_BYTES["Q4_K_M"])


def get_quant_quality(quant: str) -> float:
    """Quality retention factor for a quantization (1.0 = lossless)."""
    return QUANT_QUALITY.get(quant.upper(), 0.90)


def calculate_vram_gb(
    params_b: float,
    quant: str,
    context_length: int = 4096,
    is_moe: bool = False,
    active_params_b: float | None = None,
    num_layers: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
) -> tuple[float, str]:
    """
    Calculate total VRAM needed to run a model.

    Returns (total_vram_gb, confidence_level).
    confidence: "high" (exact arch info) | "medium" (approximated KV) | "low" (guessed params).

    Formula:
        total = weight_memory + kv_cache + activation_memory + overhead
    """
    bytes_per_weight = get_quant_bytes(quant)

    # For MoE: VRAM must hold ALL weights even though only active experts
    # are computed per token. Weight memory always uses total params.
    weight_params = params_b

    # 1. Weight memory (GB)
    weights_gb = weight_params * bytes_per_weight

    # 2. KV cache memory (GB)
    if num_layers and num_kv_heads and head_dim:
        # Exact: 2 (K+V) × layers × kv_heads × head_dim × ctx_len × 2 bytes (fp16) / 1e9
        kv_gb = (
            2 * num_layers * num_kv_heads * head_dim * context_length * 2
        ) / 1e9
        confidence = "high"
    else:
        # Approximation: empirically ~0.125 GB/B params at 4K context
        ctx_scale = context_length / 4096
        kv_gb = weight_params * 0.125 * ctx_scale
        confidence = "medium"

    # 3. Activation memory — scales with active params for MoE
    active = active_params_b if (is_moe and active_params_b) else params_b
    activation_gb = active * bytes_per_weight * 0.1

    # 4. Framework overhead (fixed — CUDA context, buffers, etc.)
    overhead_gb = 0.5

    total = weights_gb + kv_gb + activation_gb + overhead_gb

    # If params_b was guessed (very small), mark low confidence
    if params_b <= 0:
        confidence = "low"

    return round(total, 2), confidence


def extract_active_params(
    model_id: str,
    total_params_b: float,
    gguf_metadata: dict[str, Any] | None = None,
) -> tuple[float, bool]:
    """
    Determine active parameters for inference (MoE-aware).

    Returns (active_params_b, is_moe).

    Priority:
    1. GGUF metadata (most accurate — llm.expert_count / llm.expert_used_count)
    2. Model name pattern: -a{N}b (Qwen3 style, e.g. Qwen3-30B-A3B → 3.0B)
    3. Known MoE lookup table
    4. Assume dense (active = total)
    """
    # Priority 1: GGUF metadata fields
    if gguf_metadata:
        expert_count = gguf_metadata.get("llm.expert_count", 0)
        experts_used = gguf_metadata.get("llm.expert_used_count", 0)
        if isinstance(expert_count, (int, float)) and isinstance(experts_used, (int, float)):
            if expert_count > 0 and experts_used > 0:
                active = total_params_b * (experts_used / expert_count)
                return round(active, 1), True

    # Priority 2: Name pattern -a{N}b (case-insensitive)
    match = re.search(r"-a(\d+(?:\.\d+)?)b", model_id.lower())
    if match:
        active_b = float(match.group(1))
        return active_b, True

    # Priority 3: Known MoE lookup
    model_lower = model_id.lower()
    for pattern, active_b in KNOWN_MOE_MODELS.items():
        if pattern in model_lower:
            return active_b, True

    # Priority 4: Dense model — active = total
    return total_params_b, False
