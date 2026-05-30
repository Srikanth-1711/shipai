"""
Tests for the hardware intelligence upgrades:
- AMD/Intel GPU detection
- VRAM calculator from first principles
- Speed estimator (tok/s)
- Benchmark fetcher
- MoE active parameter extraction
- Display output format
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

os.environ["SHIPAI_BOOTSTRAP_ONLY"] = "1"

from app.install.benchmark_fetcher import (
    BenchmarkScore,
    apply_decay,
    fetch_all_benchmarks,
    load_bundled_benchmark_fallback,
    merge_scores,
    recency_decay,
)
from app.install.hardware_profile import (
    GPU_BANDWIDTH_GB_S,
    _detect_amd_gpu,
    _detect_cpu_features,
    _detect_intel_gpu,
    get_bandwidth,
)
from app.install.speed_estimator import (
    BACKEND_EFFICIENCY,
    QUANT_EFFICIENCY,
    estimate_tokens_per_second,
    format_speed_display,
    infer_backend,
)
from app.install.tools.feasibility_filter import (
    FeasibilityResult,
    is_feasible,
    parse_params_billions,
)
from app.install.types import HardwareProfile
from app.install.vram_calculator import (
    KNOWN_MOE_MODELS,
    QUANT_BYTES,
    QUANT_QUALITY,
    calculate_vram_gb,
    extract_active_params,
    get_quant_bytes,
    get_quant_quality,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE DETECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAMDDetection:
    """Tests for AMD GPU detection via rocm-smi, lspci, and WMI."""

    def test_amd_detected_via_rocm_smi(self):
        mock_output = json.dumps({
            "card0": {
                "Card series": "AMD Radeon RX 7900 XTX",
                "VRAM Total Memory (B)": 25769803776,
            }
        })
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = mock_output
            mock_run.return_value = mock_result

            name, vram_gb, has_rocm = _detect_amd_gpu()

        assert name is not None
        assert "Radeon" in name or "AMD" in name
        assert vram_gb > 0
        assert has_rocm is True

    def test_amd_detected_via_lspci(self):
        lspci_output = (
            "01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. "
            "[AMD/ATI] Navi 31 [Radeon RX 7900 XTX/XT]"
        )
        with patch("subprocess.run") as mock_run:
            # rocm-smi fails, lspci succeeds
            def side_effect(cmd, **kwargs):
                result = MagicMock()
                if cmd[0] == "rocm-smi":
                    raise FileNotFoundError()
                elif cmd[0] == "lspci":
                    result.returncode = 0
                    result.stdout = lspci_output
                return result

            mock_run.side_effect = side_effect

            with patch(
                "app.install.hardware_profile._read_amd_vram_from_sysfs",
                return_value=24.0,
            ):
                name, vram_gb, has_rocm = _detect_amd_gpu()

        assert name is not None
        assert vram_gb == 24.0
        assert has_rocm is False

    @patch("platform.system", return_value="Windows")
    def test_amd_detected_via_windows_wmi(self, mock_sys):
        wmi_output = json.dumps([{
            "Name": "AMD Radeon RX 6700 XT",
            "AdapterRAM": 12884901888,
        }])
        with patch("subprocess.run") as mock_run:
            def side_effect(cmd, **kwargs):
                result = MagicMock()
                if cmd[0] in ("rocm-smi", "lspci"):
                    raise FileNotFoundError()
                elif cmd[0] == "powershell":
                    result.returncode = 0
                    result.stdout = wmi_output
                return result

            mock_run.side_effect = side_effect
            name, vram_gb, has_rocm = _detect_amd_gpu()

        assert name is not None
        assert "AMD" in name or "Radeon" in name
        assert vram_gb > 0


class TestIntelDetection:
    """Tests for Intel Arc / Xe GPU detection."""

    def test_intel_arc_detected_via_xpu_smi(self):
        mock_output = json.dumps({
            "device_list": [{
                "device_name": "Intel Arc A770",
                "memory_physical_size_byte": 17179869184,
            }]
        })
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = mock_output
            mock_run.return_value = mock_result

            name, vram_gb = _detect_intel_gpu()

        assert name is not None
        assert "Arc" in name or "Intel" in name
        assert vram_gb > 0


class TestBandwidthLookup:
    """Tests for GPU memory bandwidth table."""

    def test_known_gpu_bandwidth_lookup(self):
        assert get_bandwidth("NVIDIA GeForce RTX 4090") == 1008
        assert get_bandwidth("RTX 3080") == 760
        assert get_bandwidth("GTX 1650") == 128

    def test_amd_bandwidth_lookup(self):
        assert get_bandwidth("AMD Radeon RX 7900 XTX") == 960
        assert get_bandwidth("RX 6700 XT") == 384

    def test_apple_bandwidth_lookup(self):
        assert get_bandwidth("Apple M3 Max") == 400
        assert get_bandwidth("M1") == 68

    def test_intel_arc_bandwidth(self):
        assert get_bandwidth("Intel Arc A770") == 512

    def test_unknown_gpu_returns_conservative_bandwidth(self):
        bw = get_bandwidth("Some Unknown GPU XYZ")
        assert bw == 100.0  # conservative floor, not 0

    def test_empty_name_returns_default(self):
        assert get_bandwidth("") == 100.0
        assert get_bandwidth(None) == 100.0

    def test_bandwidth_specificity(self):
        """More specific name should match before less specific."""
        assert get_bandwidth("RTX 4080 SUPER") == 736  # not 717
        assert get_bandwidth("RTX 4080") == 717

    def test_datacenter_gpus(self):
        assert get_bandwidth("NVIDIA A100 80GB") == 2039
        assert get_bandwidth("H100") == 3352


class TestCPUFeatures:
    """Tests for CPU feature detection."""

    @patch("platform.system", return_value="Linux")
    def test_linux_detects_avx2(self, _):
        proc_cpuinfo = "flags\t\t: fpu vme de sse sse2 avx avx2 fma\n"
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            mock_open.return_value.read = MagicMock(return_value=proc_cpuinfo)
            features = _detect_cpu_features()
        assert "avx2" in features
        assert "avx" in features


# ═══════════════════════════════════════════════════════════════════════════════
# VRAM CALCULATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestQuantBytes:
    """Tests for quantization bytes-per-weight table."""

    def test_q4_km_bytes_per_weight(self):
        assert QUANT_BYTES["Q4_K_M"] == 0.563

    def test_f16_is_2_bytes(self):
        assert QUANT_BYTES["F16"] == 2.000

    def test_q8_0_is_1_byte(self):
        assert QUANT_BYTES["Q8_0"] == 1.000

    def test_unknown_quant_defaults_to_q4km(self):
        assert get_quant_bytes("UNKNOWN_QUANT") == 0.563

    def test_case_insensitive(self):
        assert get_quant_bytes("q4_k_m") == 0.563
        assert get_quant_bytes("Q4_K_M") == 0.563


class TestQuantQuality:
    """Tests for quality retention factors."""

    def test_f16_is_lossless(self):
        assert QUANT_QUALITY["F16"] == 1.00

    def test_q4_km_sweet_spot(self):
        assert QUANT_QUALITY["Q4_K_M"] == 0.92

    def test_q2_k_lower_quality(self):
        assert QUANT_QUALITY["Q2_K"] < QUANT_QUALITY["Q4_K_M"]

    def test_quality_increases_with_quant_level(self):
        assert QUANT_QUALITY["Q2_K"] < QUANT_QUALITY["Q4_K_M"] < QUANT_QUALITY["Q6_K"] < QUANT_QUALITY["Q8_0"]


class TestVRAMCalculation:
    """Tests for VRAM calculation from model parameters."""

    def test_llama31_8b_within_known_range(self):
        """Llama 3.1 8B Q4_K_M at 4K ctx should need ~5-6GB VRAM."""
        vram, conf = calculate_vram_gb(
            params_b=8.0,
            quant="Q4_K_M",
            context_length=4096,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
        )
        assert 4.5 <= vram <= 7.0, f"Expected 4.5-7.0 GB, got {vram}"
        assert conf == "high"

    def test_medium_confidence_without_arch_info(self):
        vram, conf = calculate_vram_gb(params_b=8.0, quant="Q4_K_M")
        assert conf == "medium"
        assert vram > 0

    def test_moe_uses_total_for_weights(self):
        """Mixtral 8x7B: weights need 46.7B params worth of memory."""
        vram_moe, _ = calculate_vram_gb(
            params_b=46.7,
            quant="Q4_K_M",
            is_moe=True,
            active_params_b=12.9,
        )
        # Should be larger than a 12.9B dense model
        vram_dense, _ = calculate_vram_gb(params_b=12.9, quant="Q4_K_M")
        assert vram_moe > vram_dense, (
            f"MoE ({vram_moe}GB) should need more VRAM than dense ({vram_dense}GB)"
        )

    def test_kv_cache_scales_with_context(self):
        v4k, _ = calculate_vram_gb(7.0, "Q4_K_M", 4096)
        v32k, _ = calculate_vram_gb(7.0, "Q4_K_M", 32768)
        assert v32k > v4k * 1.5, f"32K ctx ({v32k}) should need >1.5x of 4K ({v4k})"

    def test_higher_quant_needs_more_vram(self):
        v_q4, _ = calculate_vram_gb(8.0, "Q4_K_M")
        v_f16, _ = calculate_vram_gb(8.0, "F16")
        assert v_f16 > v_q4 * 2, f"F16 ({v_f16}) should need >2x of Q4 ({v_q4})"

    def test_zero_params_low_confidence(self):
        vram, conf = calculate_vram_gb(0.0, "Q4_K_M")
        assert conf == "low"

    def test_small_model_reasonable_vram(self):
        """TinyLlama 1.1B Q4_K_M should need <2GB."""
        vram, _ = calculate_vram_gb(1.1, "Q4_K_M")
        assert vram < 2.0, f"TinyLlama should need <2GB, got {vram}"


# ═══════════════════════════════════════════════════════════════════════════════
# MOE EXTRACTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMoEExtraction:
    """Tests for MoE active parameter extraction."""

    def test_qwen3_active_from_name(self):
        active, is_moe = extract_active_params("Qwen3-30B-A3B", 30.0)
        assert active == 3.0
        assert is_moe is True

    def test_qwen3_235b_from_name(self):
        active, is_moe = extract_active_params("qwen3-235b-a22b", 235.0)
        assert active == 22.0
        assert is_moe is True

    def test_mixtral_from_lookup(self):
        active, is_moe = extract_active_params(
            "mistralai/Mixtral-8x7B-Instruct-v0.1", 46.7
        )
        assert active == 12.9
        assert is_moe is True

    def test_dense_model_returns_total(self):
        active, is_moe = extract_active_params(
            "meta-llama/Llama-3.1-8B", 8.0
        )
        assert active == 8.0
        assert is_moe is False

    def test_gguf_metadata_priority(self):
        """GGUF metadata should take highest priority."""
        active, is_moe = extract_active_params(
            "some-model-8x7b",
            56.0,
            gguf_metadata={
                "llm.expert_count": 8,
                "llm.expert_used_count": 2,
            },
        )
        assert is_moe is True
        assert active == 14.0  # 56 * 2/8

    def test_deepseek_v3_from_lookup(self):
        active, is_moe = extract_active_params("deepseek-v3-somevariant", 671.0)
        assert active == 37.0
        assert is_moe is True


# ═══════════════════════════════════════════════════════════════════════════════
# SPEED ESTIMATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpeedEstimation:
    """Tests for tok/s speed estimation."""

    def test_gtx1650_llama3_speed_range(self):
        """GTX 1650 + llama3:8b Q4_K_M = ~10-15 tok/s in practice."""
        tps, low, high, conf = estimate_tokens_per_second(
            params_b=8.0,
            quant="Q4_K_M",
            bandwidth_gb_s=128.0,  # GTX 1650
            backend="nvidia_cuda",
        )
        assert 5 <= tps <= 25, f"Expected 5-25 tok/s, got {tps}"
        assert conf == "~"  # known GPU

    def test_rtx4090_much_faster_than_gtx1650(self):
        tps_4090, _, _, _ = estimate_tokens_per_second(
            8.0, "Q4_K_M", 1008.0, "nvidia_cuda",
        )
        tps_1650, _, _, _ = estimate_tokens_per_second(
            8.0, "Q4_K_M", 128.0, "nvidia_cuda",
        )
        assert tps_4090 > tps_1650 * 5, (
            f"RTX 4090 ({tps_4090}) should be >5x GTX 1650 ({tps_1650})"
        )

    def test_moe_faster_than_dense_same_total_params(self):
        """Mixtral 8x7B (active 12.9B) should be faster than 46.7B dense."""
        tps_moe, _, _, _ = estimate_tokens_per_second(
            params_b=46.7,
            quant="Q4_K_M",
            bandwidth_gb_s=760.0,
            backend="nvidia_cuda",
            is_moe=True,
            active_params_b=12.9,
        )
        tps_dense, _, _, _ = estimate_tokens_per_second(
            params_b=46.7,
            quant="Q4_K_M",
            bandwidth_gb_s=760.0,
            backend="nvidia_cuda",
        )
        assert tps_moe > tps_dense * 2, (
            f"MoE ({tps_moe}) should be >2x dense ({tps_dense})"
        )

    def test_unknown_gpu_shows_low_confidence(self):
        _, _, _, conf = estimate_tokens_per_second(
            8.0, "Q4_K_M", 100.0, "nvidia_cuda",
        )
        assert conf == "?"  # 100.0 = default unknown bandwidth

    def test_cpu_much_slower_than_gpu(self):
        tps_gpu, _, _, _ = estimate_tokens_per_second(
            8.0, "Q4_K_M", 760.0, "nvidia_cuda",
        )
        tps_cpu, _, _, _ = estimate_tokens_per_second(
            8.0, "Q4_K_M", 30.0, "cpu_avx2",
        )
        assert tps_gpu > tps_cpu * 10

    def test_low_high_range(self):
        _, low, high, _ = estimate_tokens_per_second(
            8.0, "Q4_K_M", 760.0, "nvidia_cuda",
        )
        assert low < high
        assert low > 0

    def test_amd_slightly_slower_than_nvidia(self):
        tps_nv, _, _, _ = estimate_tokens_per_second(
            8.0, "Q4_K_M", 960.0, "nvidia_cuda",
        )
        tps_amd, _, _, _ = estimate_tokens_per_second(
            8.0, "Q4_K_M", 960.0, "amd_rocm",
        )
        assert tps_amd < tps_nv  # ROCm efficiency < CUDA


class TestInferBackend:
    """Tests for backend inference from hardware vendor."""

    def test_nvidia_returns_cuda(self):
        assert infer_backend("nvidia") == "nvidia_cuda"

    def test_amd_returns_rocm(self):
        assert infer_backend("amd") == "amd_rocm"

    def test_apple_returns_metal(self):
        assert infer_backend("apple") == "apple_metal"

    def test_cpu_with_avx2(self):
        assert infer_backend("none", ["avx2"]) == "cpu_avx2"

    def test_cpu_with_avx512(self):
        assert infer_backend("none", ["avx512f", "avx2"]) == "cpu_avx512"

    def test_cpu_no_features(self):
        assert infer_backend("none", []) == "cpu_basic"


class TestSpeedDisplay:
    """Tests for human-readable display output."""

    def test_display_output_contains_tok_s(self):
        display = format_speed_display(
            "gemma2:9b",
            median_tps=20.0,
            low_tps=16.0,
            high_tps=25.0,
            vram_needed_gb=5.0,
            vram_available_gb=8.0,
            confidence="~",
        )
        assert "tok/s" in display
        assert "gemma2:9b" in display

    def test_display_output_contains_vram_gb(self):
        display = format_speed_display(
            "llama3:8b",
            median_tps=15.0,
            low_tps=12.0,
            high_tps=19.0,
            vram_needed_gb=5.5,
            vram_available_gb=8.0,
            confidence="~",
        )
        assert "GB" in display
        assert "fits" in display

    def test_offload_display(self):
        display = format_speed_display(
            "mixtral:8x7b",
            median_tps=5.0,
            low_tps=4.0,
            high_tps=6.0,
            vram_needed_gb=26.0,
            vram_available_gb=8.0,
            confidence="~",
        )
        assert "offload" in display

    def test_cpu_only_display(self):
        display = format_speed_display(
            "phi3:mini",
            median_tps=3.0,
            low_tps=2.0,
            high_tps=4.0,
            vram_needed_gb=2.0,
            vram_available_gb=0.0,
            confidence="?",
        )
        assert "CPU" in display


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK FETCHER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecencyDecay:
    """Tests for benchmark score recency decay."""

    def test_recent_score_no_decay(self):
        assert recency_decay(datetime.now() - timedelta(days=30)) == 1.00

    def test_six_months_slight_decay(self):
        assert recency_decay(datetime.now() - timedelta(days=200)) == 0.90

    def test_one_year_moderate_decay(self):
        assert recency_decay(datetime.now() - timedelta(days=400)) == 0.75

    def test_two_year_heavy_decay(self):
        assert recency_decay(datetime.now() - timedelta(days=800)) == 0.60

    def test_very_old_floor(self):
        assert recency_decay(datetime.now() - timedelta(days=1200)) == 0.50

    def test_none_date_conservative(self):
        assert recency_decay(None) == 0.60


class TestScoreMerging:
    """Tests for benchmark score merge logic."""

    def test_current_tier_overrides_frozen(self):
        frozen = {"model/x": BenchmarkScore(score=85.0, source="leaderboard", tier="frozen")}
        current = {"model/x": BenchmarkScore(score=78.0, source="arena", tier="current")}
        result = merge_scores(frozen, current)
        assert result["model/x"].tier == "current"
        assert result["model/x"].score == 78.0

    def test_same_tier_higher_score_wins(self):
        a = {"model/x": BenchmarkScore(score=70.0, source="a", tier="frozen")}
        b = {"model/x": BenchmarkScore(score=80.0, source="b", tier="frozen")}
        result = merge_scores(a, b)
        assert result["model/x"].score == 80.0

    def test_new_model_added(self):
        a = {"model/x": BenchmarkScore(score=70.0, source="a", tier="frozen")}
        b = {"model/y": BenchmarkScore(score=80.0, source="b", tier="current")}
        result = merge_scores(a, b)
        assert "model/x" in result
        assert "model/y" in result


class TestApplyDecay:
    """Tests for applying recency decay to benchmark scores."""

    def test_frozen_score_decayed_after_2_years(self):
        score = BenchmarkScore(
            score=80.0,
            source="test",
            tier="frozen",
            date=datetime.now() - timedelta(days=800),
        )
        decayed = apply_decay(score)
        assert decayed.score < 80.0 * 0.80
        assert decayed.decay_applied > 0

    def test_current_tier_not_decayed(self):
        score = BenchmarkScore(
            score=80.0,
            source="test",
            tier="current",
            date=datetime.now() - timedelta(days=800),
        )
        result = apply_decay(score)
        assert result.score == 80.0


class TestBenchmarkFetcher:
    """Tests for the full benchmark fetching pipeline."""

    def test_offline_fallback_never_crashes(self):
        """Even with no internet, fetch_all_benchmarks should return bundled data."""
        result = asyncio.get_event_loop().run_until_complete(
            fetch_all_benchmarks(offline=True)
        )
        assert result.used_fallback is True
        # Should have at least some scores from bundled matrix
        assert len(result.scores) > 0

    def test_bundled_fallback_has_scores(self):
        scores = load_bundled_benchmark_fallback()
        assert len(scores) > 0
        # All scores should have reasonable values
        for key, sc in scores.items():
            assert 0 <= sc.score <= 100, f"Score for {key} out of range: {sc.score}"


# ═══════════════════════════════════════════════════════════════════════════════
# FEASIBILITY FILTER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeasibilityIntegration:
    """Tests that the upgraded feasibility filter works with new math."""

    def _make_hw(self, **overrides) -> HardwareProfile:
        defaults = dict(
            os_name="Linux",
            os_version="5.15",
            architecture="x86_64",
            cpu_name="AMD Ryzen 9 5900X",
            cpu_cores_physical=12,
            cpu_cores_logical=24,
            ram_total_gb=64.0,
            ram_available_gb=48.0,
            disk_free_gb=500.0,
            disk_total_gb=1000.0,
            has_gpu=True,
            gpu_name="NVIDIA GeForce RTX 3080",
            gpu_vram_total_gb=10.0,
            gpu_vram_free_gb=9.0,
            has_cuda=True,
            gpu_vendor="nvidia",
            gpu_bandwidth_gb_s=760.0,
            cpu_features=["avx2", "fma"],
            effective_vram_gb=8.5,
            effective_ram_gb=54.4,
            max_model_params_b=14.2,
        )
        defaults.update(overrides)
        return HardwareProfile(**defaults)

    def test_small_model_feasible(self):
        hw = self._make_hw()
        result = is_feasible("gemma2:2b", hw, size_gb=1.6)
        assert result.feasible is True
        assert result.fits_vram is True
        assert result.estimated_tps > 0

    def test_huge_model_infeasible_on_small_gpu(self):
        hw = self._make_hw(
            gpu_vram_total_gb=4.0,
            gpu_vram_free_gb=3.5,
            effective_vram_gb=3.0,
            ram_total_gb=8.0,
            ram_available_gb=6.0,
            effective_ram_gb=6.8,
        )
        result = is_feasible("llama3:70b", hw, size_gb=40.0)
        assert result.feasible is False

    def test_has_vram_confidence(self):
        hw = self._make_hw()
        result = is_feasible("qwen2.5:3b", hw)
        assert result.vram_confidence in ("high", "medium", "low")

    def test_has_tps_confidence(self):
        hw = self._make_hw()
        result = is_feasible("qwen2.5:3b", hw)
        assert result.tps_confidence in ("~", "?")

    def test_params_parsed_from_name(self):
        assert parse_params_billions("llama3:8b") == 8.0
        assert parse_params_billions("mixtral-8x7b") == 7.0  # regex finds '7b'
        assert parse_params_billions("qwen2.5-coder:7b") == 7.0
