"""
Tests for scale + acquisition upgrade:
- Multi-GPU detection (all vendors)
- NVLink/PCIe topology
- VRAM pooling math
- Scale tier classification
- Runtime recommendation (with Fix 3: PCIe single-GPU preference)
- Deployment config generation
- Model acquisition chain (all sources)
- Model registration (GGUF + HF cache, Fix 2: safetensors fallback)
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import re

os.environ["SHIPAI_BOOTSTRAP_ONLY"] = "1"

from app.install.gpu_topology import (
    GPUTopology,
    calculate_pooled_vram,
    classify_scale_tier,
    detect_all_nvidia_gpus,
    detect_all_amd_gpus,
    detect_nvlink_topology,
    detect_gpu_topology,
    recommend_parallelism,
    POOLING_EFFICIENCY,
)
from app.install.scale_classifier import ScaleProfile, classify
from app.install.runtime_recommender import (
    RuntimeRecommendation,
    recommend_runtime,
    map_hf_to_ollama_name,
    find_gguf_repo,
)
from app.install.deployment_generator import (
    generate_docker_compose,
    generate_vllm_launch_script,
    generate_k8s_deployment,
    generate_systemd_unit,
    write_deployment_configs,
)
from app.install.model_registrar import (
    find_best_gguf_in_path,
    register_gguf_with_ollama,
    register_hf_cache,
    _ollama_available,
)
from app.install.model_acquirer import (
    AcquisitionResult,
    _pick_best_quant,
    _is_already_served,
    _hf_cli_available,
    acquire_model,
)
from app.install.types import GPUInfo, HardwareProfile, LLMRuntime, ModelInfo


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_hw(**overrides) -> HardwareProfile:
    defaults = dict(
        os_name="Linux", os_version="5.15", architecture="x86_64",
        cpu_name="AMD Ryzen 9 5900X", cpu_cores_physical=12, cpu_cores_logical=24,
        ram_total_gb=64.0, ram_available_gb=48.0,
        disk_free_gb=500.0, disk_total_gb=1000.0,
        has_gpu=True, gpu_name="NVIDIA RTX 4090",
        gpu_vram_total_gb=24.0, gpu_vram_free_gb=22.0,
        has_cuda=True, gpu_vendor="nvidia", gpu_bandwidth_gb_s=1008.0,
        effective_vram_gb=21.5, effective_ram_gb=54.0, max_model_params_b=35.0,
        gpu_count=1, total_vram_gb=21.5, topology_type="none", scale_tier="workstation",
    )
    defaults.update(overrides)
    return HardwareProfile(**defaults)


def _make_gpu(idx=0, name="RTX 4090", vram=24.0, vendor="nvidia", bw=1008.0) -> GPUInfo:
    return GPUInfo(
        index=idx, name=name, vram_total_gb=vram,
        vram_free_gb=vram * 0.9, vendor=vendor,
        bandwidth_gb_s=bw, has_cuda=(vendor == "nvidia"), has_rocm=(vendor == "amd"),
    )


def _make_topology(gpus, interconnect="none", tp=1, pp=1) -> GPUTopology:
    raw = sum(g.vram_total_gb for g in gpus)
    eff = POOLING_EFFICIENCY.get(interconnect, 1.0)
    pooled = raw * eff if len(gpus) > 1 else raw
    return GPUTopology(
        gpus=gpus, interconnect=interconnect,
        total_vram_gb=raw, pooled_vram_gb=pooled,
        recommended_tp_degree=tp, recommended_pp_degree=pp,
        scale_tier=classify_scale_tier(gpus, interconnect),
    )


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-GPU DETECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAllNvidiaGPUsParsed:
    """Verify ALL GPUs are parsed from nvidia-smi output."""

    def test_all_nvidia_gpus_parsed(self):
        """4-GPU system: all 4 must be detected (old code only read first line)."""
        nvidia_smi_output = "\n".join([
            "0, NVIDIA A100, 81920, 80000, GPU-aaa111",
            "1, NVIDIA A100, 81920, 79500, GPU-bbb222",
            "2, NVIDIA A100, 81920, 79000, GPU-ccc333",
            "3, NVIDIA A100, 81920, 78500, GPU-ddd444",
        ])
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.returncode = 0
            result.stdout = nvidia_smi_output
            mock_run.return_value = result
            gpus = detect_all_nvidia_gpus()

        assert len(gpus) == 4, f"Expected 4 GPUs, got {len(gpus)}"
        for i, gpu in enumerate(gpus):
            assert gpu.index == i
            assert gpu.vram_total_gb == pytest.approx(80.0, abs=0.1)
            assert gpu.vendor == "nvidia"
            assert gpu.has_cuda is True

    def test_single_gpu_still_works(self):
        """Single GPU system must still work."""
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.returncode = 0
            result.stdout = "0, RTX 4090, 24576, 23000, GPU-abc123"
            mock_run.return_value = result
            gpus = detect_all_nvidia_gpus()
        assert len(gpus) == 1
        assert gpus[0].name == "RTX 4090"

    def test_no_gpu_returns_empty(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            gpus = detect_all_nvidia_gpus()
        assert gpus == []


class TestNvlinkTopology:

    def test_nvlink_topology_detected(self):
        """NV2 markers in nvidia-smi topo output → nvlink detected."""
        topo_output = """
        GPU0    GPU1    GPU2    GPU3
GPU0    X       NV2     NV2     NV1
GPU1    NV2     X       NV1     NV2
GPU2    NV2     NV1     X       NV2
GPU3    NV1     NV2     NV2     X
"""
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.returncode = 0
            result.stdout = topo_output
            mock_run.return_value = result
            interconnect, bw = detect_nvlink_topology(4)

        assert interconnect == "nvlink"
        assert bw > 0

    def test_pcie_topology_fallback(self):
        """PIX markers → PCIe detected."""
        topo_output = """
        GPU0    GPU1
GPU0    X       PIX
GPU1    PIX     X
"""
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.returncode = 0
            result.stdout = topo_output
            mock_run.return_value = result
            interconnect, bw = detect_nvlink_topology(2)

        assert "pcie" in interconnect
        assert bw > 0

    def test_single_gpu_no_topology_needed(self):
        interconnect, bw = detect_nvlink_topology(1)
        assert interconnect == "none"
        assert bw == 0.0


class TestAMDMultiGPU:

    def test_amd_multi_gpu_detected(self):
        rocm_output = json.dumps({
            "card0": {
                "Card series": "AMD Radeon RX 7900 XTX",
                "VRAM Total Memory (B)": 25769803776,
            },
            "card1": {
                "Card series": "AMD Radeon RX 7900 XTX",
                "VRAM Total Memory (B)": 25769803776,
            },
        })
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.returncode = 0
            result.stdout = rocm_output
            mock_run.return_value = result
            gpus = detect_all_amd_gpus()

        assert len(gpus) == 2
        for gpu in gpus:
            assert gpu.vendor == "amd"
            assert gpu.vram_total_gb > 20


# ═══════════════════════════════════════════════════════════════════════════
# VRAM POOLING TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestVRAMPooling:

    def test_nvlink_pooling_92pct(self):
        """4×24GB NVLink → 88.3GB pooled (92% of 96GB)."""
        gpus = [_make_gpu(i, vram=24.0) for i in range(4)]
        pooled = calculate_pooled_vram(gpus, "nvlink")
        assert pooled == pytest.approx(96.0 * 0.92, abs=0.5)

    def test_pcie_pooling_78pct(self):
        """4×24GB PCIe → 74.9GB pooled (78% of 96GB)."""
        gpus = [_make_gpu(i, vram=24.0) for i in range(4)]
        pooled = calculate_pooled_vram(gpus, "pcie_x16")
        assert pooled == pytest.approx(96.0 * 0.78, abs=0.5)

    def test_single_gpu_no_overhead(self):
        """Single GPU → 100% VRAM, no overhead."""
        gpus = [_make_gpu(0, vram=24.0)]
        pooled = calculate_pooled_vram(gpus, "none")
        assert pooled == pytest.approx(24.0, abs=0.1)

    def test_empty_gpus_returns_zero(self):
        assert calculate_pooled_vram([], "nvlink") == 0.0

    def test_nvlink_more_efficient_than_pcie(self):
        gpus = [_make_gpu(i, vram=80.0) for i in range(8)]
        nvlink_pooled = calculate_pooled_vram(gpus, "nvlink")
        pcie_pooled = calculate_pooled_vram(gpus, "pcie_x16")
        assert nvlink_pooled > pcie_pooled


# ═══════════════════════════════════════════════════════════════════════════
# SCALE CLASSIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestScaleClassification:

    def test_1gpu_8gb_is_laptop(self):
        gpus = [_make_gpu(0, vram=8.0)]
        assert classify_scale_tier(gpus, "none") == "laptop"

    def test_1gpu_24gb_is_workstation(self):
        gpus = [_make_gpu(0, vram=24.0)]
        assert classify_scale_tier(gpus, "none") == "workstation"

    def test_2gpu_pcie_is_multi_gpu(self):
        gpus = [_make_gpu(i, vram=24.0) for i in range(2)]
        assert classify_scale_tier(gpus, "pcie_x16") == "multi_gpu"

    def test_4gpu_nvlink_is_server(self):
        """4 GPUs with NVLink → server tier."""
        gpus = [_make_gpu(i, vram=80.0) for i in range(4)]
        assert classify_scale_tier(gpus, "nvlink") == "server"

    def test_4gpu_pcie_is_server(self):
        gpus = [_make_gpu(i, vram=80.0) for i in range(4)]
        assert classify_scale_tier(gpus, "pcie_x16") == "server"

    def test_8gpu_is_hyperscale(self):
        gpus = [_make_gpu(i, vram=80.0) for i in range(8)]
        tier = classify_scale_tier(gpus, "nvlink")
        assert tier == "hyperscale"

    def test_no_gpu_is_laptop_by_default(self):
        assert classify_scale_tier([], "none") == "laptop"

    def test_no_gpu_big_cpu_is_cpu_cluster(self):
        assert classify_scale_tier([], "none", cpu_cores=64, ram_gb=512.0) == "cpu_cluster"


# ═══════════════════════════════════════════════════════════════════════════
# PARALLELISM RECOMMENDATION (Fix 3)
# ═══════════════════════════════════════════════════════════════════════════

class TestParallelismRecommendation:

    def test_nvlink_4gpu_tp4(self):
        gpus = [_make_gpu(i, vram=80.0) for i in range(4)]
        tp, pp = recommend_parallelism(gpus, "nvlink")
        assert tp == 4
        assert pp == 1

    def test_nvlink_8gpu_splits_tp_pp(self):
        gpus = [_make_gpu(i, vram=80.0) for i in range(8)]
        tp, pp = recommend_parallelism(gpus, "nvlink")
        assert tp * pp == 8

    def test_pcie_model_fits_single_gpu_tp1(self):
        """Fix 3: if model fits single GPU on PCIe, recommend TP=1 (single GPU faster)."""
        gpus = [_make_gpu(i, vram=24.0) for i in range(2)]
        # 7B model needs ~4.5GB → fits in 24GB single GPU
        tp, pp = recommend_parallelism(gpus, "pcie_x16", model_vram_needed_gb=4.5)
        assert tp == 1
        assert pp == 1

    def test_pcie_model_doesnt_fit_uses_tp(self):
        """Fix 3: model doesn't fit single GPU → tensor parallel (not pipeline parallel)."""
        gpus = [_make_gpu(i, vram=24.0) for i in range(2)]
        # 70B model needs ~40GB → doesn't fit in 24GB
        tp, pp = recommend_parallelism(gpus, "pcie_x16", model_vram_needed_gb=40.0)
        assert tp == 2   # tensor parallel
        assert pp == 1   # never pipeline parallel on PCIe

    def test_single_gpu_always_tp1(self):
        gpus = [_make_gpu(0, vram=24.0)]
        tp, pp = recommend_parallelism(gpus, "none")
        assert tp == 1 and pp == 1


# ═══════════════════════════════════════════════════════════════════════════
# RUNTIME RECOMMENDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestRuntimeRecommendation:

    def _make_scale(self, tier, gpu_count=1, total_vram=24.0, pooled_vram=24.0,
                    interconnect="none", tp=1, pp=1) -> ScaleProfile:
        from app.install.vram_calculator import calculate_vram_gb
        return ScaleProfile(
            tier=tier, gpu_count=gpu_count,
            total_vram_gb=total_vram, pooled_vram_gb=pooled_vram,
            interconnect=interconnect, recommended_runtime="ollama",
            recommended_tp=tp, recommended_pp=pp,
            max_model_size_b=pooled_vram / 0.563,
            can_run_7b=pooled_vram >= 4.5,
            can_run_13b=pooled_vram >= 8.0,
            can_run_70b=pooled_vram >= 40.0,
            can_run_405b=pooled_vram >= 230.0,
            reasoning="test",
        )

    def test_laptop_recommends_ollama(self):
        scale = self._make_scale("laptop", gpu_count=1, total_vram=8.0, pooled_vram=8.0)
        hw = _make_hw(gpu_vram_total_gb=8.0, effective_vram_gb=7.5)
        rec = recommend_runtime(scale, "llama3:8b", hw)
        assert rec.runtime == "ollama"

    def test_multi_gpu_nvlink_recommends_vllm_tp(self):
        scale = self._make_scale("server", gpu_count=4, total_vram=320.0, pooled_vram=294.0,
                                 interconnect="nvlink", tp=4)
        hw = _make_hw()
        rec = recommend_runtime(scale, "llama3:70b", hw)
        assert rec.runtime == "vllm"
        assert rec.tensor_parallel == 4

    def test_hyperscale_recommends_ray_vllm(self):
        scale = self._make_scale("hyperscale", gpu_count=8, total_vram=640.0,
                                 pooled_vram=589.0, interconnect="nvlink", tp=4, pp=2)
        hw = _make_hw()
        rec = recommend_runtime(scale, "llama3:405b", hw)
        assert rec.runtime == "ray_vllm"

    def test_apple_silicon_recommends_ollama(self):
        scale = self._make_scale("workstation", gpu_count=0)
        hw = _make_hw(is_apple_silicon=True, unified_memory_gb=48.0,
                      has_gpu=False, gpu_vendor="apple")
        rec = recommend_runtime(scale, "gemma2:9b", hw)
        assert rec.runtime == "ollama"

    def test_cpu_only_recommends_llamacpp(self):
        scale = self._make_scale("cpu_cluster", gpu_count=0, total_vram=0.0, pooled_vram=0.0)
        hw = _make_hw(has_gpu=False, gpu_vendor="none", effective_vram_gb=0.0,
                      gpu_count=0, total_vram_gb=0.0, is_apple_silicon=False)
        # Manually set all_gpus to empty so recommender sees no GPU
        hw.all_gpus = []
        rec = recommend_runtime(scale, "phi3:mini", hw)
        assert rec.runtime == "llamacpp"
        assert any("--threads" in a for a in rec.launch_args)

    def test_pcie_single_gpu_faster_than_multi_gpu(self):
        """The missing test from plan review — critical Fix 3 validation."""
        scale = self._make_scale("multi_gpu", gpu_count=2, total_vram=48.0,
                                 pooled_vram=37.4, interconnect="pcie_x16", tp=2)
        gpus = [_make_gpu(i, vram=24.0) for i in range(2)]
        hw = _make_hw(all_gpus=gpus, gpu_count=2, total_vram_gb=37.4,
                      topology_type="pcie_x16")
        # 7B model fits in 24GB single GPU → single GPU recommended
        rec = recommend_runtime(scale, "llama3:7b", hw, model_params_b=7.0)
        assert rec.tensor_parallel == 1
        assert "single" in rec.explanation.lower() or "faster" in rec.explanation.lower()

    def test_tp_degree_matches_gpu_count_nvlink(self):
        scale = self._make_scale("server", gpu_count=4, total_vram=320.0,
                                 pooled_vram=294.0, interconnect="nvlink", tp=4)
        hw = _make_hw()
        rec = recommend_runtime(scale, "llama3:70b", hw)
        assert rec.tensor_parallel == 4


# ═══════════════════════════════════════════════════════════════════════════
# DEPLOYMENT CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════

def _make_scale_and_rec(tier="server", tp=4, pp=1, runtime="vllm",
                        gpu_count=4, interconnect="nvlink"):
    from app.install.vram_calculator import calculate_vram_gb
    scale = ScaleProfile(
        tier=tier, gpu_count=gpu_count, total_vram_gb=320.0, pooled_vram_gb=294.0,
        interconnect=interconnect, recommended_runtime=runtime,
        recommended_tp=tp, recommended_pp=pp,
        max_model_size_b=500.0, can_run_7b=True, can_run_13b=True,
        can_run_70b=True, can_run_405b=False, reasoning="test",
    )
    rec = RuntimeRecommendation(
        runtime=runtime,
        launch_args=["serve", "meta-llama/Llama-3.1-70B",
                     "--tensor-parallel-size", str(tp),
                     "--gpu-memory-utilization", "0.90"],
        env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
        tensor_parallel=tp, pipeline_parallel=pp,
        gpu_device_ids=list(range(gpu_count)),
        explanation="test explanation",
    )
    return scale, rec


class TestDeploymentConfigs:

    def test_docker_compose_has_gpu_device_ids(self):
        scale, rec = _make_scale_and_rec()
        compose = generate_docker_compose(scale, rec, "llama3:70b")
        assert "device_ids" in compose
        # All 4 GPUs should be listed
        assert '"0"' in compose
        assert '"3"' in compose

    def test_vllm_launch_has_tp_flag(self):
        scale, rec = _make_scale_and_rec(tp=4)
        script = generate_vllm_launch_script(rec, "llama3:70b", scale)
        assert "tensor-parallel-size" in script
        assert "tok/s" not in script  # should not have speed info in script

    def test_k8s_yaml_is_valid_yaml(self):
        scale, rec = _make_scale_and_rec()
        k8s = generate_k8s_deployment(scale, rec, "llama3:70b")
        # Check for required YAML markers without PyYAML dependency
        assert "apiVersion:" in k8s
        assert "kind: Deployment" in k8s
        assert "kind: Service" in k8s
        assert "---" in k8s  # multi-document separator

    def test_k8s_yaml_has_gpu_resource(self):
        scale, rec = _make_scale_and_rec(gpu_count=4)
        k8s = generate_k8s_deployment(scale, rec, "llama3:70b")
        assert "nvidia.com/gpu" in k8s
        assert "4" in k8s

    def test_systemd_unit_has_restart_policy(self):
        _, rec = _make_scale_and_rec()
        unit = generate_systemd_unit(rec, "llama3:70b")
        assert "Restart=always" in unit or "Restart=" in unit

    def test_ollama_compose_has_pull_command(self):
        scale = ScaleProfile(
            tier="laptop", gpu_count=1, total_vram_gb=8.0, pooled_vram_gb=8.0,
            interconnect="none", recommended_runtime="ollama",
            recommended_tp=1, recommended_pp=1, max_model_size_b=10.0,
            can_run_7b=True, can_run_13b=False, can_run_70b=False,
            can_run_405b=False, reasoning="test",
        )
        rec = RuntimeRecommendation(runtime="ollama", gpu_device_ids=[0])
        compose = generate_docker_compose(scale, rec, "gemma2:9b")
        assert "ollama" in compose.lower()

    def test_write_configs_returns_paths(self, tmp_path):
        scale, rec = _make_scale_and_rec()
        paths = write_deployment_configs(scale, rec, "llama3:70b", output_dir=tmp_path)
        assert "docker_compose" in paths
        assert Path(paths["docker_compose"]).is_file()


# ═══════════════════════════════════════════════════════════════════════════
# MODEL ACQUISITION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestModelAcquisition:

    def _make_hw_laptop(self) -> HardwareProfile:
        return _make_hw(
            gpu_vram_total_gb=8.0, effective_vram_gb=7.5,
            scale_tier="laptop", gpu_count=1, total_vram_gb=7.5,
        )

    def test_acquisition_skips_if_already_served(self):
        """Step 1: model already in runtime → return immediately."""
        runtimes = [LLMRuntime(
            name="ollama", is_running=True,
            available_models=[ModelInfo(name="llama3:8b")],
        )]
        hw = self._make_hw_laptop()
        result = acquire_model("llama3:8b", hw, runtimes)
        assert result.success is True
        assert result.method == "already_served"

    def test_ollama_pull_attempted_when_model_missing(self):
        """Step 4: model not on disk → try ollama pull."""
        runtimes = []
        hw = self._make_hw_laptop()
        with (
            patch("app.install.model_acquirer._find_gguf_on_disk", return_value=None),
            patch("app.install.model_acquirer._find_hf_cache", return_value=None),
            patch("app.install.model_acquirer._ollama_available", return_value=True),
            patch("app.install.model_acquirer._pull_ollama", return_value=True) as mock_pull,
        ):
            result = acquire_model("gemma2:9b", hw, runtimes)
        mock_pull.assert_called_once_with("gemma2:9b")
        assert result.success is True
        assert result.method == "ollama_pulled"

    def test_gguf_on_disk_registered_not_downloaded(self):
        """Step 2: GGUF on disk → register, never download."""
        runtimes = [LLMRuntime(name="ollama", is_running=True, available_models=[])]
        hw = self._make_hw_laptop()
        fake_gguf = Path("/models/llama3-8b-Q4_K_M.gguf")
        with (
            patch("app.install.model_acquirer._find_gguf_on_disk", return_value=fake_gguf),
            patch("app.install.model_acquirer._ollama_available", return_value=True),
            patch("app.install.model_acquirer.register_gguf_with_ollama", return_value=True),
            patch("app.install.model_acquirer._download_huggingface") as mock_dl,
        ):
            result = acquire_model("llama3:8b", hw, runtimes)
        mock_dl.assert_not_called()  # Never downloaded when already on disk
        assert result.success is True
        assert result.method == "gguf_registered"

    def test_hf_cache_registered_not_downloaded(self):
        """Step 3: HF cache on disk → register, never download."""
        runtimes = []
        hw = self._make_hw_laptop()
        fake_hf_dir = Path("/home/user/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B")
        with (
            patch("app.install.model_acquirer._find_gguf_on_disk", return_value=None),
            patch("app.install.model_acquirer._find_hf_cache", return_value=fake_hf_dir),
            patch("app.install.model_acquirer.register_hf_cache", return_value={
                "success": True, "method": "gguf_registered_from_hf_cache",
                "path": "/some/model.gguf", "runtime": "ollama",
            }),
            patch("app.install.model_acquirer._download_huggingface") as mock_dl,
        ):
            result = acquire_model("meta-llama/Llama-3.1-8B", hw, runtimes)
        mock_dl.assert_not_called()
        assert result.success is True
        assert result.method == "hf_cache_registered"

    def test_hf_download_picks_correct_quant_for_vram(self):
        """Fix 1: 8GB VRAM → must pick Q4_K_M not Q8_0 for 8B model."""
        hw = _make_hw(
            effective_vram_gb=7.5, total_vram_gb=7.5, gpu_count=1,
        )
        quant = _pick_best_quant(hw, 8.0)
        # Q8_0 for 8B = 8.0 × 1.0 + overhead ≈ 9.5GB → doesn't fit 7.5GB
        # Q4_K_M for 8B = 8.0 × 0.563 + overhead ≈ 5.5GB → fits
        assert quant in ("Q4_K_M", "Q5_K_M", "Q6_K"), f"Expected Q4-Q6 range, got {quant}"
        assert quant != "Q8_0"

    def test_hf_download_picks_q8_for_high_vram(self):
        """Fix 1: 24GB VRAM → Q8_0 fits for 8B model."""
        hw = _make_hw(
            effective_vram_gb=21.5, total_vram_gb=21.5, gpu_count=1,
        )
        quant = _pick_best_quant(hw, 8.0)
        # Q8_0 for 8B = ~9.5GB → fits in 21.5GB
        assert quant == "Q8_0"


# ═══════════════════════════════════════════════════════════════════════════
# MODEL REGISTRATION TESTS (Fix 2)
# ═══════════════════════════════════════════════════════════════════════════

class TestModelRegistration:

    def test_find_best_gguf_prefers_q4km(self, tmp_path):
        """Best GGUF picker should prefer Q4_K_M over Q8_0."""
        (tmp_path / "model-Q8_0.gguf").write_bytes(b"fake")
        (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"fake")
        (tmp_path / "model-Q6_K.gguf").write_bytes(b"fake")
        best = find_best_gguf_in_path(tmp_path)
        assert best is not None
        assert "Q4_K_M" in best.name

    def test_find_best_gguf_single_file(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"fake")
        best = find_best_gguf_in_path(tmp_path)
        assert best == f

    def test_find_best_gguf_no_files_returns_none(self, tmp_path):
        assert find_best_gguf_in_path(tmp_path) is None

    def test_register_gguf_creates_ollama_modelfile(self, tmp_path):
        """ollama create should be called with a temp Modelfile."""
        fake_gguf = tmp_path / "llama3-Q4_K_M.gguf"
        fake_gguf.write_bytes(b"fake")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = register_gguf_with_ollama(fake_gguf, "llama3-local")
        assert result is True
        # Verify ollama create was called
        args = mock_run.call_args[0][0]
        assert args[0] == "ollama"
        assert args[1] == "create"
        assert args[2] == "llama3-local"
        assert "-f" in args

    def test_register_gguf_cleans_up_modelfile(self, tmp_path):
        """Temp Modelfile must be deleted after ollama create."""
        fake_gguf = tmp_path / "model.gguf"
        fake_gguf.write_bytes(b"fake")
        captured_modelfile = []

        def capture_args(cmd, **kwargs):
            # Find the -f argument and capture Modelfile path
            try:
                idx = cmd.index("-f")
                captured_modelfile.append(cmd[idx + 1])
            except (ValueError, IndexError):
                pass
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=capture_args):
            register_gguf_with_ollama(fake_gguf, "test-model")

        if captured_modelfile:
            assert not Path(captured_modelfile[0]).exists(), "Modelfile was not cleaned up"

    def test_register_hf_cache_detects_gguf_vs_safetensors(self, tmp_path):
        """Fix 2: GGUF in HF cache → register directly, no Ollama FROM safetensors."""
        # Create a fake HF cache with a GGUF inside
        gguf_file = tmp_path / "snapshots" / "abc123" / "model-Q4_K_M.gguf"
        gguf_file.parent.mkdir(parents=True)
        gguf_file.write_bytes(b"fake gguf")

        with patch("app.install.model_registrar.register_gguf_with_ollama", return_value=True):
            result = register_hf_cache(tmp_path, "meta-llama/llama-3.1-8b", ["ollama"])

        assert result["success"] is True
        assert "gguf" in result["method"].lower()

    def test_safetensors_does_not_try_ollama_from(self, tmp_path):
        """Fix 2: safetensors in cache → must NOT try FROM <path> in Modelfile."""
        # Create fake safetensors (no GGUF)
        (tmp_path / "model.safetensors").write_bytes(b"fake safetensors")

        result = register_hf_cache(tmp_path, "meta-llama/llama-3.1-8b", ["ollama"])

        # Should NOT succeed directly (safetensors can't be loaded this way)
        assert result["success"] is False
        # Should provide a useful suggestion
        assert "suggestion" in result or "reason" in result
        # Specifically: reason should mention safetensors
        reason = result.get("reason", "")
        assert "safetensors" in reason

    def test_repo_id_injection_blocked(self):
        """Security: invalid repo_id must be rejected before subprocess."""
        from app.install.model_acquirer import _download_huggingface, _HF_REPO_PATTERN

        malicious = "org/repo; rm -rf /"
        assert not _HF_REPO_PATTERN.match(malicious)

        with patch("subprocess.run") as mock_run:
            result = _download_huggingface(malicious, 8.0, _make_hw())
        mock_run.assert_not_called()
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# HF → OLLAMA NAME MAPPING
# ═══════════════════════════════════════════════════════════════════════════

class TestHFOllamaMapping:

    def test_llama31_maps_to_ollama(self):
        name = map_hf_to_ollama_name("meta-llama/Llama-3.1-8B-Instruct")
        assert name == "llama3.1:8b"

    def test_gemma2_maps_to_ollama(self):
        name = map_hf_to_ollama_name("google/gemma-2-9b-it")
        assert name == "gemma2:9b"

    def test_unknown_model_returns_none(self):
        name = map_hf_to_ollama_name("some-org/completely-unknown-model")
        assert name is None

    def test_find_gguf_repo_returns_candidate(self):
        repo = find_gguf_repo("meta-llama/Llama-3.1-8B-Instruct")
        assert repo is not None
        assert "/" in repo   # should be org/repo format
        assert "GGUF" in repo.upper() or "gguf" in repo


# ═══════════════════════════════════════════════════════════════════════════
# SCALE PROFILE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

class TestScaleProfileIntegration:

    def test_4xa100_can_run_70b(self):
        """4×A100 80GB should be able to run 70B models."""
        gpus = [_make_gpu(i, name="A100", vram=80.0, bw=2039.0) for i in range(4)]
        topology = _make_topology(gpus, "nvlink", tp=4)
        hw = _make_hw(
            all_gpus=gpus, gpu_count=4, gpu_name="A100",
            gpu_vram_total_gb=80.0, effective_vram_gb=78.0,
            total_vram_gb=topology.pooled_vram_gb,
            topology_type="nvlink", scale_tier="server",
        )
        profile = classify(topology, hw)
        assert profile.can_run_70b is True
        assert profile.recommended_runtime == "vllm"

    def test_laptop_rtx3060_cannot_run_70b(self):
        """RTX 3060 12GB laptop cannot run 70B at Q4_K_M."""
        gpus = [_make_gpu(0, name="RTX 3060", vram=12.0, bw=360.0)]
        topology = _make_topology(gpus, "none")
        hw = _make_hw(
            all_gpus=gpus, gpu_count=1, gpu_name="RTX 3060",
            gpu_vram_total_gb=12.0, effective_vram_gb=11.0,
            total_vram_gb=11.0, scale_tier="laptop",
        )
        profile = classify(topology, hw)
        assert profile.can_run_70b is False
        assert profile.can_run_7b is True

    def test_cpu_only_cluster(self):
        topology = GPUTopology(
            gpus=[], interconnect="none", total_vram_gb=0.0,
            pooled_vram_gb=0.0, recommended_tp_degree=1,
            recommended_pp_degree=1, scale_tier="cpu_cluster",
        )
        hw = _make_hw(
            has_gpu=False, gpu_vendor="none", effective_vram_gb=0.0,
            gpu_count=0, total_vram_gb=0.0, scale_tier="cpu_cluster",
            ram_total_gb=512.0, cpu_cores_logical=64,
        )
        profile = classify(topology, hw)
        assert profile.tier == "cpu_cluster"
        assert profile.recommended_runtime == "llamacpp"
