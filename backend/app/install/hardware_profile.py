"""Hardware detection for bootstrap layer — measurements only."""
from __future__ import annotations

import platform
import subprocess

import psutil

from app.install.types import HardwareProfile


def _detect_nvidia() -> tuple[str | None, float, float, bool]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 3:
                name = parts[0]
                total_gb = float(parts[1]) / 1024
                free_gb = float(parts[2]) / 1024
                return name, total_gb, free_gb, True
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None, 0.0, 0.0, False


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in (
        "arm64",
        "aarch64",
    )


def detect_hardware_profile() -> HardwareProfile:
    os_name = platform.system()
    os_version = platform.version()
    architecture = platform.machine()
    cpu_name = platform.processor() or "Unknown"
    cpu_physical = psutil.cpu_count(logical=False) or 1
    cpu_logical = psutil.cpu_count(logical=True) or 1

    mem = psutil.virtual_memory()
    ram_total_gb = round(mem.total / (1024**3), 2)
    ram_available_gb = round(mem.available / (1024**3), 2)

    disk_root = "C:\\" if os_name == "Windows" else "/"
    disk = psutil.disk_usage(disk_root)
    disk_total_gb = round(disk.total / (1024**3), 2)
    disk_free_gb = round(disk.free / (1024**3), 2)

    apple = _is_apple_silicon()
    gpu_name, vram_total, vram_free, has_cuda = _detect_nvidia()
    has_gpu = gpu_name is not None

    unified: float | None = None
    if apple:
        unified = ram_total_gb

    return HardwareProfile(
        os_name=os_name,
        os_version=os_version,
        architecture=architecture,
        cpu_name=cpu_name,
        cpu_cores_physical=cpu_physical,
        cpu_cores_logical=cpu_logical,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        disk_free_gb=disk_free_gb,
        disk_total_gb=disk_total_gb,
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        gpu_vram_total_gb=round(vram_total, 2),
        gpu_vram_free_gb=round(vram_free, 2),
        has_cuda=has_cuda,
        is_apple_silicon=apple,
        unified_memory_gb=unified,
    )
