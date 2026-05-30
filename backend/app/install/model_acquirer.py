"""
Model acquirer: single entry point for getting a model onto ANY machine.

Acquisition priority chain (correct order — never download what's already there):
  1. Already served by a running runtime?  → use it
  2. GGUF file on disk?                    → register with runtime
  3. HF cache on disk?                     → register (GGUF) or pull equivalent
  4. In Ollama library?                    → ollama pull
  5. Not anywhere?                         → huggingface-cli download + register

Works on ALL scales: laptop with one GGUF → 8×A100 server with shared HF cache.
HuggingFace support is NOT enterprise-only — a laptop user's manually downloaded
GGUF gets the same auto-registration treatment.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.install.inventory_scanner import _hf_cache_dirs, _gguf_search_roots
from app.install.model_registrar import (
    find_best_gguf_in_path,
    register_gguf_with_ollama,
    register_hf_cache,
    _ollama_available,
)
from app.install.types import DiscoveredModel, HardwareProfile, LLMRuntime
from app.install.vram_calculator import calculate_vram_gb

logger = logging.getLogger("shipai.acquirer")

# Validated repo ID pattern — Fix from plan review: blocks RCE via injection
_HF_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+$")

# Default HF download target dir
_HF_DOWNLOAD_DIR = Path.home() / ".shipai" / "models"


@dataclass
class AcquisitionResult:
    """Result of attempting to acquire a model."""
    success: bool
    method: str         # "already_served" | "gguf_registered" | "hf_cache_registered" |
                        # "ollama_pulled" | "hf_downloaded" | "failed"
    model_id: str = ""
    path: Optional[str] = None
    runtime: str = ""
    endpoint: str = ""
    reason: str = ""
    suggestion: str = ""
    warnings: List[str] = field(default_factory=list)


# ── Fix 1: Model-aware quant selection ──────────────────────────────────────

def _pick_best_quant(hw: HardwareProfile, model_params_b: float) -> str:
    """
    Pick highest quality quantization that actually FITS in available VRAM.

    Fix 1 from plan review: quant selection must be model-param-aware.
    Don't just check VRAM threshold — check if the specific model fits at that quant.
    """
    eff_vram = getattr(hw, "effective_vram_gb", 0) or 0
    eff_ram = getattr(hw, "effective_ram_gb", hw.ram_available_gb)

    # Use pooled VRAM if multi-GPU
    total_vram = getattr(hw, "total_vram_gb", 0) or eff_vram
    budget = max(total_vram, eff_vram)

    for quant in ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]:
        vram_needed, _ = calculate_vram_gb(model_params_b, quant)
        if vram_needed <= budget:
            return quant  # highest quality that fits

    # CPU-only fallback: check RAM
    for quant in ["Q4_K_M", "Q3_K_M", "Q2_K"]:
        vram_needed, _ = calculate_vram_gb(model_params_b, quant)
        if vram_needed <= eff_ram * 0.5:
            return quant

    return "Q2_K"  # last resort


# ── Source checkers ──────────────────────────────────────────────────────────

def _is_already_served(model_id: str, runtimes: List[LLMRuntime]) -> Optional[str]:
    """Check if model is already served by a running runtime. Returns runtime name or None."""
    for rt in runtimes:
        if not rt.is_running:
            continue
        for m in rt.available_models:
            if model_id.lower() in m.name.lower() or m.name.lower() in model_id.lower():
                return rt.name
    return None


def _find_gguf_on_disk(model_id: str) -> Optional[Path]:
    """Search GGUF search roots for a matching GGUF file."""
    model_lower = model_id.lower().replace(":", "-").replace("/", "-")
    for root in _gguf_search_roots():
        if not root.is_dir():
            continue
        for gguf in root.rglob("*.gguf"):
            if model_lower in gguf.stem.lower() or gguf.stem.lower() in model_lower:
                return gguf
    return None


def _find_hf_cache(model_id: str) -> Optional[Path]:
    """Search HF cache dirs for a matching model directory."""
    search = model_id.lower().replace("/", "--").replace("-", "--")
    for hub in _hf_cache_dirs():
        if not hub.is_dir():
            continue
        for entry in hub.iterdir():
            if not entry.is_dir() or not entry.name.startswith("models--"):
                continue
            folder_name = entry.name.lower()
            if search in folder_name or folder_name in search:
                return entry
    return None


# ── Download backends ────────────────────────────────────────────────────────

def _pull_ollama(model_id: str, timeout: int = 600) -> bool:
    """
    Pull a model via ollama pull with progress streaming.
    Returns True on success.
    """
    if not _ollama_available():
        return False
    try:
        result = subprocess.run(
            ["ollama", "pull", model_id],
            timeout=timeout,
            # Don't capture — let progress stream to terminal
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("ollama pull failed for %s: %s", model_id, e)
        return False


def _hf_cli_available() -> bool:
    """Check if huggingface-cli is available."""
    try:
        r = subprocess.run(
            ["huggingface-cli", "--version"],
            capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _download_huggingface(
    repo_id: str,
    model_params_b: float,
    hw: HardwareProfile,
    target_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Download a GGUF from HuggingFace using huggingface-cli or direct httpx stream.

    Security: repo_id validated against strict pattern before use.
    Never uses shell=True.

    Returns path to downloaded GGUF file, or None on failure.
    """
    # Security: validate repo_id (Fix from plan review)
    if not _HF_REPO_PATTERN.match(repo_id):
        logger.error("Invalid HF repo_id rejected: %r", repo_id)
        return None

    quant = _pick_best_quant(hw, model_params_b)
    out_dir = target_dir or _HF_DOWNLOAD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Method 1: huggingface-cli download (preferred)
    if _hf_cli_available():
        try:
            result = subprocess.run(
                [
                    "huggingface-cli", "download",
                    repo_id,
                    "--include", f"*{quant}*.gguf",
                    "--local-dir", str(out_dir),
                    "--local-dir-use-symlinks", "False",
                ],
                timeout=1800,  # 30 min timeout for large models
            )
            if result.returncode == 0:
                # Find the downloaded file
                downloaded = list(out_dir.rglob(f"*{quant}*.gguf"))
                if downloaded:
                    logger.info("Downloaded via hf-cli: %s", downloaded[0])
                    return downloaded[0]
                # Fallback: any GGUF
                any_gguf = list(out_dir.rglob("*.gguf"))
                if any_gguf:
                    return any_gguf[0]
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug("huggingface-cli download failed: %s", e)

    # Method 2: Direct httpx streaming download
    return _download_hf_direct(repo_id, quant, out_dir)


def _download_hf_direct(
    repo_id: str,
    quant: str,
    out_dir: Path,
) -> Optional[Path]:
    """
    Direct HuggingFace CDN download via httpx.
    Fetches file list from HF API, picks best GGUF, streams to disk.
    """
    try:
        import httpx
    except ImportError:
        logger.debug("httpx not available for direct HF download")
        return None

    api_url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            r = client.get(api_url)
            if r.status_code != 200:
                return None
            data = r.json()
            siblings = data.get("siblings", [])
            # Find best GGUF file
            gguf_files = [
                s["rfilename"] for s in siblings
                if isinstance(s, dict) and s.get("rfilename", "").endswith(".gguf")
            ]
            if not gguf_files:
                return None
            # Pick preferred quant
            target_file = None
            for q in [quant, "Q4_K_M", "Q5_K_M", "Q8_0"]:
                for f in gguf_files:
                    if q.upper() in f.upper():
                        target_file = f
                        break
                if target_file:
                    break
            if not target_file:
                target_file = gguf_files[0]

            # Stream download
            download_url = f"https://huggingface.co/{repo_id}/resolve/main/{target_file}"
            out_path = out_dir / Path(target_file).name
            with client.stream("GET", download_url) as resp:
                if resp.status_code != 200:
                    return None
                with out_path.open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
            logger.info("Downloaded via httpx: %s", out_path)
            return out_path
    except Exception as e:
        logger.debug("Direct HF download failed: %s", e)
        return None


# ── Main acquisition function ────────────────────────────────────────────────

def acquire_model(
    model_id: str,
    hw: HardwareProfile,
    runtimes: List[LLMRuntime],
    inventory: Optional[List[DiscoveredModel]] = None,
    model_params_b: float = 0.0,
) -> AcquisitionResult:
    """
    Full acquisition chain — works on laptop, workstation, server, everywhere.

    Priority:
    1. Already served?  → done
    2. GGUF on disk?    → register (laptop: ollama create; server: llama-server)
    3. HF cache?        → register or fallback chain (Fix 2)
    4. Ollama pull?     → pull from Ollama library
    5. HF download?     → download GGUF + register
    """
    warnings: list[str] = []
    available_runtimes = [rt.name for rt in runtimes if rt.is_running]

    # ── Step 1: Already served ───────────────────────────────────────────────
    serving_rt = _is_already_served(model_id, runtimes)
    if serving_rt:
        rt_obj = next((r for r in runtimes if r.name == serving_rt), None)
        endpoint = rt_obj.base_url or "" if rt_obj else ""
        return AcquisitionResult(
            success=True, method="already_served",
            model_id=model_id, runtime=serving_rt, endpoint=endpoint,
        )

    # ── Step 2: GGUF on disk ────────────────────────────────────────────────
    gguf_path = _find_gguf_on_disk(model_id)
    if gguf_path:
        if "ollama" in available_runtimes and _ollama_available():
            ok = register_gguf_with_ollama(gguf_path, model_id.replace(":", "-"))
            if ok:
                return AcquisitionResult(
                    success=True, method="gguf_registered",
                    model_id=model_id, path=str(gguf_path), runtime="ollama",
                    endpoint="http://localhost:11434",
                )
        # Even without Ollama, we found the file — report it
        return AcquisitionResult(
            success=True, method="gguf_found",
            model_id=model_id, path=str(gguf_path),
            suggestion=f"Run: ollama create {model_id.replace(':', '-')} -f Modelfile",
        )

    # ── Step 3: HF cache on disk ────────────────────────────────────────────
    hf_dir = _find_hf_cache(model_id)
    if hf_dir:
        reg_result = register_hf_cache(hf_dir, model_id, available_runtimes)
        if reg_result.get("success"):
            return AcquisitionResult(
                success=True, method="hf_cache_registered",
                model_id=model_id, path=reg_result.get("path"),
                runtime=reg_result.get("runtime", "ollama"),
                endpoint="http://localhost:11434",
            )
        # Safetensors → follow suggestion
        if reg_result.get("suggestion_ollama_pull"):
            ollama_name = reg_result["suggestion_ollama_pull"]
            warnings.append(f"Safetensors cache found but not directly loadable → trying ollama pull {ollama_name}")
            if _pull_ollama(ollama_name):
                return AcquisitionResult(
                    success=True, method="hf_cache_ollama_pull",
                    model_id=model_id, runtime="ollama",
                    endpoint="http://localhost:11434",
                    warnings=warnings,
                )
        if reg_result.get("suggestion_hf_repo"):
            gguf_repo = reg_result["suggestion_hf_repo"]
            warnings.append(f"Safetensors found → downloading GGUF from {gguf_repo}")
            # Fall through to step 5 with the GGUF repo
            downloaded = _download_huggingface(gguf_repo, model_params_b, hw)
            if downloaded:
                register_gguf_with_ollama(downloaded, model_id.replace(":", "-"))
                return AcquisitionResult(
                    success=True, method="hf_gguf_downloaded_from_safetensors_fallback",
                    model_id=model_id, path=str(downloaded), runtime="ollama",
                    warnings=warnings,
                )

    # ── Step 4: Ollama pull ─────────────────────────────────────────────────
    if _ollama_available():
        logger.info("Attempting ollama pull: %s", model_id)
        if _pull_ollama(model_id):
            return AcquisitionResult(
                success=True, method="ollama_pulled",
                model_id=model_id, runtime="ollama",
                endpoint="http://localhost:11434",
                warnings=warnings,
            )

    # ── Step 5: HuggingFace download (laptop OR server) ─────────────────────
    # Determine HF repo to download from
    from app.install.runtime_recommender import map_hf_to_ollama_name, find_gguf_repo
    hf_repo = None
    # Try to find a GGUF community repo
    hf_repo = find_gguf_repo(model_id)
    if hf_repo and _HF_REPO_PATTERN.match(hf_repo):
        logger.info("Attempting HF download: %s", hf_repo)
        downloaded = _download_huggingface(hf_repo, model_params_b, hw)
        if downloaded:
            # Register with Ollama if available
            registered = False
            if _ollama_available():
                registered = register_gguf_with_ollama(
                    downloaded, model_id.replace(":", "-")
                )
            return AcquisitionResult(
                success=True,
                method="hf_downloaded",
                model_id=model_id,
                path=str(downloaded),
                runtime="ollama" if registered else "llamacpp",
                endpoint="http://localhost:11434" if registered else "http://localhost:8080",
                warnings=warnings,
            )

    # ── All sources failed ───────────────────────────────────────────────────
    return AcquisitionResult(
        success=False,
        method="failed",
        model_id=model_id,
        reason="all_acquisition_methods_exhausted",
        suggestion=(
            f"Try manually: ollama pull {model_id}  OR  "
            f"huggingface-cli download bartowski/{model_id.replace('/', '-')}-GGUF "
            f"--include '*.Q4_K_M.gguf'"
        ),
        warnings=warnings,
    )
