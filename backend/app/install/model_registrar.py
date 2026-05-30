"""
Model registrar: takes a model already on disk → makes it accessible via a runtime.

Handles:
  GGUF file → ollama create (via Modelfile) OR llama-server direct launch
  HF cache  → detect GGUF inside → register; safetensors → fallback chain (Fix 2)
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("shipai.registrar")

# Quant preference order for picking best GGUF in a directory
_QUANT_PREFERENCE = [
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S",
    "Q6_K", "Q8_0", "Q3_K_M", "Q3_K_L", "Q2_K",
    "IQ4_XS", "IQ3_M", "F16",
]


def find_best_gguf_in_path(directory: Path) -> Optional[Path]:
    """
    Walk a directory and find the best GGUF file by quant preference.
    Returns None if no GGUF found.
    """
    if not directory.is_dir():
        if directory.is_file() and directory.suffix.lower() == ".gguf":
            return directory
        return None

    gguf_files = list(directory.rglob("*.gguf"))
    if not gguf_files:
        return None

    if len(gguf_files) == 1:
        return gguf_files[0]

    # Score by quant preference
    def quant_score(p: Path) -> int:
        name_upper = p.name.upper()
        for i, q in enumerate(_QUANT_PREFERENCE):
            if q in name_upper or q.replace("_", "-") in name_upper:
                return i
        return len(_QUANT_PREFERENCE)  # unknown → lowest priority

    return min(gguf_files, key=quant_score)


def _has_safetensors(directory: Path) -> bool:
    """Check if a directory contains safetensor model files."""
    return any(directory.rglob("*.safetensors"))


def register_gguf_with_ollama(
    gguf_path: Path,
    model_name: str,
) -> bool:
    """
    Register a GGUF file with Ollama using a temporary Modelfile.

    Creates: FROM /absolute/path/to/model.gguf
    Runs: ollama create <model_name> -f <modelfile>
    Cleans up the temp Modelfile afterwards.

    Returns True on success.
    """
    abs_path = str(gguf_path.resolve())
    modelfile_content = f"FROM {abs_path}\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".Modelfile", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(modelfile_content)
        modelfile_path = tf.name

    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            logger.info("Registered GGUF with Ollama: %s → %s", abs_path, model_name)
            return True
        else:
            logger.warning(
                "ollama create failed for %s: %s", model_name, result.stderr[:500]
            )
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug("ollama create failed: %s", e)
        return False
    finally:
        try:
            os.unlink(modelfile_path)
        except OSError:
            pass


def register_gguf_with_llamacpp(
    gguf_path: Path,
    port: int = 8080,
    ctx_size: int = 4096,
    threads: int = 0,
) -> Optional[subprocess.Popen]:
    """
    Launch llama-server pointing at a GGUF file.
    Returns the Popen process handle for lifecycle management, or None on failure.
    """
    import psutil
    auto_threads = threads or max(1, psutil.cpu_count(logical=False) or 4)
    cmd = [
        "llama-server",
        "--model", str(gguf_path.resolve()),
        "--port", str(port),
        "--ctx-size", str(ctx_size),
        "--threads", str(auto_threads),
        "--host", "0.0.0.0",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info("Launched llama-server PID %d for %s", proc.pid, gguf_path.name)
        return proc
    except FileNotFoundError:
        logger.debug("llama-server not found in PATH")
        return None


def register_hf_cache(
    hf_cache_path: Path,
    model_name: str,
    runtimes_available: list[str] | None = None,
) -> dict:
    """
    Register a HuggingFace cache directory with a runtime.

    Fix 2 from plan review — safetensors → Ollama FROM is WRONG.
    Correct fallback chain:
    1. GGUF in cache?             → register_gguf_with_ollama
    2. Safetensors?               → map_hf_to_ollama_name → ollama pull
    3. Can't map?                 → find_gguf_repo → download GGUF
    4. Everything fails?          → honest error with suggestion

    Returns dict with: success, method, path, suggestion
    """
    runtimes = runtimes_available or []

    # Step 1: Look for GGUF inside cache
    gguf = find_best_gguf_in_path(hf_cache_path)
    if gguf:
        if "ollama" in runtimes or _ollama_available():
            success = register_gguf_with_ollama(gguf, model_name)
            if success:
                return {
                    "success": True,
                    "method": "gguf_registered_from_hf_cache",
                    "path": str(gguf),
                    "runtime": "ollama",
                }
        return {
            "success": True,
            "method": "gguf_found_in_cache",
            "path": str(gguf),
            "suggestion": f"Run: ollama create {model_name} -f Modelfile (FROM {gguf})",
        }

    # Step 2: Safetensors detected — don't try Ollama FROM (it won't work)
    if _has_safetensors(hf_cache_path):
        from app.install.runtime_recommender import map_hf_to_ollama_name, find_gguf_repo
        ollama_name = map_hf_to_ollama_name(model_name)
        if ollama_name and _ollama_available():
            logger.info(
                "Safetensors detected — pulling Ollama equivalent: %s", ollama_name
            )
            return {
                "success": False,  # caller should do the pull
                "method": "safetensors_map_to_ollama",
                "suggestion_ollama_pull": ollama_name,
                "reason": "safetensors_not_directly_loadable_use_ollama_pull",
            }

        gguf_repo = find_gguf_repo(model_name)
        if gguf_repo:
            return {
                "success": False,
                "method": "safetensors_suggest_gguf_download",
                "suggestion_hf_repo": gguf_repo,
                "reason": "safetensors_not_directly_loadable_download_gguf",
            }

    # Step 4: Nothing worked
    return {
        "success": False,
        "method": "no_loadable_format",
        "reason": "safetensors_not_directly_loadable",
        "suggestion": f"Run: ollama pull {model_name}  OR  huggingface-cli download <repo> --include '*.Q4_K_M.gguf'",
    }


def _ollama_available() -> bool:
    """Quick check: is ollama in PATH?"""
    try:
        r = subprocess.run(
            ["ollama", "--version"],
            capture_output=True, timeout=3,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
