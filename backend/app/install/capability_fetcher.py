"""
Live capability matrix + optional Ollama catalog enrichment.
Fetches ShipAI intelligence JSON (GitHub or env URL), caches offline, falls back to bundled.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUNDLED_MATRIX = _REPO_ROOT / "intelligence" / "model_capabilities.json"
_CACHE_DIR = Path.home() / ".shipai" / "cache"
_CACHE_FILE = _CACHE_DIR / "model_capabilities.json"
_CATALOG_CACHE = _CACHE_DIR / "ollama_catalog.json"

DEFAULT_MATRIX_URL = os.getenv(
    "SHIPAI_CAPABILITY_MATRIX_URL",
    "https://raw.githubusercontent.com/shipai/shipai/main/intelligence/model_capabilities.json",
)

OLLAMA_LIBRARY_URL = os.getenv(
    "SHIPAI_OLLAMA_LIBRARY_URL",
    "https://ollama.com/library.json",
)

CACHE_MAX_AGE_SECONDS = int(os.getenv("SHIPAI_MATRIX_CACHE_TTL", str(7 * 24 * 3600)))


class CapabilityMatrixError(Exception):
    pass


def bundled_matrix_path() -> Path:
    if _BUNDLED_MATRIX.is_file():
        return _BUNDLED_MATRIX
    alt = Path(__file__).resolve().parents[4] / "intelligence" / "model_capabilities.json"
    return alt if alt.is_file() else _BUNDLED_MATRIX


def cache_path() -> Path:
    return _CACHE_FILE


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate_matrix(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise CapabilityMatrixError("Matrix must be a JSON object")
    if "models" not in data or not isinstance(data["models"], dict):
        raise CapabilityMatrixError("Matrix missing 'models' object")
    if "shipai_node_requirements" not in data:
        raise CapabilityMatrixError("Matrix missing 'shipai_node_requirements'")
    if not isinstance(data["shipai_node_requirements"], dict):
        raise CapabilityMatrixError("'shipai_node_requirements' must be an object")
    return data


def load_bundled_matrix() -> dict[str, Any]:
    path = bundled_matrix_path()
    if not path.is_file():
        raise CapabilityMatrixError(f"Bundled matrix not found at {path}")
    return validate_matrix(_load_json_file(path))


def load_cached_matrix(max_age: int = CACHE_MAX_AGE_SECONDS) -> dict[str, Any] | None:
    if not _CACHE_FILE.is_file():
        return None
    age = time.time() - _CACHE_FILE.stat().st_mtime
    if age > max_age:
        return None
    try:
        return validate_matrix(_load_json_file(_CACHE_FILE))
    except (json.JSONDecodeError, CapabilityMatrixError):
        return None


def save_cached_matrix(matrix: dict[str, Any]) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    validated = validate_matrix(matrix)
    with _CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(validated, f, indent=2)
    return _CACHE_FILE


async def fetch_remote_matrix(
    url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    url = url or DEFAULT_MATRIX_URL
    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=10.0, follow_redirects=True)
    assert client is not None
    try:
        r = await client.get(url)
        r.raise_for_status()
        data = validate_matrix(r.json())
        save_cached_matrix(data)
        return data
    finally:
        if owns:
            await client.aclose()


async def get_capability_matrix(
    force_refresh: bool = False,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """
    Resolution order:
    1. Remote fetch (if force_refresh or no valid cache)
    2. Fresh cache on disk
    3. Bundled repo matrix
    """
    if not force_refresh:
        cached = load_cached_matrix()
        if cached is not None:
            return cached

    try:
        return await fetch_remote_matrix(client=client)
    except Exception:
        pass

    cached_stale = None
    if _CACHE_FILE.is_file():
        try:
            cached_stale = validate_matrix(_load_json_file(_CACHE_FILE))
        except (json.JSONDecodeError, CapabilityMatrixError):
            pass
    if cached_stale is not None:
        return cached_stale

    return load_bundled_matrix()


def get_model_entry(matrix: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    models = matrix.get("models", {})
    if model_id in models:
        return models[model_id]
    base = model_id.split(":")[0]
    if base in models:
        return models[base]
    for key, entry in models.items():
        if entry.get("ollama_id") == model_id or entry.get("ollama_id") == base:
            return entry
    return None


def get_node_requirements(matrix: dict[str, Any], node: str) -> dict[str, Any]:
    return matrix.get("shipai_node_requirements", {}).get(node, {})


def list_ollama_ids(matrix: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for _key, entry in matrix.get("models", {}).items():
        oid = entry.get("ollama_id")
        if oid:
            ids.append(oid)
    return ids


async def fetch_ollama_catalog(
    client: httpx.AsyncClient | None = None,
    matrix_fallback: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
  Try live Ollama library JSON; on failure return matrix-derived catalog entries.
  Each entry: {name, pulls?, size_gb?} — structure depends on upstream API.
    """
    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=8.0, follow_redirects=True)
    assert client is not None

    catalog: list[dict[str, Any]] = []
    try:
        r = await client.get(OLLAMA_LIBRARY_URL)
        if r.status_code == 200:
            payload = r.json()
            if isinstance(payload, list):
                catalog = payload
            elif isinstance(payload, dict):
                catalog = payload.get("models", payload.get("library", []))
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with _CATALOG_CACHE.open("w", encoding="utf-8") as f:
                json.dump({"fetched_at": time.time(), "models": catalog}, f)
    except Exception:
        pass
    finally:
        if owns:
            await client.aclose()

    if catalog:
        return catalog

    matrix = matrix_fallback or load_bundled_matrix()
    return [
        {
            "name": entry.get("ollama_id", key),
            "size_gb": entry.get("size_gb"),
            "source": "capability_matrix",
        }
        for key, entry in matrix.get("models", {}).items()
        if entry.get("ollama_id") and not entry.get("capabilities", {}).get("embeddings")
    ]


def rank_catalog_for_hardware(
    catalog: list[dict[str, Any]],
    matrix: dict[str, Any],
    effective_vram_gb: float,
    effective_ram_gb: float,
    has_cuda: bool,
    top_n: int = 10,
    ram_total_gb: float | None = None,
) -> list[dict[str, Any]]:
    """
    Filter catalog by matrix min_vram/min_ram; return top_n feasible (matrix order).
    No hardcoded model names in Python — uses matrix specs only.
    """
    # Instantaneous free RAM can be near zero on Windows; use total RAM as ceiling.
    budget_ram = effective_ram_gb
    if ram_total_gb is not None:
        budget_ram = max(effective_ram_gb, ram_total_gb * 0.85)

    scored: list[tuple[float, dict[str, Any]]] = []

    for item in catalog:
        name = item.get("name") or item.get("model") or ""
        if not name:
            continue
        entry = get_model_entry(matrix, name)
        if not entry:
            continue
        if entry.get("capabilities", {}).get("embeddings"):
            continue
        min_vram = float(entry.get("min_vram_gb", 0))
        min_ram = float(entry.get("min_ram_gb", 0))
        if min_vram > effective_vram_gb or min_ram > budget_ram:
            continue
        if not entry.get("fits_cpu", False) and not has_cuda and min_vram > 0:
            continue
        json_score = float(entry.get("capabilities", {}).get("json_reliable", 0))
        scored.append((json_score, {**item, "matrix_key": name, "capabilities": entry}))

    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:top_n]]
