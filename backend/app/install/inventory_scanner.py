"""Phase B — deep inventory: disk artifacts, cross-linking, readiness flags."""
from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from app.install.canonical import names_likely_same, normalize_canonical_id
from app.install.types import DiscoveredModel, LLMRuntime, ModelInfo

_QUANT_PATTERN = re.compile(
    r"(Q\d[_A-Z0-9]*|F16|F32|BF16|fp16|fp32|q4|q8)",
    re.IGNORECASE,
)


def _hf_cache_dirs() -> List[Path]:
    dirs: List[Path] = []
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        dirs.append(Path(hf_home) / "hub")
    xdg = os.getenv("XDG_CACHE_HOME")
    if xdg:
        dirs.append(Path(xdg) / "huggingface" / "hub")
    dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    seen: set[str] = set()
    out: List[Path] = []
    for p in dirs:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _ollama_models_dir() -> Path:
    custom = os.getenv("OLLAMA_MODELS")
    if custom:
        return Path(custom)
    return Path.home() / ".ollama" / "models"


def _gguf_search_roots() -> List[Path]:
    roots: List[Path] = [
        Path.home() / "models",
        Path.home() / "llm",
        Path.home() / "Downloads",
        Path.home() / ".local" / "share" / "models",
        Path.home() / "Documents" / "models",
    ]
    if platform.system() == "Windows":
        for letter in ("D", "E"):
            roots.append(Path(f"{letter}:/models"))
    extra = os.getenv("SHIPAI_GGUF_PATHS", "")
    for part in extra.split(","):
        part = part.strip()
        if part:
            roots.append(Path(part))
    return [p for p in roots if p]


def _hf_folder_to_name(folder_name: str) -> str:
    if folder_name.startswith("models--"):
        parts = folder_name[len("models--") :].split("--", 1)
        if len(parts) == 2:
            return f"{parts[0]}/{parts[1].replace('--', '-')}"
    return folder_name


def _guess_quantization(path: Path, name: str) -> Optional[str]:
    for text in (name, path.name, str(path.parent.name)):
        m = _QUANT_PATTERN.search(text)
        if m:
            return m.group(1).upper()
    return None


def _dir_size_gb(path: Path) -> Optional[float]:
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / (1024**3), 2)
    except OSError:
        return None


def scan_huggingface_cache() -> List[DiscoveredModel]:
    found: List[DiscoveredModel] = []
    for hub in _hf_cache_dirs():
        if not hub.is_dir():
            continue
        for entry in hub.iterdir():
            if not entry.is_dir() or not entry.name.startswith("models--"):
                continue
            name = _hf_folder_to_name(entry.name)
            found.append(
                DiscoveredModel(
                    name=name,
                    canonical_id=normalize_canonical_id(name),
                    source="huggingface_cache",
                    sources=["huggingface_cache"],
                    path=str(entry),
                    size_gb=_dir_size_gb(entry),
                    is_served=False,
                    needs_runtime=True,
                    on_disk_only=True,
                    quantization=_guess_quantization(entry, name),
                )
            )
    return found


def scan_ollama_disk() -> List[DiscoveredModel]:
    """Read Ollama library manifests when API is down or as supplement."""
    found: List[DiscoveredModel] = []
    library = (
        _ollama_models_dir()
        / "manifests"
        / "registry.ollama.ai"
        / "library"
    )
    if not library.is_dir():
        return found

    for model_dir in library.iterdir():
        if not model_dir.is_dir():
            continue
        for tag_dir in model_dir.iterdir():
            if not tag_dir.is_dir():
                continue
            tag_name = tag_dir.name
            full_name = f"{model_dir.name}:{tag_name}"
            blob_size = _dir_size_gb(tag_dir)
            found.append(
                DiscoveredModel(
                    name=full_name,
                    canonical_id=normalize_canonical_id(full_name),
                    source="ollama_disk",
                    sources=["ollama_disk"],
                    path=str(tag_dir),
                    size_gb=blob_size,
                    is_served=False,
                    needs_runtime=True,
                    on_disk_only=True,
                    quantization=_guess_quantization(tag_dir, full_name),
                )
            )
    return found


def scan_gguf_files(max_files: int = 100, max_depth: int = 6) -> List[DiscoveredModel]:
    found: List[DiscoveredModel] = []
    count = 0
    for root in _gguf_search_roots():
        if not root.is_dir() or count >= max_files:
            continue
        try:
            for path in root.rglob("*.gguf"):
                if count >= max_files:
                    break
                depth = len(path.relative_to(root).parts)
                if depth > max_depth:
                    continue
                try:
                    size_gb = round(path.stat().st_size / (1024**3), 2)
                except OSError:
                    size_gb = None
                name = path.stem
                found.append(
                    DiscoveredModel(
                        name=name,
                        canonical_id=normalize_canonical_id(name),
                        source="gguf_file",
                        sources=["gguf_file"],
                        path=str(path),
                        size_gb=size_gb,
                        is_served=False,
                        needs_runtime=True,
                        on_disk_only=True,
                        quantization=_guess_quantization(path, name),
                    )
                )
                count += 1
        except OSError:
            continue
    return found


def inventory_from_runtimes(runtimes: Iterable[LLMRuntime]) -> List[DiscoveredModel]:
    items: List[DiscoveredModel] = []
    for rt in runtimes:
        if not rt.is_running:
            continue
        runtime_label = "ollama" if rt.name == "ollama_custom" else rt.name
        for model in rt.available_models:
            src = runtime_label
            items.append(
                DiscoveredModel(
                    name=model.name,
                    canonical_id=normalize_canonical_id(model.name),
                    source=src,
                    sources=[src],
                    path=None,
                    size_gb=model.size_gb,
                    runtime_name=runtime_label,
                    serving_runtime=runtime_label,
                    is_served=True,
                    needs_runtime=False,
                    on_disk_only=False,
                    quantization=model.quantization,
                )
            )
    return items


def _merge_group(group: List[DiscoveredModel]) -> DiscoveredModel:
    """Merge duplicate discoveries into one record."""
    served = [g for g in group if g.is_served]
    primary = served[0] if served else group[0]

    all_sources: List[str] = []
    for g in group:
        for s in g.sources or [g.source]:
            if s not in all_sources:
                all_sources.append(s)

    is_served = any(g.is_served for g in group)
    serving = next((g.serving_runtime for g in group if g.serving_runtime), None)
    if not serving and is_served:
        serving = primary.runtime_name

    on_disk_only = not is_served and any(
        g.on_disk_only or g.source in ("huggingface_cache", "gguf_file", "ollama_disk")
        for g in group
    )
    needs_runtime = not is_served and any(g.needs_runtime for g in group)

    best_path = next((g.path for g in group if g.path), None)
    size = next((g.size_gb for g in group if g.size_gb is not None), None)
    quant = next((g.quantization for g in group if g.quantization), None)

    return DiscoveredModel(
        name=primary.name,
        canonical_id=primary.canonical_id,
        source=primary.source,
        sources=all_sources,
        path=best_path,
        size_gb=size,
        runtime_name=primary.runtime_name,
        serving_runtime=serving,
        is_served=is_served,
        needs_runtime=needs_runtime,
        on_disk_only=on_disk_only,
        quantization=quant,
    )


def _cluster_for_merge(items: List[DiscoveredModel]) -> List[List[DiscoveredModel]]:
    """Union-find style clustering by canonical id and fuzzy name match."""
    clusters: List[List[DiscoveredModel]] = []
    for item in items:
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            if item.canonical_id == rep.canonical_id or names_likely_same(
                item.name, rep.name
            ):
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
    return clusters


def consolidate_inventory(raw: Sequence[DiscoveredModel]) -> List[DiscoveredModel]:
    clusters = _cluster_for_merge(list(raw))
    return [_merge_group(c) for c in clusters]


def scan_all_artifacts() -> List[DiscoveredModel]:
    return (
        scan_huggingface_cache()
        + scan_ollama_disk()
        + scan_gguf_files()
    )


def build_inventory(runtimes: List[LLMRuntime]) -> List[DiscoveredModel]:
    """
    Full Phase B inventory: runtime tags + disk artifacts, merged and cross-linked.
    """
    raw: List[DiscoveredModel] = []
    raw.extend(inventory_from_runtimes(runtimes))
    raw.extend(scan_all_artifacts())

    ollama_up = any(
        r.is_running and r.name in ("ollama", "ollama_custom") for r in runtimes
    )
    if ollama_up:
        served_names = {
            normalize_canonical_id(m.name)
            for r in runtimes
            if r.is_running and r.name in ("ollama", "ollama_custom")
            for m in r.available_models
        }
        raw = [
            i
            for i in raw
            if not (i.source == "ollama_disk" and i.canonical_id in served_names)
        ]

    return consolidate_inventory(raw)
