#!/usr/bin/env python3
"""Show capability matrix source and top models feasible for this machine."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.install.capability_fetcher import (
    bundled_matrix_path,
    cache_path,
    fetch_ollama_catalog,
    get_capability_matrix,
    rank_catalog_for_hardware,
)
from app.install.hardware_profile import detect_hardware_profile


async def main() -> None:
    hw = detect_hardware_profile()
    vram = hw.gpu_vram_free_gb if hw.has_gpu else 0.0
    if hw.is_apple_silicon and hw.unified_memory_gb:
        vram = max(vram, hw.unified_memory_gb * 0.6)

    matrix = await get_capability_matrix(force_refresh=False)
    catalog = await fetch_ollama_catalog(matrix_fallback=matrix)

    ranked = rank_catalog_for_hardware(
        catalog,
        matrix,
        effective_vram_gb=max(vram, 0.5),
        effective_ram_gb=hw.ram_available_gb,
        ram_total_gb=hw.ram_total_gb,
        has_cuda=hw.has_cuda,
        top_n=10,
    )
    top3 = ranked[:3]

    print("Capability matrix")
    print(f"  bundled: {bundled_matrix_path()}")
    print(f"  cache:   {cache_path()} ({'exists' if cache_path().is_file() else 'missing'})")
    print(f"  version: {matrix.get('schema_version')}  updated: {matrix.get('updated', '?')}")
    print(f"  models in matrix: {len(matrix.get('models', {}))}")
    print()
    print(f"Hardware budget: VRAM~{vram:.1f}GB  RAM avail {hw.ram_available_gb}GB")
    print(f"Top feasible (up to 10): {len(ranked)}")
    for i, m in enumerate(ranked, 1):
        print(f"  {i}. {m.get('name')}  ({m.get('size_gb', '?')} GB)")
    print()
    print("Top 3 for negotiation (Phase D):")
    for i, m in enumerate(top3, 1):
        print(f"  {i}. {m.get('name')}")


if __name__ == "__main__":
    asyncio.run(main())
