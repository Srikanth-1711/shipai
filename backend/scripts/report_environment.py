#!/usr/bin/env python3
"""Print EnvironmentReport for the current machine (Phase A diagnostic)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.install.environment import build_environment_report, format_environment_report


async def main() -> None:
    report = await build_environment_report()
    print(format_environment_report(report))
    running = [r.name for r in report.runtimes if r.is_running]
    print("\n--- summary ---")
    print(f"OS: {report.hardware.os_name} | GPU: {report.hardware.gpu_name or 'none'}")
    print(
        f"VRAM: {report.hardware.gpu_vram_total_gb}GB total, "
        f"{report.hardware.gpu_vram_free_gb}GB free"
    )
    print(f"RAM: {report.hardware.ram_available_gb}/{report.hardware.ram_total_gb} GB")
    print(f"Runtimes up: {running or 'none'}")
    print(f"Inventory items: {len(report.inventory)}")
    served = sum(1 for i in report.inventory if i.is_served)
    needs_rt = sum(1 for i in report.inventory if i.needs_runtime)
    on_disk = sum(1 for i in report.inventory if i.on_disk_only)
    print(f"  served: {served} | needs_runtime: {needs_rt} | on_disk_only: {on_disk}")
    print(f"Conflicts: {report.conflicts or 'none'}")


if __name__ == "__main__":
    asyncio.run(main())
