"""Assemble the full environment report (hardware + runtimes + inventory + conflicts)."""
from __future__ import annotations

import json
from dataclasses import asdict

from app.install.conflicts import detect_conflicts
from app.install.hardware_profile import detect_hardware_profile
from app.install.inventory import build_inventory
from app.install.runtime_detector import detect_all_runtimes
from app.install.types import EnvironmentReport


async def build_environment_report(
    client=None,
) -> EnvironmentReport:
    hardware = detect_hardware_profile()
    runtimes = await detect_all_runtimes(client=client)
    inventory = build_inventory(runtimes)
    conflicts = detect_conflicts(runtimes, hardware)
    return EnvironmentReport(
        hardware=hardware,
        runtimes=runtimes,
        inventory=inventory,
        conflicts=conflicts,
    )


def environment_report_to_dict(report: EnvironmentReport) -> dict:
    return asdict(report)


def format_environment_report(report: EnvironmentReport) -> str:
    return json.dumps(environment_report_to_dict(report), indent=2, default=str)
