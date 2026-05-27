"""Runtime capability checks for optional dependencies.

Used to provide startup diagnostics and graceful degradation reporting.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class CapabilityStatus:
    name: str
    available: bool
    required: bool
    module: str
    reason: str = ""


OPTIONAL_CAPABILITIES = {
    "chat_orchestration": ["langgraph", "langchain_community", "langchain_core"],
    "api_server": ["fastapi", "pydantic", "pydantic_settings"],
    "scheduler": ["feedparser"],
}

REQUIRED_CAPABILITIES = {
    "core_runtime": ["json", "pathlib"],
}


def _has_module(mod_name: str) -> bool:
    try:
        importlib.import_module(mod_name)
        return True
    except Exception:
        return False


def get_runtime_capabilities() -> Dict[str, List[dict]]:
    required: List[CapabilityStatus] = []
    optional: List[CapabilityStatus] = []

    for group, modules in REQUIRED_CAPABILITIES.items():
        for mod in modules:
            ok = _has_module(mod)
            required.append(
                CapabilityStatus(
                    name=group,
                    available=ok,
                    required=True,
                    module=mod,
                    reason="" if ok else "missing required dependency",
                )
            )

    for group, modules in OPTIONAL_CAPABILITIES.items():
        for mod in modules:
            ok = _has_module(mod)
            optional.append(
                CapabilityStatus(
                    name=group,
                    available=ok,
                    required=False,
                    module=mod,
                    reason="" if ok else "optional feature disabled",
                )
            )

    return {
        "required": [asdict(x) for x in required],
        "optional": [asdict(x) for x in optional],
    }


def summarize_capabilities() -> dict:
    caps = get_runtime_capabilities()
    req_ok = all(x["available"] for x in caps["required"])
    opt_avail = sum(1 for x in caps["optional"] if x["available"])
    return {
        "required_ok": req_ok,
        "optional_available": opt_avail,
        "optional_total": len(caps["optional"]),
        "details": caps,
    }

