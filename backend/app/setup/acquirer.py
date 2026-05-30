"""Setup Fleet — Acquirer: download models via Ollama (or report manual steps)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from app.install.model_plan import ModelPlan

logger = logging.getLogger("shipai.setup.acquirer")


@dataclass
class AcquirerResult:
    pulled: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)


def _unique_download_list(plan: ModelPlan) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for m in plan.models_to_download:
        if m and m not in seen:
            seen.add(m)
            names.append(m)
    for asn in plan.node_assignments.values():
        if asn.status == "needs_download" and asn.model and asn.model not in seen:
            seen.add(asn.model)
            names.append(asn.model)
    return names


async def acquire_models(
    plan: ModelPlan,
    *,
    auto_pull: bool = True,
    primary_runtime: str = "ollama",
) -> AcquirerResult:
    result = AcquirerResult()
    to_pull = _unique_download_list(plan)

    if not to_pull:
        return result

    if primary_runtime not in ("ollama", "none") and primary_runtime != "ollama_custom":
        result.skipped = list(to_pull)
        logger.info("Downloads skipped — primary runtime is %s", primary_runtime)
        return result

    if not auto_pull:
        result.skipped = list(to_pull)
        return result

    from app.services.ollama_service import ollama_service

    health = await ollama_service.health_check()
    if health.get("status") != "healthy":
        result.failed = list(to_pull)
        return result

    for model in to_pull:
        logger.info("Pulling %s ...", model)
        pull = await ollama_service.pull_model(model)
        if pull.get("status") == "success":
            result.pulled.append(model)
        else:
            result.failed.append(model)
            logger.error("Pull failed for %s: %s", model, pull.get("error"))

    return result
