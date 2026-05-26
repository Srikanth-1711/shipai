"""
Setup Fleet — orchestrates bootstrap phases A→D plus acquire + validate.

Agents (logical roles, mostly deterministic):
  Scout      → environment report (A+B)
  Librarian  → capability matrix + catalog (C)
  Negotiator → model plan draft (D)
  Acquirer   → ollama pull for missing models
  Validator  → re-scan + final model_plan.json
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import httpx

from app.install.capability_fetcher import fetch_ollama_catalog, get_capability_matrix
from app.install.environment import build_environment_report
from app.install.model_negotiator import negotiate_from_report
from app.install.model_plan import ModelPlan, save_model_plan
from app.install.model_negotiator import CHAT_NODES
from app.setup.acquirer import acquire_models
from app.setup.state import SetupState

logger = logging.getLogger("shipai.setup")

ProgressCallback = Optional[Callable[[str, str], None]]


@dataclass
class SetupResult:
    state: SetupState
    plan: ModelPlan | None = None


def _emit(cb: ProgressCallback, agent: str, message: str) -> None:
    if cb:
        cb(agent, message)
    logger.info("[%s] %s", agent, message)


async def scout_step(state: SetupState, client: httpx.AsyncClient | None, cb: ProgressCallback) -> None:
    _emit(cb, "Scout", "Detecting hardware, runtimes, and inventory...")
    state.report = await build_environment_report(client=client)
    state.steps.append("scout:ok")
    n_inv = len(state.report.inventory)
    n_rt = sum(1 for r in state.report.runtimes if r.is_running)
    _emit(cb, "Scout", f"Found {n_rt} runtime(s), {n_inv} inventory item(s)")


async def librarian_step(state: SetupState, client: httpx.AsyncClient | None, cb: ProgressCallback) -> None:
    _emit(cb, "Librarian", "Loading capability matrix and model catalog...")
    state.matrix = await get_capability_matrix(client=client)
    state.catalog = await fetch_ollama_catalog(client=client, matrix_fallback=state.matrix)
    state.steps.append("librarian:ok")
    _emit(
        cb,
        "Librarian",
        f"Matrix v{state.matrix.get('schema_version', '?')} — "
        f"{len(state.matrix.get('models', {}))} known models",
    )


async def negotiator_step(state: SetupState, cb: ProgressCallback) -> None:
    _emit(cb, "Negotiator", "Assigning models per ShipAI node from genuine inventory...")
    assert state.report and state.matrix
    neg = negotiate_from_report(state.report, state.matrix, state.catalog)
    state.plan = ModelPlan(
        primary_runtime=neg.primary_runtime,
        runtime_base_url=neg.runtime_base_url,
        embedding=neg.embedding,
        node_assignments=neg.node_assignments,
        models_to_download=neg.models_to_download,
        warnings=list(neg.warnings),
        metadata={
            "setup_fleet": True,
            "phase": "negotiator_draft",
        },
    )
    state.warnings.extend(neg.warnings)
    state.steps.append("negotiator:ok")
    dl = len(state.plan.models_to_download)
    _emit(cb, "Negotiator", f"Draft plan ready — {dl} model(s) need download")


async def acquirer_step(
    state: SetupState,
    cb: ProgressCallback,
    *,
    auto_pull: bool,
) -> None:
    assert state.plan
    to_pull = list(state.plan.models_to_download)
    for asn in state.plan.node_assignments.values():
        if asn.status == "needs_download" and asn.model and asn.model not in to_pull:
            to_pull.append(asn.model)

    if not to_pull:
        _emit(cb, "Acquirer", "All required models already on disk — nothing to pull")
        state.steps.append("acquirer:skip")
        return

    if not auto_pull:
        _emit(cb, "Acquirer", f"Skipped auto-pull ({len(to_pull)} model(s)) — run manually")
        state.steps.append("acquirer:manual")
        return

    _emit(cb, "Acquirer", f"Pulling {len(to_pull)} model(s) via Ollama...")
    result = await acquire_models(
        state.plan,
        auto_pull=True,
        primary_runtime=state.plan.primary_runtime,
    )
    state.pulled_models = result.pulled
    state.failed_pulls = result.failed
    state.steps.append("acquirer:ok")
    if result.pulled:
        _emit(cb, "Acquirer", f"Pulled: {', '.join(result.pulled)}")
    if result.failed:
        state.warnings.append(f"Failed to pull: {', '.join(result.failed)}")
        _emit(cb, "Acquirer", f"Failed: {', '.join(result.failed)}")


async def validator_step(
    state: SetupState,
    client: httpx.AsyncClient | None,
    cb: ProgressCallback,
) -> None:
    _emit(cb, "Validator", "Re-scanning environment and finalizing model plan...")
    state.report = await build_environment_report(client=client)
    neg = negotiate_from_report(state.report, state.matrix or {}, state.catalog)
    state.plan = ModelPlan(
        primary_runtime=neg.primary_runtime,
        runtime_base_url=neg.runtime_base_url,
        embedding=neg.embedding,
        node_assignments=neg.node_assignments,
        models_to_download=neg.models_to_download,
        warnings=list(set(state.warnings + neg.warnings)),
        metadata={
            "setup_fleet": True,
            "phase": "final",
            "inventory_count": len(state.report.inventory),
            "pulled_this_session": state.pulled_models,
        },
    )
    save_model_plan(state.plan)
    state.steps.append("validator:ok")

    installed_nodes = sum(
        1
        for n in CHAT_NODES
        if n in state.plan.node_assignments
        and state.plan.node_assignments[n].status == "installed"
        and state.plan.node_assignments[n].model
    )
    state.success = (
        state.plan.primary_runtime != "none"
        and installed_nodes >= 1
        and not state.failed_pulls
    )
    _emit(
        cb,
        "Validator",
        f"Plan saved — {installed_nodes}/{len(CHAT_NODES)} chat nodes have installed models",
    )


async def run_setup_fleet(
    *,
    auto_pull: bool = True,
    client: httpx.AsyncClient | None = None,
    on_progress: ProgressCallback = None,
) -> SetupResult:
    """
    Run the full Setup Fleet end-to-end.
    Writes ~/.shipai/model_plan.json on success.
    """
    state = SetupState()
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=10.0)

    try:
        await scout_step(state, client, on_progress)
        await librarian_step(state, client, on_progress)
        await negotiator_step(state, on_progress)
        await acquirer_step(state, on_progress, auto_pull=auto_pull)
        await validator_step(state, client, on_progress)
    finally:
        if owns_client and client:
            await client.aclose()

    return SetupResult(state=state, plan=state.plan)
