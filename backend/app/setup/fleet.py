"""
Setup Fleet — fusion intelligence + acquire + validate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import httpx

from app.install.fusion_intelligence import run_fusion_intelligence
from app.install.environment import build_environment_report
from app.install.model_negotiator import DEFAULT_CHAT_NODES
from app.install.model_plan import ModelPlan, save_model_plan
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


async def acquirer_step(
    state: SetupState,
    cb: ProgressCallback,
    *,
    auto_pull: bool,
) -> None:
    assert state.plan
    to_pull = list(state.plan.models_to_download)
    if state.plan.final_three:
        for mid in state.plan.final_three.values():
            if mid and mid not in to_pull:
                to_pull.append(mid)
    if state.plan.embedding and state.plan.embedding.model:
        if state.plan.embedding.model not in to_pull:
            to_pull.append(state.plan.embedding.model)

    installed = {
        m.name.lower()
        for m in (state.report.inventory if state.report else [])
        if m.is_served
    }
    to_pull = [m for m in to_pull if m.lower() not in installed]

    if not to_pull:
        _emit(cb, "Acquirer", "All models already on disk")
        state.steps.append("acquirer:skip")
        return

    if not auto_pull:
        _emit(cb, "Acquirer", f"Skipped pull ({len(to_pull)} needed) — run ollama pull manually")
        state.steps.append("acquirer:manual")
        return

    _emit(cb, "Acquirer", f"Pulling {len(to_pull)} model(s)...")
    result = await acquire_models(state.plan, auto_pull=True, primary_runtime=state.plan.primary_runtime)
    state.pulled_models = result.pulled
    state.failed_pulls = result.failed
    state.steps.append("acquirer:ok")
    if result.pulled:
        _emit(cb, "Acquirer", f"Pulled: {', '.join(result.pulled)}")
    if result.failed:
        state.warnings.append(f"Failed pulls: {', '.join(result.failed)}")
        _emit(cb, "Acquirer", f"Failed: {', '.join(result.failed)}")


async def validator_step(
    state: SetupState,
    client: httpx.AsyncClient | None,
    cb: ProgressCallback,
    *,
    use_case: str | None,
    offline: bool,
    use_gemini_explain: bool,
) -> None:
    _emit(cb, "Validator", "Re-scan and refresh plan after downloads...")
    state.report = await build_environment_report(client=client)
    state.plan = await run_fusion_intelligence(
        use_case=use_case,
        offline=offline,
        use_gemini_explain=use_gemini_explain,
        client=client,
        on_progress=None,
    )
    state.plan.warnings = list(set(state.warnings + state.plan.warnings))
    save_model_plan(state.plan)
    state.steps.append("validator:ok")

    installed_nodes = sum(
        1
        for n in DEFAULT_CHAT_NODES
        if n in state.plan.node_assignments
        and state.plan.node_assignments[n].model
        and state.plan.node_assignments[n].status
        in ("installed", "user_suggested")
        and state.plan.node_assignments[n].model
    )
    state.success = state.plan.primary_runtime != "none" and installed_nodes >= 1 and not state.failed_pulls
    _emit(cb, "Validator", f"{installed_nodes}/{len(DEFAULT_CHAT_NODES)} nodes ready")


async def run_setup_fleet(
    *,
    auto_pull: bool = True,
    use_case: str | None = None,
    offline: bool = False,
    use_gemini_explain: bool = False,
    client: httpx.AsyncClient | None = None,
    on_progress: ProgressCallback = None,
) -> SetupResult:
    state = SetupState()
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=12.0, follow_redirects=True)

    try:
        state.plan = await run_fusion_intelligence(
            use_case=use_case,
            offline=offline,
            use_gemini_explain=False,
            client=client,
            on_progress=on_progress,
        )
        state.warnings.extend(state.plan.warnings)
        state.steps.append("fusion:ok")
        state.report = await build_environment_report(client=client)

        await acquirer_step(state, on_progress, auto_pull=auto_pull)
        await validator_step(
            state,
            client,
            on_progress,
            use_case=use_case,
            offline=offline,
            use_gemini_explain=use_gemini_explain,
        )
    finally:
        if owns_client and client:
            await client.aclose()

    return SetupResult(state=state, plan=state.plan)
