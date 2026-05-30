# ShipAI Compiler Pipeline (End-to-End)

This document describes the end-to-end flow that turns a user idea into a generated runnable project.

## 1) Installer (CLI)

Entry points:
- `python -m shipai install`
- `python -m shipai plan`

Key modules:
- `backend/app/install/fusion_intelligence.py` (fusion scoring: CEO policy + feasibility + web/matrix)
- `backend/app/setup/fleet.py` (scout/librarian/acquirer/validator orchestration)

Output:
- `~/.shipai/model_plan.json`
  - per-node model assignments (`interview_node`, `research_node`, `plan_node`, `explain_node`)
  - embedding model (for RAG)
  - hardware metadata and final roles (fast/medium/heavy)

## 2) Architect (Runtime / WebSocket)

Entry point:
- `python -m shipai serve`

WebSocket:
- `/ws/chat` → `backend/app/api/ws.py`

Graph:
- `backend/app/engine/orchestrator.py` (LangGraph)

Nodes:
1. `interview_node`: extract structured requirements from chat
2. `research_node`: ground decisions with MCP tools (books + GitHub)
3. `plan_node`: generate `config_json` (architecture blueprint)
4. `config_validator_node`: validate + hardware-aware overrides
5. `builder_node`: generate project files using `ShipAITemplateEngine`
6. `explain_node`: produce a human-readable explanation

## 3) Builder and Determinism

The builder writes a real project to disk using:
- `backend/app/services/template_engine.py`

Stabilization mechanisms:
- `backend/app/engine/builder_utils.py`
  - deterministic output directory name derived from a hash of `config_json`
  - `manifest.json` generation that captures:
    - config snapshot
    - config hash
    - generated files with SHA256
    - model_plan summary (best-effort)

If building fails, the graph does not crash:
- state records `build_error`
- WebSocket emits an `error` event

## 4) Artifacts

Project root contains:
- `manifest.json`
- `backend/` application code
- `Dockerfile`
- `README.md`
- conditional modules (retrieval, reranking, cache, infra scaffolds)

