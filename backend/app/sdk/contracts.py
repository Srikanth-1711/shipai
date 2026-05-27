"""Typed SDK contracts for ShipAI orchestration and artifacts.

These abstractions are intentionally lightweight and runtime-agnostic so they
can back future Python/TS SDKs and remote execution APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ArtifactRef:
    path: str
    artifact_type: str
    sha256: str = ""
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRequest:
    task_id: str
    task_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    status: str
    message: str = ""
    artifacts: List[ArtifactRef] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class RuntimeExecutor(Protocol):
    async def run(self, request: TaskRequest) -> TaskResult:
        """Execute a task asynchronously and return structured result."""
        ...


class ToolInterface(Protocol):
    def name(self) -> str:
        ...

    async def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass
class ExecutionContract:
    contract_version: str = "1.0"
    supports_streaming: bool = True
    supports_cancellation: bool = False
    max_timeout_seconds: int = 900
    allowed_artifact_types: List[str] = field(
        default_factory=lambda: [
            "project_file",
            "manifest",
            "report",
            "log",
        ]
    )

