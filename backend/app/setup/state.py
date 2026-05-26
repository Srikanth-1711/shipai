"""State passed through the Setup Fleet pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.install.model_plan import ModelPlan
from app.install.types import EnvironmentReport


@dataclass
class SetupState:
    report: Optional[EnvironmentReport] = None
    matrix: Optional[dict] = None
    catalog: Optional[List[dict]] = None
    plan: Optional[ModelPlan] = None
    pulled_models: List[str] = field(default_factory=list)
    failed_pulls: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
