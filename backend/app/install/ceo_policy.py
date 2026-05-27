"""CEO policy — editorial guardrails (Layer 3), no LLM."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUNDLED = _REPO_ROOT / "intelligence" / "ceo_policy.json"
_CACHE = Path.home() / ".shipai" / "cache" / "ceo_policy.json"

_DEFAULT: dict[str, Any] = {
    "schema_version": "1",
    "approved_families": ["qwen", "gemma", "llama", "mistral", "phi", "deepseek"],
    "banned": [],
    "family_boost": 0.1,
    "flagship_boost": 0.15,
    "role_preferences": {},
}


def load_ceo_policy() -> dict[str, Any]:
    for path in (_CACHE, _BUNDLED):
        if path.is_file():
            try:
                with path.open(encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    return dict(_DEFAULT)


def model_matches_family(model_id: str, family: str) -> bool:
    return family.lower() in model_id.lower()


def is_banned(model_id: str, policy: dict[str, Any]) -> bool:
    low = model_id.lower()
    for b in policy.get("banned", []):
        if b.lower() in low:
            return True
    return False


def is_approved(model_id: str, policy: dict[str, Any]) -> bool:
    families = policy.get("approved_families", [])
    if not families:
        return True
    return any(model_matches_family(model_id, f) for f in families)


def ceo_boost_for_model(model_id: str, policy: dict[str, Any], role: str | None = None) -> float:
    boost = 0.0
    if not is_approved(model_id, policy) or is_banned(model_id, policy):
        return -1.0
    flagship = policy.get("flagship_multi_agent", "")
    if flagship and flagship.lower() in model_id.lower():
        boost += float(policy.get("flagship_boost", 0.15))
    for fam in policy.get("approved_families", []):
        if model_matches_family(model_id, fam):
            boost += float(policy.get("family_boost", 0.1))
            break
    if role:
        prefs = policy.get("role_preferences", {}).get(role, {})
        for pref in prefs.get("prefer", []):
            if pref.lower() in model_id.lower():
                boost += 0.08
                break
    return boost
