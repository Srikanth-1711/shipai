"""
ShipAI -- License Key System
Offline-first license validation. No phone-home required.
Keys are cryptographically signed so they can be validated locally.

Tiers:
- free: Basic templates, 3 infra patterns
- starter ($19): All templates, 5 infra patterns
- pro ($49): Everything + priority support
- enterprise (custom): White-label, custom policies
"""
import hashlib
import hmac
import json
import time
import logging
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger(__name__)

# License tiers and their feature maps
LICENSE_TIERS = {
    "free": {
        "label": "Free",
        "price": "$0",
        "templates": ["rag_chatbot", "data_analyzer"],
        "infra_patterns": ["rate_limiting", "caching", "api_gateway"],
        "max_projects": 3,
        "builder_bot": False,
        "priority_support": False,
        "custom_templates": False,
        "features": [
            "2 project templates",
            "3 production infra patterns",
            "Up to 3 projects",
            "Community support",
        ],
    },
    "starter": {
        "label": "Starter",
        "price": "$19/month",
        "templates": ["rag_chatbot", "data_analyzer", "multi_agent"],
        "infra_patterns": ["rate_limiting", "caching", "api_gateway", "circuit_breaker", "message_queue"],
        "max_projects": 10,
        "builder_bot": False,
        "priority_support": False,
        "custom_templates": False,
        "features": [
            "All 3 project templates",
            "5 production infra patterns",
            "Up to 10 projects",
            "Email support",
        ],
    },
    "pro": {
        "label": "Pro",
        "price": "$49/month",
        "templates": ["rag_chatbot", "data_analyzer", "multi_agent"],
        "infra_patterns": ["rate_limiting", "caching", "api_gateway", "circuit_breaker", "message_queue", "load_balancing", "auto_scaling"],
        "max_projects": -1,  # unlimited
        "builder_bot": True,
        "priority_support": True,
        "custom_templates": True,
        "features": [
            "All templates + custom",
            "All 7 production infra patterns",
            "Unlimited projects",
            "Builder Bot (no-code)",
            "Priority support",
        ],
    },
    "enterprise": {
        "label": "Enterprise",
        "price": "Custom",
        "templates": ["rag_chatbot", "data_analyzer", "multi_agent"],
        "infra_patterns": ["rate_limiting", "caching", "api_gateway", "circuit_breaker", "message_queue", "load_balancing", "auto_scaling"],
        "max_projects": -1,
        "builder_bot": True,
        "priority_support": True,
        "custom_templates": True,
        "features": [
            "Everything in Pro",
            "White-label branding",
            "Custom scaling policies",
            "Dedicated support",
            "SLA guarantee",
        ],
    },
}

# Secret used to sign license keys (in production, use a proper key management system)
_LICENSE_SECRET = settings.SECRET_KEY.encode()


@dataclass
class LicenseInfo:
    """Decoded license information."""
    key: str
    tier: str
    email: str
    issued_at: str
    expires_at: str
    is_valid: bool
    is_expired: bool
    days_remaining: int
    features: dict


def generate_license_key(
    tier: str = "free",
    email: str = "",
    duration_days: int = 365,
) -> dict:
    """
    Generate a cryptographically signed license key.
    Format: SK-{tier}-{payload}-{signature}
    """
    if tier not in LICENSE_TIERS:
        return {"error": f"Invalid tier: {tier}. Valid: {list(LICENSE_TIERS.keys())}"}

    issued_at = datetime.utcnow().isoformat()
    expires_at = (datetime.utcnow() + timedelta(days=duration_days)).isoformat()

    # Build payload
    payload = {
        "tier": tier,
        "email": email,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "nonce": secrets.token_hex(4),
    }

    # Encode payload as hex
    payload_json = json.dumps(payload, separators=(",", ":"))
    payload_hex = payload_json.encode().hex()

    # Sign it
    signature = hmac.new(
        _LICENSE_SECRET,
        payload_hex.encode(),
        hashlib.sha256,
    ).hexdigest()[:16]

    # Build key
    tier_prefix = tier[0].upper()  # F, S, P, E
    key = f"SK-{tier_prefix}-{payload_hex[:24]}-{signature}"

    # Store full key data for validation
    license_data = {
        "key": key,
        "tier": tier,
        "email": email,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "payload_hex": payload_hex,
        "signature": signature,
    }

    # Save to local license store
    _save_license(license_data)

    logger.info(f"License generated: {key} (tier={tier}, email={email})")

    return {
        "key": key,
        "tier": tier,
        "tier_label": LICENSE_TIERS[tier]["label"],
        "email": email,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "features": LICENSE_TIERS[tier]["features"],
    }


def validate_license_key(key: str) -> LicenseInfo:
    """
    Validate a license key offline.
    Checks signature integrity and expiration.
    """
    # Check if it's a stored license
    stored = _load_license(key)
    if not stored:
        return LicenseInfo(
            key=key,
            tier="free",
            email="",
            issued_at="",
            expires_at="",
            is_valid=False,
            is_expired=False,
            days_remaining=0,
            features=LICENSE_TIERS["free"],
        )

    # Verify signature
    expected_sig = hmac.new(
        _LICENSE_SECRET,
        stored["payload_hex"].encode(),
        hashlib.sha256,
    ).hexdigest()[:16]

    is_valid = hmac.compare_digest(expected_sig, stored["signature"])

    # Check expiration
    expires_at = datetime.fromisoformat(stored["expires_at"])
    now = datetime.utcnow()
    is_expired = now > expires_at
    days_remaining = max(0, (expires_at - now).days)

    tier = stored["tier"] if is_valid and not is_expired else "free"

    return LicenseInfo(
        key=key,
        tier=tier,
        email=stored.get("email", ""),
        issued_at=stored.get("issued_at", ""),
        expires_at=stored.get("expires_at", ""),
        is_valid=is_valid,
        is_expired=is_expired,
        days_remaining=days_remaining,
        features=LICENSE_TIERS.get(tier, LICENSE_TIERS["free"]),
    )


def get_current_license() -> LicenseInfo:
    """Get the currently active license (or free tier if none)."""
    license_path = _get_license_store_path()
    if license_path.exists():
        try:
            data = json.loads(license_path.read_text(encoding="utf-8"))
            if "active_key" in data:
                return validate_license_key(data["active_key"])
        except Exception as e:
            logger.error(f"Error reading license: {e}")

    # Default to free
    return LicenseInfo(
        key="",
        tier="free",
        email="",
        issued_at="",
        expires_at="",
        is_valid=True,
        is_expired=False,
        days_remaining=-1,
        features=LICENSE_TIERS["free"],
    )


def activate_license(key: str) -> dict:
    """Activate a license key on this machine."""
    info = validate_license_key(key)

    if not info.is_valid:
        return {"status": "invalid", "error": "Invalid license key"}
    if info.is_expired:
        return {"status": "expired", "error": "License key has expired"}

    # Save as active
    license_path = _get_license_store_path()
    try:
        data = {}
        if license_path.exists():
            data = json.loads(license_path.read_text(encoding="utf-8"))
        data["active_key"] = key
        license_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        return {"status": "error", "error": str(e)}

    logger.info(f"License activated: {info.tier} tier")
    return {
        "status": "activated",
        "tier": info.tier,
        "email": info.email,
        "expires_at": info.expires_at,
        "days_remaining": info.days_remaining,
        "features": info.features,
    }


def check_feature_access(feature: str) -> bool:
    """Check if the current license allows a specific feature."""
    license_info = get_current_license()
    tier_config = LICENSE_TIERS.get(license_info.tier, LICENSE_TIERS["free"])

    # Check template access
    if feature.startswith("template:"):
        template_id = feature.split(":")[1]
        return template_id in tier_config["templates"]

    # Check infra pattern access
    if feature.startswith("infra:"):
        pattern = feature.split(":")[1]
        return pattern in tier_config["infra_patterns"]

    # Check boolean features
    if feature == "builder_bot":
        return tier_config.get("builder_bot", False)
    if feature == "custom_templates":
        return tier_config.get("custom_templates", False)
    if feature == "priority_support":
        return tier_config.get("priority_support", False)

    # Check project limit
    if feature == "create_project":
        max_projects = tier_config.get("max_projects", 3)
        if max_projects == -1:
            return True
        # Count existing projects
        projects_dir = Path(settings.PROJECTS_DIR)
        if projects_dir.exists():
            project_count = sum(1 for p in projects_dir.iterdir() if p.is_dir())
            return project_count < max_projects
        return True

    return False


def get_tier_comparison() -> list[dict]:
    """Get a comparison of all tiers for display."""
    return [
        {
            "tier": tier_id,
            "label": tier["label"],
            "price": tier["price"],
            "features": tier["features"],
            "templates_count": len(tier["templates"]),
            "infra_patterns_count": len(tier["infra_patterns"]),
            "max_projects": "Unlimited" if tier["max_projects"] == -1 else tier["max_projects"],
            "builder_bot": tier["builder_bot"],
        }
        for tier_id, tier in LICENSE_TIERS.items()
    ]


# --- Internal helpers ---

def _get_license_store_path() -> Path:
    """Get path to the local license store."""
    store_dir = Path.home() / ".shipai"
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir / "license.json"


def _save_license(license_data: dict):
    """Save a license to local store."""
    store_path = _get_license_store_path()
    try:
        data = {}
        if store_path.exists():
            data = json.loads(store_path.read_text(encoding="utf-8"))
        if "licenses" not in data:
            data["licenses"] = {}
        data["licenses"][license_data["key"]] = license_data
        store_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to save license: {e}")


def _load_license(key: str) -> Optional[dict]:
    """Load a license from local store."""
    store_path = _get_license_store_path()
    if not store_path.exists():
        return None
    try:
        data = json.loads(store_path.read_text(encoding="utf-8"))
        return data.get("licenses", {}).get(key)
    except Exception:
        return None
