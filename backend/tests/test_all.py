"""
ShipAI -- Test Suite
Comprehensive tests for all backend services and API endpoints.
Run: pytest tests/ -v
"""
import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock


# ═══════════════════════════════════════════════
# Hardware Checker Tests
# ═══════════════════════════════════════════════

class TestHardwareChecker:
    """Test hardware detection and tier classification."""

    def test_check_hardware_returns_result(self):
        from app.services.hardware_checker import check_hardware
        hw = check_hardware()
        assert hw is not None
        assert hw.ram_total_gb > 0
        assert hw.cpu_cores_physical > 0
        assert hw.os_name in ("Windows", "Linux", "Darwin")

    def test_hardware_tier_classification(self):
        from app.services.hardware_checker import _determine_tier
        from app.install.types import HardwareProfile
        
        class MockProfile:
            def __init__(self, tier):
                self.scale_tier = tier
                
        def _mock_profile(tier):
            return MockProfile(tier)
            
        assert _determine_tier(_mock_profile("hyperscale"))[0] == "ultra"
        assert _determine_tier(_mock_profile("server"))[0] == "ultra"
        assert _determine_tier(_mock_profile("workstation"))[0] == "high"
        assert _determine_tier(_mock_profile("laptop"))[0] == "standard"
        assert _determine_tier(_mock_profile("cpu_cluster"))[0] == "standard"
        assert _determine_tier(_mock_profile("unknown"))[0] == "standard"

    def test_recommended_models_empty(self):
        # Modern pipeline doesn't hardcode models in hardware_checker
        from app.services.hardware_checker import check_hardware
        hw = check_hardware()
        assert len(hw.recommended_models) == 0

    def test_hardware_has_disk_info(self):
        from app.services.hardware_checker import check_hardware
        hw = check_hardware()
        assert hw.disk_total_gb > 0
        assert hw.disk_free_gb >= 0

    def test_hardware_tier_has_features(self):
        from app.services.hardware_checker import check_hardware
        hw = check_hardware()
        assert len(hw.available_features) > 0
        assert hw.tier_label != ""


# ═══════════════════════════════════════════════
# License Service Tests
# ═══════════════════════════════════════════════

class TestLicenseService:
    """Test license key generation, validation, and feature gating."""

    def test_generate_license_key(self):
        from app.services.license_service import generate_license_key
        result = generate_license_key(tier="pro", email="test@test.com")
        assert "key" in result
        assert result["tier"] == "pro"
        assert result["key"].startswith("SK-P-")

    def test_generate_free_key(self):
        from app.services.license_service import generate_license_key
        result = generate_license_key(tier="free", email="free@test.com")
        assert result["key"].startswith("SK-F-")

    def test_generate_invalid_tier(self):
        from app.services.license_service import generate_license_key
        result = generate_license_key(tier="nonexistent")
        assert "error" in result

    def test_validate_generated_key(self):
        from app.services.license_service import generate_license_key, validate_license_key
        gen = generate_license_key(tier="starter", email="val@test.com")
        info = validate_license_key(gen["key"])
        assert info.is_valid is True
        assert info.is_expired is False
        assert info.tier == "starter"

    def test_validate_invalid_key(self):
        from app.services.license_service import validate_license_key
        info = validate_license_key("SK-X-invalid-key-here")
        assert info.is_valid is False

    def test_tier_comparison(self):
        from app.services.license_service import get_tier_comparison
        tiers = get_tier_comparison()
        assert len(tiers) == 4
        tier_ids = [t["tier"] for t in tiers]
        assert "free" in tier_ids
        assert "pro" in tier_ids
        assert "enterprise" in tier_ids

    def test_feature_gating_templates(self):
        from app.services.license_service import LICENSE_TIERS
        free = LICENSE_TIERS["free"]
        pro = LICENSE_TIERS["pro"]
        assert "rag_chatbot" in free["templates"]
        assert "multi_agent" not in free["templates"]
        assert "multi_agent" in pro["templates"]

    def test_feature_gating_infra(self):
        from app.services.license_service import LICENSE_TIERS
        free = LICENSE_TIERS["free"]
        pro = LICENSE_TIERS["pro"]
        assert len(free["infra_patterns"]) == 3
        assert len(pro["infra_patterns"]) == 7


# ═══════════════════════════════════════════════
# AI Advisor Tests
# ═══════════════════════════════════════════════

class TestAIAdvisor:
    """Test AI pattern knowledge base and matching."""

    def test_all_patterns_loaded(self):
        from app.services.ai_advisor import AI_PATTERNS
        assert len(AI_PATTERNS) >= 16

    def test_get_all_patterns_categorized(self):
        from app.services.ai_advisor import get_all_patterns
        patterns = get_all_patterns()
        assert "rag" in patterns
        assert "agent" in patterns
        assert "fine_tuning" in patterns
        assert "tools" in patterns

    def test_keyword_match_rag(self):
        from app.services.ai_advisor import _keyword_match
        matches = _keyword_match("I need a simple FAQ chatbot for documentation")
        assert len(matches) > 0
        pattern_names = [m["pattern"] for m in matches]
        assert "Naive RAG" in pattern_names

    def test_keyword_match_agent(self):
        from app.services.ai_advisor import _keyword_match
        matches = _keyword_match("I need multiple roles collaborating on a workflow")
        assert len(matches) > 0
        pattern_names = [m["pattern"] for m in matches]
        assert "Multi-Agent System" in pattern_names

    def test_keyword_match_empty(self):
        from app.services.ai_advisor import _keyword_match
        matches = _keyword_match("xyzzy nonsense gibberish")
        assert len(matches) == 0

    def test_pattern_has_required_fields(self):
        from app.services.ai_advisor import AI_PATTERNS
        for key, pattern in AI_PATTERNS.items():
            assert "name" in pattern, f"{key} missing name"
            assert "category" in pattern, f"{key} missing category"
            assert "description" in pattern, f"{key} missing description"
            assert "when_to_use" in pattern, f"{key} missing when_to_use"
            assert "complexity" in pattern, f"{key} missing complexity"
            assert "min_hardware_tier" in pattern, f"{key} missing min_hardware_tier"


# ═══════════════════════════════════════════════
# Infra Generator Tests
# ═══════════════════════════════════════════════

class TestInfraGenerator:
    """Test production infrastructure pattern generation."""

    def test_tier_patterns_defined(self):
        from app.services.infra_generator import TIER_PATTERNS
        assert len(TIER_PATTERNS["free"]) == 3
        assert len(TIER_PATTERNS["starter"]) == 5
        assert len(TIER_PATTERNS["pro"]) == 9

    def test_get_infra_patterns_free(self):
        from app.services.infra_generator import get_infra_patterns
        patterns = get_infra_patterns("free")
        assert "rate_limiting" in patterns
        assert "caching" in patterns
        assert "api_gateway" in patterns
        assert "load_balancing" not in patterns

    def test_get_infra_patterns_pro(self):
        from app.services.infra_generator import get_infra_patterns
        patterns = get_infra_patterns("pro")
        assert len(patterns) == 9

    def test_generate_infra_files(self, tmp_path):
        from app.services.infra_generator import generate_infra_files
        files = generate_infra_files(str(tmp_path), tier="free")
        assert len(files) == 3
        assert any("rate_limiter.py" in f for f in files)

    def test_generate_infra_files_pro(self, tmp_path):
        from app.services.infra_generator import generate_infra_files
        files = generate_infra_files(str(tmp_path), tier="pro")
        assert len(files) == 10

    def test_get_infra_summary(self):
        from app.services.infra_generator import get_infra_summary
        summary = get_infra_summary("free")
        assert len(summary) == 9  # all 9 patterns listed
        included = [s for s in summary if s["included"]]
        assert len(included) == 3


# ═══════════════════════════════════════════════
# Template Engine Tests
# ═══════════════════════════════════════════════

class TestTemplateEngine:
    """Test project template listing and generation."""

    def test_list_templates(self):
        from app.services.template_engine import list_templates
        templates = list_templates()
        assert len(templates) == 3
        ids = [t["id"] for t in templates]
        assert "rag_chatbot" in ids
        assert "multi_agent" in ids
        assert "data_analyzer" in ids

    def test_get_template(self):
        from app.services.template_engine import get_template
        t = get_template("rag_chatbot")
        assert t is not None
        assert t["name"] == "RAG Chatbot"
        assert len(t["inputs"]) > 0

    def test_get_template_not_found(self):
        from app.services.template_engine import get_template
        t = get_template("nonexistent")
        assert t is None

    @pytest.mark.asyncio
    async def test_generate_rag_project(self, tmp_path):
        from app.services.template_engine import generate_project
        result = await generate_project(
            template_id="rag_chatbot",
            config={"project_name": "test_bot", "model": "tinyllama"},
            output_dir=str(tmp_path / "test_bot"),
            tier="free",
        )
        assert result["status"] == "success"
        assert result["files_generated"] > 0
        assert (tmp_path / "test_bot" / "backend" / "app" / "main.py").exists()
        assert (tmp_path / "test_bot" / "README.md").exists()
        assert (tmp_path / "test_bot" / "Dockerfile").exists()
        assert (tmp_path / "test_bot" / "requirements.txt").exists()

    @pytest.mark.asyncio
    async def test_generate_multi_agent_project(self, tmp_path):
        from app.services.template_engine import generate_project
        result = await generate_project(
            template_id="multi_agent",
            config={"project_name": "test_agents"},
            output_dir=str(tmp_path / "test_agents"),
            tier="starter",
        )
        assert result["status"] == "success"
        assert result["infra_patterns_included"] == 5  # starter = 5 patterns

    @pytest.mark.asyncio
    async def test_generate_data_analyzer_project(self, tmp_path):
        from app.services.template_engine import generate_project
        result = await generate_project(
            template_id="data_analyzer",
            config={"project_name": "test_analyzer"},
            output_dir=str(tmp_path / "test_analyzer"),
            tier="pro",
        )
        assert result["status"] == "success"
        assert result["infra_patterns_included"] == 9  # pro = 9 patterns

    @pytest.mark.asyncio
    async def test_generate_invalid_template(self):
        from app.services.template_engine import generate_project
        result = await generate_project(
            template_id="nonexistent",
            config={"project_name": "test"},
        )
        assert "error" in result


# ═══════════════════════════════════════════════
# Ollama Service Tests
# ═══════════════════════════════════════════════

class TestOllamaService:
    """Test Ollama integration (requires running Ollama)."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        from app.services.ollama_service import ollama_service
        result = await ollama_service.health_check()
        assert result["status"] in ("healthy", "not_running")

    @pytest.mark.asyncio
    async def test_list_models(self):
        from app.services.ollama_service import ollama_service
        models = await ollama_service.list_models()
        assert isinstance(models, list)

    @pytest.mark.asyncio
    async def test_get_running_models(self):
        from app.services.ollama_service import ollama_service
        models = await ollama_service.get_running_models()
        assert isinstance(models, list)


# ═══════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════

class TestConfig:
    """Test configuration management."""

    def test_settings_loaded(self):
        from app.config import settings
        assert settings.APP_NAME == "ShipAI"
        assert settings.APP_VERSION == "0.1.0"
        assert settings.PORT == 8000

    def test_ollama_config(self):
        from app.config import settings
        assert "localhost" in settings.OLLAMA_BASE_URL
        assert settings.OLLAMA_TIMEOUT > 0

    def test_rate_limit_config(self):
        from app.config import settings
        assert settings.RATE_LIMIT_PER_MINUTE > 0
        assert settings.RATE_LIMIT_BURST > 0


# ═══════════════════════════════════════════════
# API Integration Tests (FastAPI TestClient)
# ═══════════════════════════════════════════════

class TestAPI:
    """Test API endpoints using FastAPI test client."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_root(self, client):
        resp = client.get("/api")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "ShipAI"
        assert "endpoints" in data

    def test_health(self, client):
        resp = client.get("/api/system/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_hardware(self, client):
        resp = client.get("/api/system/hardware")
        assert resp.status_code == 200
        data = resp.json()
        assert "cpu" in data
        assert "memory" in data
        assert "recommendation" in data

    def test_infra_patterns(self, client):
        resp = client.get("/api/system/infra-patterns?tier=pro")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "pro"
        assert len(data["patterns"]) == 9

    def test_list_models(self, client):
        resp = client.get("/api/models/")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "count" in data

    def test_advisor_patterns(self, client):
        resp = client.get("/api/advisor/patterns")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_patterns"] >= 16
        assert "rag" in data["categories"]

    def test_list_templates(self, client):
        resp = client.get("/api/templates/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3

    def test_get_template_detail(self, client):
        resp = client.get("/api/templates/rag_chatbot")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "RAG Chatbot"

    def test_license_tiers(self, client):
        resp = client.get("/api/license/tiers")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tiers"]) == 4

    def test_license_current(self, client):
        resp = client.get("/api/license/current")
        assert resp.status_code == 200
        data = resp.json()
        assert "tier" in data

    def test_license_generate_and_validate(self, client):
        # Generate
        import os
        headers = {"Authorization": "Bearer test_secret"}
        with patch.dict(os.environ, {"SHIPAI_ADMIN_SECRET": "test_secret"}):
            resp = client.post("/api/license/generate", headers=headers, json={
                "tier": "starter",
                "email": "test@api.com",
                "duration_days": 30,
            })
            assert resp.status_code == 200
            key = resp.json()["key"]
            assert key.startswith("SK-S-")

        # Validate
        resp = client.post("/api/license/validate", json={"key": key})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_valid"] is True
        assert data["tier"] == "starter"

    def test_generate_project_api(self, client, tmp_path):
        resp = client.post("/api/templates/generate", json={
            "template_id": "rag_chatbot",
            "config": {"project_name": "api_test_bot"},
            "output_dir": str(tmp_path / "api_test_bot"),
            "tier": "free",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["files_generated"] > 0

    def test_docs_endpoint(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
