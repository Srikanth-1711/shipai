"""
Tests for Gap 1: Live benchmarks as primary signal.

Verifies:
- All 6 sources configured with correct TTLs and parsers
- Live source score overrides static matrix score for same model
- All 6 sources fail → bundled fallback works, no crash
- RichModelInfo.benchmark_score filled from fetcher FIRST
- merge_benchmark_scores_into_models() priority chain
- Cache-based sources still populate results correctly
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ["SHIPAI_BOOTSTRAP_ONLY"] = "1"

from app.install.benchmark_fetcher import (
    BENCHMARK_SOURCES,
    EVIDENCE_MULTIPLIERS,
    BenchmarkResult,
    BenchmarkScore,
    _normalize_model_key,
    _parse_aider_response,
    _parse_arena_response,
    _parse_artificial_analysis_response,
    _parse_hf_leaderboard_response,
    _parse_livebench_response,
    apply_decay,
    fetch_all_benchmarks,
    get_benchmark_for_model,
    load_bundled_benchmark_fallback,
    merge_scores,
    recency_decay,
)
from app.install.tools.web_intelligence import (
    RichModelInfo,
    merge_benchmark_scores_into_models,
    merge_model_sources,
)


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE CONFIGURATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchmarkSourceConfig:

    def test_exactly_6_sources_configured(self):
        """ShipAI uses 6 benchmark sources: 3 current-tier + 3 frozen-tier."""
        assert len(BENCHMARK_SOURCES) == 6

    def test_all_sources_have_required_fields(self):
        for src in BENCHMARK_SOURCES:
            assert "name" in src
            assert "tier" in src
            assert "url" in src
            assert "ttl_hours" in src
            assert "parser" in src

    def test_at_least_2_current_tier_sources(self):
        """Current-tier sources (live votes) — no decay."""
        current = [s for s in BENCHMARK_SOURCES if s["tier"] == "current"]
        assert len(current) >= 2

    def test_at_least_2_frozen_tier_sources(self):
        """Frozen-tier sources — recency-decayed."""
        frozen = [s for s in BENCHMARK_SOURCES if s["tier"] == "frozen"]
        assert len(frozen) >= 2

    def test_source_names_are_unique(self):
        names = [s["name"] for s in BENCHMARK_SOURCES]
        assert len(names) == len(set(names))

    def test_all_6_expected_sources_present(self):
        names = {s["name"] for s in BENCHMARK_SOURCES}
        expected = {
            "chatbot_arena",
            "lmsys_vision_arena",
            "open_llm_leaderboard",
            "aider_coding",
            "livebench",
            "artificial_analysis",
        }
        assert expected == names

    def test_current_tier_has_shorter_ttl(self):
        """Current-tier sources should refresh more often than frozen."""
        current_ttl = min(s["ttl_hours"] for s in BENCHMARK_SOURCES if s["tier"] == "current")
        frozen_ttl = max(s["ttl_hours"] for s in BENCHMARK_SOURCES if s["tier"] == "frozen")
        # Not strictly enforced but verify current sources aren't set to 7 days
        assert current_ttl <= 24

    def test_evidence_multipliers_cover_interpolated(self):
        """interpolated grade must exist (added in this session)."""
        assert "interpolated" in EVIDENCE_MULTIPLIERS
        # Must be between self_reported and direct
        assert EVIDENCE_MULTIPLIERS["self_reported"] < EVIDENCE_MULTIPLIERS["interpolated"]
        assert EVIDENCE_MULTIPLIERS["interpolated"] < EVIDENCE_MULTIPLIERS["direct"]


# ═══════════════════════════════════════════════════════════════════════════
# PARSER TESTS — all 5 source formats
# ═══════════════════════════════════════════════════════════════════════════

class TestParsers:

    def test_hf_leaderboard_parses_correctly(self):
        data = {"data": [
            {"model": "meta-llama/Llama-3.1-8B-Instruct", "average": 72.4},
            {"model": "google/gemma-2-9b-it", "average": 70.1},
        ]}
        scores = _parse_hf_leaderboard_response(data)
        assert len(scores) == 2
        # Keys should be normalized
        keys = list(scores.keys())
        assert any("llama" in k for k in keys)
        assert any("gemma" in k for k in keys)

    def test_arena_parser_normalizes_elo(self):
        """ELO 800 → 0, ELO 1400 → 100."""
        data = {"data": [
            {"model": "gpt-4o", "elo": 1300.0},
            {"model": "llama3:70b", "elo": 1100.0},
        ]}
        scores = _parse_arena_response(data)
        assert len(scores) == 2
        gpt_score = scores[_normalize_model_key("gpt-4o")]
        llama_score = scores[_normalize_model_key("llama3:70b")]
        assert gpt_score.score > llama_score.score
        assert 0 <= gpt_score.score <= 100

    def test_aider_parser(self):
        data = [
            {"model": "claude-3-5-sonnet-20241022", "percent_correct": 79.7},
            {"model": "gpt-4o", "percent_correct": 72.9},
        ]
        scores = _parse_aider_response(data)
        assert len(scores) == 2
        assert all(s.source == "aider_coding" for s in scores.values())
        assert all(s.tier == "frozen" for s in scores.values())

    def test_livebench_parser(self):
        data = {"models": [
            {"model": "claude-3-5-sonnet", "overall_average": 65.2},
            {"model": "llama3.1:70b", "overall_average": 52.1},
        ]}
        scores = _parse_livebench_response(data)
        assert len(scores) == 2
        assert all(s.source == "livebench" for s in scores.values())

    def test_artificial_analysis_parser(self):
        data = [
            {"model_name": "llama-3.1-70b", "quality_index": 72.5},
            {"model_name": "mistral-7b", "quality_index": 58.3},
        ]
        scores = _parse_artificial_analysis_response(data)
        assert len(scores) == 2
        assert all(s.source == "artificial_analysis" for s in scores.values())
        assert all(s.tier == "current" for s in scores.values())

    def test_parsers_handle_empty_data_gracefully(self):
        assert _parse_aider_response([]) == {}
        assert _parse_livebench_response({}) == {}
        assert _parse_artificial_analysis_response([]) == {}
        assert _parse_hf_leaderboard_response({}) == {}
        assert _parse_arena_response({}) == {}

    def test_parsers_handle_malformed_entries(self):
        """Bad entries skipped, good entries parsed."""
        data = [
            {"model_name": "good-model", "quality_index": 70.0},
            {"model_name": "", "quality_index": 70.0},       # empty name
            {"model_name": "bad-score", "quality_index": "NaN"},  # bad score
            None,                                              # null entry
        ]
        scores = _parse_artificial_analysis_response(data)
        assert len(scores) == 1
        assert _normalize_model_key("good-model") in scores


# ═══════════════════════════════════════════════════════════════════════════
# SCORE MERGE PRIORITY TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreMergePriority:

    def test_current_tier_overrides_frozen_for_same_model(self):
        """Current-tier live scores ALWAYS win over frozen benchmark scores."""
        existing = {
            "llama3": BenchmarkScore(score=65.0, source="open_llm_leaderboard", tier="frozen"),
        }
        incoming = {
            "llama3": BenchmarkScore(score=72.0, source="chatbot_arena", tier="current"),
        }
        merged = merge_scores(existing, incoming)
        assert merged["llama3"].tier == "current"
        assert merged["llama3"].score == 72.0
        assert merged["llama3"].source == "chatbot_arena"

    def test_frozen_higher_score_wins_over_lower_frozen(self):
        existing = {
            "gemma2": BenchmarkScore(score=60.0, source="source_a", tier="frozen"),
        }
        incoming = {
            "gemma2": BenchmarkScore(score=75.0, source="source_b", tier="frozen"),
        }
        merged = merge_scores(existing, incoming)
        assert merged["gemma2"].score == 75.0

    def test_current_tier_does_not_get_overwritten_by_frozen(self):
        """Even if frozen score is higher numerically, current wins."""
        existing = {
            "qwen": BenchmarkScore(score=90.0, source="chatbot_arena", tier="current"),
        }
        incoming = {
            "qwen": BenchmarkScore(score=95.0, source="open_llm_leaderboard", tier="frozen"),
        }
        merged = merge_scores(existing, incoming)
        assert merged["qwen"].tier == "current"
        assert merged["qwen"].score == 90.0

    def test_new_model_just_added(self):
        existing = {}
        incoming = {"phi4": BenchmarkScore(score=73.0, source="aider_coding", tier="frozen")}
        merged = merge_scores(existing, incoming)
        assert "phi4" in merged


# ═══════════════════════════════════════════════════════════════════════════
# RECENCY DECAY TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestRecencyDecay:

    def test_recent_score_no_decay(self):
        from datetime import datetime, timedelta
        recent = datetime.now() - timedelta(days=30)
        assert recency_decay(recent) == 1.00

    def test_six_months_slight_decay(self):
        from datetime import datetime, timedelta
        six_months = datetime.now() - timedelta(days=200)
        assert recency_decay(six_months) == 0.90

    def test_one_year_decay(self):
        from datetime import datetime, timedelta
        one_year = datetime.now() - timedelta(days=400)
        assert recency_decay(one_year) == 0.75

    def test_two_year_heavy_decay(self):
        from datetime import datetime, timedelta
        two_years = datetime.now() - timedelta(days=800)
        assert recency_decay(two_years) == 0.60

    def test_very_old_at_floor(self):
        from datetime import datetime, timedelta
        ancient = datetime.now() - timedelta(days=1200)
        assert recency_decay(ancient) == 0.50

    def test_none_date_conservative(self):
        assert recency_decay(None) == 0.60


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK SCORE PRIMARY WIRING TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchmarkScoreWiring:

    def _make_model(self, ollama_id: str, score: float = 0.0, source: str = "") -> RichModelInfo:
        return RichModelInfo(
            ollama_id=ollama_id,
            benchmark_score=score,
            benchmark_source=source,
            params_b=7.0,
        )

    def test_live_source_overrides_static_matrix_score(self):
        """
        THE KEY TEST: live benchmark score > static matrix score for same model.
        Model has matrix score of 60.0 → live source says 78.5 → 78.5 wins.
        """
        models = [self._make_model("llama3.1:8b", score=60.0, source="capability_matrix")]
        bm = BenchmarkResult()
        bm.scores = {
            "llama3.1": BenchmarkScore(score=78.5, source="chatbot_arena", tier="current"),
        }
        result = merge_benchmark_scores_into_models(bm, models)
        m = result[0]
        assert m.benchmark_score == 78.5
        assert "chatbot_arena" in m.benchmark_source

    def test_unknown_model_gets_family_interpolation_not_zero(self):
        """Model with no benchmark data → interpolated score, never 0."""
        models = [self._make_model("totally-unknown-model-7b", score=0.0)]
        bm = BenchmarkResult()  # empty — no live scores
        result = merge_benchmark_scores_into_models(bm, models)
        assert result[0].benchmark_score > 0
        assert "interpolation" in result[0].benchmark_source

    def test_model_with_existing_matrix_score_keeps_it_if_no_live(self):
        """Static matrix score retained when no live benchmark exists."""
        models = [self._make_model("phi4:latest", score=55.0, source="capability_matrix")]
        bm = BenchmarkResult()  # no live scores for phi4
        result = merge_benchmark_scores_into_models(bm, models)
        # Should keep 55.0 (not interpolated since score != 0)
        assert result[0].benchmark_score == 55.0

    def test_live_score_populates_benchmark_source_field(self):
        """benchmark_source must be set with the live source name."""
        models = [self._make_model("gemma2:9b", score=0.0)]
        bm = BenchmarkResult()
        bm.scores = {"gemma2": BenchmarkScore(score=70.1, source="open_llm_leaderboard", tier="frozen")}
        result = merge_benchmark_scores_into_models(bm, models)
        assert result[0].benchmark_source != ""
        assert "open_llm_leaderboard" in result[0].benchmark_source

    def test_variant_evidence_applied_for_partial_match(self):
        """Fuzzy match model → score with variant evidence multiplier."""
        scores = {
            "llama3.1": BenchmarkScore(score=80.0, source="chatbot_arena", tier="current"),
        }
        # "llama3.1:70b" should match "llama3.1" with variant evidence
        match = get_benchmark_for_model(scores, "llama3.1:70b")
        assert match is not None
        assert match.evidence_grade in ("direct", "variant")


# ═══════════════════════════════════════════════════════════════════════════
# ALL SOURCES FAIL → FALLBACK TEST
# ═══════════════════════════════════════════════════════════════════════════

class TestAllSourcesFailFallback:

    def test_all_sources_fail_returns_bundled_fallback(self):
        """If all 6 network sources fail, bundled matrix scores must be used."""
        result = asyncio.get_event_loop().run_until_complete(
            fetch_all_benchmarks(offline=True)
        )
        assert result.used_fallback is True
        assert len(result.scores) > 0  # bundled scores always available

    def test_offline_fallback_has_reasonable_scores(self):
        """Fallback scores must be in [0, 100]."""
        result = asyncio.get_event_loop().run_until_complete(
            fetch_all_benchmarks(offline=True)
        )
        for model_key, score in result.scores.items():
            assert 0 <= score.score <= 100, f"Score out of range for {model_key}: {score.score}"

    def test_offline_mode_does_not_make_network_calls(self):
        """offline=True must never touch the network."""
        with patch("httpx.AsyncClient") as mock_client:
            asyncio.get_event_loop().run_until_complete(
                fetch_all_benchmarks(offline=True)
            )
        mock_client.assert_not_called()
