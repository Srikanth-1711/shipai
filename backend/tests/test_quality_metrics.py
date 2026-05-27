from __future__ import annotations

from app.verification.quality_metrics import score_verification


def test_quality_metrics_scoring():
    report = {
        "syntax_valid": True,
        "imports_valid": True,
        "structure_valid": True,
        "dependency_valid": False,
        "manifest_valid": True,
        "smoke_test_valid": True,
        "runtime_validation_score": 80.0,
        "errors": ["x"],
        "warnings": ["w1", "w2"],
        "semantic_warnings": ["s1"],
        "verified_files": ["a.py", "b.py"],
        "verification_duration_ms": 12,
    }
    m = score_verification(report)
    assert m["checks_total"] == 6
    assert m["checks_passed"] == 5
    assert m["static_score"] == round((5 / 6) * 100.0, 2)
    assert m["runtime_validation_score"] == 80.0
    assert m["quality_score"] == round(m["static_score"] * 0.7 + 80.0 * 0.3, 2)
    assert m["errors_count"] == 1
    assert m["warnings_count"] == 3
    assert m["verified_files_count"] == 2

