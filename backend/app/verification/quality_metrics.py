"""Quality scoring for generated projects based on verification reports."""

from __future__ import annotations

from typing import Any, Dict


def score_verification(report: Dict[str, Any]) -> Dict[str, Any]:
    """Compute deterministic quality metrics from verification report."""
    bool_fields = [
        "syntax_valid",
        "imports_valid",
        "structure_valid",
        "dependency_valid",
        "manifest_valid",
        "smoke_test_valid",
    ]
    passed = sum(1 for k in bool_fields if bool(report.get(k)))
    total = len(bool_fields)
    static_score = round((passed / total) * 100.0, 2) if total else 0.0
    runtime_score = float(report.get("runtime_validation_score", 0.0) or 0.0)
    # Weight static checks 70%, semantic smoke score 30%.
    quality_score = round((static_score * 0.7) + (runtime_score * 0.3), 2)

    return {
        "quality_score": quality_score,
        "static_score": static_score,
        "runtime_validation_score": runtime_score,
        "checks_passed": passed,
        "checks_total": total,
        "errors_count": len(report.get("errors", []) or []),
        "warnings_count": len(report.get("warnings", []) or [])
        + len(report.get("semantic_warnings", []) or []),
        "verified_files_count": len(report.get("verified_files", []) or []),
        "verification_duration_ms": int(report.get("verification_duration_ms", 0) or 0),
    }

