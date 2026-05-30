from __future__ import annotations

import json
from pathlib import Path

from app.engine.builder_utils import (
    build_manifest,
    compute_config_hash,
    deterministic_output_dir,
    safe_remove_output_dir,
    write_manifest,
)
from app.services.template_engine import ShipAITemplateEngine
from app.verification.project_verifier import verify_project
from app.verification.quality_metrics import score_verification


GOLDEN_ROOT = Path(__file__).resolve().parents[2] / "golden_projects"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_rel_files(root: Path) -> list[str]:
    files: list[str] = []
    for f in root.rglob("*"):
        if f.is_file() and "__pycache__" not in str(f):
            files.append(str(f.relative_to(root)).replace("\\", "/"))
    files.sort()
    return files


def _generate_project_into(tmp_path: Path, config: dict) -> Path:
    out_dir = deterministic_output_dir(config, base_dir=tmp_path)
    safe_remove_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ShipAITemplateEngine().generate_project(config, output_dir=str(out_dir))

    files = _collect_rel_files(out_dir)
    manifest = build_manifest(
        project_path=out_dir,
        config=config,
        config_hash=compute_config_hash(config),
        shipai_version="test",
        generated_files=files,
        model_plan_summary={},
    )
    write_manifest(out_dir, manifest)
    return out_dir


def _iter_golden_projects() -> list[Path]:
    if not GOLDEN_ROOT.exists():
        return []
    return sorted([p for p in GOLDEN_ROOT.iterdir() if p.is_dir()])


def test_golden_projects_exist():
    projs = _iter_golden_projects()
    assert len(projs) >= 1, "golden_projects/ is empty"


def test_golden_projects_regression_suite(tmp_path: Path):
    projs = _iter_golden_projects()
    assert projs, "no golden projects found"

    for proj in projs:
        expected_config = _load_json(proj / "expected_config.json")
        expected_quality = _load_json(proj / "expected_quality_threshold.json")
        expected_manifest = _load_json(proj / "expected_manifest_snapshot.json")

        out_dir = _generate_project_into(tmp_path, expected_config)

        # Deterministic output dir name
        assert out_dir.name.startswith(expected_config["template"] + "_")

        # Verification + smoke + metrics
        report = verify_project(out_dir)
        d = report.to_dict()
        metrics = score_verification(d)

        if expected_quality.get("require_smoke_test", True):
            assert d.get("smoke_test_valid") is True, f"{proj.name}: smoke test failed: {d.get('errors')}"

        assert (
            metrics["quality_score"] >= float(expected_quality["min_quality_score"])
        ), f"{proj.name}: quality_score regression {metrics['quality_score']} < {expected_quality['min_quality_score']}"
        # Snapshot assertions: config snapshot matches and required files exist
        assert expected_manifest.get("manifest_version") == "1.0"
        assert expected_manifest.get("config_snapshot") == expected_config

        present = set(_collect_rel_files(out_dir))
        for required in expected_manifest.get("generated_files_present", []):
            assert required in present, f"{proj.name}: missing expected file {required}"
