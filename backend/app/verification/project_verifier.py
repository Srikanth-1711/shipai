"""Static verification for generated projects (no execution).

Goal: before ShipAI considers a build "successful", verify the generated
artifact is structurally present, syntactically valid, requirements are parseable,
and manifest hashes match.
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import py_compile
import re

from app.engine.builder_utils import canonical_json, compute_config_hash


_REQ_LINE_RE = re.compile(r"^[A-Za-z0-9_.\\-]+(\[[A-Za-z0-9_,\\-]+\])?([<>=!~]=?.+)?$")


def _sha256_file(path: Path) -> str:
    from app.engine.builder_utils import sha256_file as _impl

    return _impl(path)


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _normalize_req_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _parse_requirements_txt(req_path: Path) -> Tuple[Dict[str, str], List[str]]:
    """Return (pins_by_pkg, errors). pin_by_pkg value is the raw version spec if present."""
    errors: List[str] = []
    pins: Dict[str, str] = {}
    if not req_path.exists():
        errors.append("requirements.txt missing")
        return pins, errors

    lines = req_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # strip env markers (best effort)
        main = line.split(";", 1)[0].strip()
        if not _REQ_LINE_RE.match(main):
            errors.append(f"requirements.txt line {i}: malformed requirement: {raw}")
            continue

        # Split on first version/operator token
        pkg = main
        version_spec = ""
        # Try common operators
        for op in ("==", ">=", "<=", "~=", "!=" , ">", "<"):
            if op in main:
                pkg, version_spec = main.split(op, 1)
                version_spec = op + version_spec.strip()
                break
        pkg = _normalize_req_name(pkg)

        if pkg in pins and pins[pkg] != version_spec:
            errors.append(
                f"requirements.txt conflict for '{pkg}': '{pins[pkg]}' vs '{version_spec}'"
            )
        pins[pkg] = version_spec

    return pins, errors


@dataclass
class VerificationReport:
    syntax_valid: bool = True
    imports_valid: bool = True
    structure_valid: bool = True
    dependency_valid: bool = True
    manifest_valid: bool = True
    smoke_test_valid: bool = True
    runtime_validation_score: float = 100.0
    semantic_warnings: List[str] = None  # type: ignore[assignment]
    errors: List[str] = None  # type: ignore[assignment]
    warnings: List[str] = None  # type: ignore[assignment]
    verified_files: List[str] = None  # type: ignore[assignment]
    verification_duration_ms: int = 0

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.verified_files is None:
            self.verified_files = []
        if self.semantic_warnings is None:
            self.semantic_warnings = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def failed(self) -> bool:
        return not (
            self.syntax_valid
            and self.structure_valid
            and self.dependency_valid
            and self.manifest_valid
            and self.imports_valid
            and self.smoke_test_valid
        )


def verify_project(project_path: str | Path) -> VerificationReport:
    start = time.perf_counter()
    root = Path(project_path)
    report = VerificationReport()

    if not root.exists() or not root.is_dir():
        report.structure_valid = False
        report.errors.append(f"project path does not exist: {root}")
        report.verification_duration_ms = int((time.perf_counter() - start) * 1000)
        return report

    # --- Structure verification ---
    required_files = [
        root / "manifest.json",
        root / "requirements.txt",
        root / "Dockerfile",
        root / "README.md",
    ]
    if not (root / "backend").exists():
        report.structure_valid = False
        report.errors.append("backend/ folder missing")

    for p in required_files:
        if not p.exists():
            report.structure_valid = False
            report.errors.append(f"required file missing: {p.name}")

    # --- Manifest verification ---
    manifest_path = root / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(_safe_read_text(manifest_path))
        except Exception as exc:
            report.manifest_valid = False
            report.errors.append(f"manifest.json parse failed: {exc}")
    else:
        report.manifest_valid = False

    config_snapshot = manifest.get("config_snapshot") if isinstance(manifest, dict) else None
    manifest_config_hash = manifest.get("config_hash") if isinstance(manifest, dict) else None
    if isinstance(config_snapshot, dict) and isinstance(manifest_config_hash, str):
        expected_hash = compute_config_hash(config_snapshot)
        if expected_hash != manifest_config_hash:
            report.manifest_valid = False
            report.errors.append("config_hash mismatch between manifest and config_snapshot")
    elif manifest:
        report.warnings.append("manifest missing config_snapshot/config_hash; skipping config_hash check")

    # Verify hashes of listed generated files
    generated_files = manifest.get("generated_files", {}) if isinstance(manifest, dict) else {}
    if isinstance(generated_files, dict) and generated_files:
        for rel_path, meta in generated_files.items():
            if not isinstance(rel_path, str):
                continue
            fp = root / rel_path
            if not fp.exists():
                report.manifest_valid = False
                report.errors.append(f"manifest lists missing file: {rel_path}")
                continue
            if not isinstance(meta, dict):
                continue
            expected_sha = meta.get("sha256")
            if isinstance(expected_sha, str):
                actual_sha = _sha256_file(fp)
                if actual_sha != expected_sha:
                    report.manifest_valid = False
                    report.errors.append(f"sha256 mismatch for {rel_path}")
    else:
        # If manifest exists but doesn't have file list, treat as invalid.
        if manifest_path.exists():
            report.manifest_valid = False
            report.errors.append("manifest.json missing generated_files mapping")

    # --- Dependency verification ---
    pins, req_errors = _parse_requirements_txt(root / "requirements.txt")
    if req_errors:
        report.dependency_valid = False
        report.errors.extend(req_errors)

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        # Best-effort structural check only (no dependency parsing).
        content = _safe_read_text(pyproject).lower()
        if "[project]" not in content and "[tool.poetry]" not in content:
            report.dependency_valid = False
            report.errors.append("pyproject.toml present but missing [project] or [tool.poetry] sections")

    # --- Syntax & imports verification ---
    py_files: List[Path] = []
    for f in root.rglob("*.py"):
        if "__pycache__" in str(f):
            continue
        py_files.append(f)

    for f in py_files:
        rel = str(f.relative_to(root))
        src = _safe_read_text(f)
        try:
            # AST parse catches syntax errors without executing.
            ast.parse(src)
            # py_compile catches encoding and compile-time errors.
            py_compile.compile(str(f), doraise=True)
            report.verified_files.append(rel)
        except SyntaxError as exc:
            report.syntax_valid = False
            report.errors.append(f"syntax error in {rel}: {exc.msg} (line {exc.lineno})")
            continue
        except Exception as exc:
            # Any compile-time error counts as syntax invalid.
            report.syntax_valid = False
            report.errors.append(f"compile error in {rel}: {exc}")
            continue

        # Imports validity (relative import resolution only).
        # We do not execute imports; we only ensure relative imports point to existing files.
        try:
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level and node.module is not None:
                    # Best-effort: build the target path within project.
                    # level==1 => one package up relative to current file package.
                    parts = (node.module or "").split(".")
                    # Resolve based on relative directory depth
                    pkg_parts = rel.replace("\\", "/").split("/")[:-1]  # drop filename
                    # node.level indicates how many levels to go up.
                    up = node.level - 1
                    base_parts = pkg_parts[: max(0, len(pkg_parts) - up)]
                    target_parts = base_parts + parts
                    # If target is a module file or package init exists, it's resolvable.
                    target_mod = root.joinpath(*target_parts).with_suffix(".py")
                    target_pkg = root.joinpath(*target_parts, "__init__.py")
                    if not (target_mod.exists() or target_pkg.exists()):
                        report.imports_valid = False
                        report.errors.append(f"relative import unresolved in {rel}: {node.module}")
        except Exception:
            # Don't block verification on import-analysis.
            report.warnings.append(f"import analysis skipped for {rel}")

    # --- Phase 13: controlled reality / smoke validation ---
    config_snapshot = manifest.get("config_snapshot") if isinstance(manifest, dict) else {}
    if not isinstance(config_snapshot, dict):
        config_snapshot = {}

    from app.verification.reality_testing import run_reality_tests

    reality = run_reality_tests(root, config_snapshot)
    report.smoke_test_valid = reality.smoke_test_valid
    report.runtime_validation_score = reality.runtime_validation_score
    report.semantic_warnings.extend(reality.semantic_warnings)
    for err in reality.errors:
        report.errors.append(f"reality: {err}")

    report.verification_duration_ms = int((time.perf_counter() - start) * 1000)
    return report

