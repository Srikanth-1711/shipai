"""Phase 13 — controlled semantic / smoke validation for generated projects.

Verifies projects are minimally runnable WITHOUT executing arbitrary user
business logic, importing generated modules dynamically, or installing deps.

Future plugins: Kubernetes, Terraform, Helm, CI workflows, MCP servers.
"""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from app.runtime.sandbox import run_sandboxed_command


class RealityCheckPlugin(Protocol):
    name: str

    def run(self, root: Path, config: Dict[str, Any]) -> "RealityCheckResult":
        ...


@dataclass
class RealityCheckResult:
    name: str
    passed: bool
    score: float = 100.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RealityTestReport:
    smoke_test_valid: bool = True
    runtime_validation_score: float = 100.0
    semantic_warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    checks: List[RealityCheckResult] = field(default_factory=list)
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _find_main_py(root: Path) -> Optional[Path]:
    candidates = [
        root / "backend" / "app" / "main.py",
        root / "app" / "main.py",
        root / "main.py",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _ast_has_fastapi_app(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "FastAPI":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "FastAPI":
                return True
    return False


def _ast_has_routes(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Attribute) and dec.attr in (
                    "get",
                    "post",
                    "put",
                    "delete",
                    "patch",
                ):
                    return True
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                    if dec.func.attr in ("get", "post", "put", "delete", "patch"):
                        return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "include_router":
                return True
    return False


def check_python_fastapi(root: Path, config: Dict[str, Any]) -> RealityCheckResult:
    """Semantic FastAPI smoke checks via AST only (no module import)."""
    result = RealityCheckResult(name="python_fastapi", passed=True)
    main_py = _find_main_py(root)
    if not main_py:
        result.passed = False
        result.score = 0.0
        result.errors.append("FastAPI main.py not found under backend/app/")
        return result

    src = _read_text(main_py)
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        result.passed = False
        result.score = 0.0
        result.errors.append(f"main.py syntax error: {exc.msg}")
        return result

    if not _ast_has_fastapi_app(tree):
        result.passed = False
        result.score = 40.0
        result.errors.append("FastAPI app object not detected in main.py")

    if not _ast_has_routes(tree):
        result.warnings.append("No route decorators or include_router detected in main.py")
        result.score = min(result.score, 70.0)

    # Import graph stability: flag absolute imports to missing local modules (static).
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module.split(".")[0]
            if mod in ("app", "backend"):
                # Expect package layout under backend/app
                pkg_root = root / "backend" / "app"
                if not pkg_root.exists():
                    result.passed = False
                    result.errors.append("backend/app package missing for app imports")
                    break

    if result.errors:
        result.passed = False
    return result


def check_docker(root: Path, config: Dict[str, Any]) -> RealityCheckResult:
    result = RealityCheckResult(name="docker", passed=True)
    dockerfile = root / "Dockerfile"
    if not dockerfile.exists():
        result.warnings.append("Dockerfile missing; skipping docker smoke checks")
        result.score = 80.0
        return result

    content = _read_text(dockerfile)
    upper = content.upper()
    if "FROM " not in upper:
        result.passed = False
        result.score = 0.0
        result.errors.append("Dockerfile missing FROM instruction")
    if "WORKDIR" not in upper:
        result.warnings.append("Dockerfile missing WORKDIR")
        result.score = min(result.score, 80.0)
    if "CMD" not in upper and "ENTRYPOINT" not in upper:
        result.warnings.append("Dockerfile missing CMD/ENTRYPOINT")
        result.score = min(result.score, 85.0)

    compose = root / "docker-compose.yml"
    if compose.exists():
        text = _read_text(compose)
        if "services:" not in text:
            result.passed = False
            result.errors.append("docker-compose.yml missing services section")
            result.score = min(result.score, 50.0)
        else:
            try:
                import yaml  # type: ignore

                data = yaml.safe_load(text)
                if not isinstance(data, dict) or "services" not in data:
                    result.passed = False
                    result.errors.append("docker-compose.yml invalid structure")
            except ImportError:
                result.warnings.append("PyYAML not installed; compose structure check skipped")
            except Exception as exc:
                result.passed = False
                result.errors.append(f"docker-compose.yml parse error: {exc}")

    if result.errors:
        result.passed = False
    return result


def check_frontend(root: Path, config: Dict[str, Any]) -> RealityCheckResult:
    result = RealityCheckResult(name="frontend", passed=True)
    frontend = root / "frontend"
    if not frontend.exists():
        return result  # not applicable

    pkg = frontend / "package.json"
    if not pkg.exists():
        result.passed = False
        result.errors.append("frontend/ present but package.json missing")
        result.score = 0.0
        return result

    try:
        data = json.loads(_read_text(pkg))
        if not isinstance(data, dict):
            raise ValueError("package.json root must be object")
        if "name" not in data and "dependencies" not in data:
            result.warnings.append("package.json missing name/dependencies")
            result.score = 70.0
    except Exception as exc:
        result.passed = False
        result.score = 0.0
        result.errors.append(f"invalid package.json: {exc}")

    for cfg_name in ("tsconfig.json", "vite.config.ts", "vite.config.js"):
        cfg = frontend / cfg_name
        if not cfg.exists():
            continue
        text = _read_text(cfg)
        if cfg_name.endswith(".json"):
            try:
                json.loads(text)
            except Exception as exc:
                result.passed = False
                result.errors.append(f"invalid {cfg_name}: {exc}")
        else:
            try:
                ast.parse(text)  # TS/JS syntax-ish; best effort for .ts as JS subset
            except SyntaxError as exc:
                result.warnings.append(f"{cfg_name} parse warning: {exc.msg}")

    if result.errors:
        result.passed = False
    return result


def check_openapi(root: Path, config: Dict[str, Any]) -> RealityCheckResult:
    result = RealityCheckResult(name="openapi", passed=True)
    candidates = list(root.rglob("openapi.json")) + list(root.rglob("openapi.yaml"))
    if not candidates:
        return result

    for spec in candidates[:3]:
        text = _read_text(spec)
        if spec.suffix == ".json":
            try:
                data = json.loads(text)
                if "openapi" not in data and "swagger" not in data:
                    result.warnings.append(f"{spec.name} missing openapi/swagger version field")
            except Exception as exc:
                result.passed = False
                result.errors.append(f"invalid OpenAPI JSON {spec.name}: {exc}")
        else:
            if "openapi:" not in text and "swagger:" not in text:
                result.warnings.append(f"{spec.name} missing openapi/swagger version key")

    if result.errors:
        result.passed = False
        result.score = 50.0
    return result


def check_sandbox_compile(root: Path, config: Dict[str, Any]) -> RealityCheckResult:
    """Run bounded py_compile in sandbox on main.py only (no imports)."""
    result = RealityCheckResult(name="sandbox_compile", passed=True)
    main_py = _find_main_py(root)
    if not main_py:
        result.warnings.append("sandbox compile skipped: main.py not found")
        return result

    cmd = ["python", "-m", "py_compile", str(main_py.name)]
    sandbox = run_sandboxed_command(cmd, cwd=main_py.parent, timeout_seconds=10)
    if sandbox.timed_out:
        result.passed = False
        result.errors.append("sandbox py_compile timed out")
        result.score = 0.0
    elif sandbox.exit_code != 0:
        result.passed = False
        result.errors.append(f"sandbox py_compile failed: {sandbox.stderr.strip() or sandbox.stdout}")
        result.score = 0.0
    return result


DEFAULT_PLUGINS: List[Callable[[Path, Dict[str, Any]], RealityCheckResult]] = [
    check_python_fastapi,
    check_docker,
    check_frontend,
    check_openapi,
    check_sandbox_compile,
]


def run_reality_tests(
    project_path: str | Path,
    config: Dict[str, Any] | None = None,
    plugins: List[Callable[[Path, Dict[str, Any]], RealityCheckResult]] | None = None,
) -> RealityTestReport:
    start = time.perf_counter()
    root = Path(project_path)
    cfg = config or {}
    report = RealityTestReport()

    if not root.exists():
        report.smoke_test_valid = False
        report.runtime_validation_score = 0.0
        report.errors.append(f"project path missing: {root}")
        report.duration_ms = int((time.perf_counter() - start) * 1000)
        return report

    checks = plugins or DEFAULT_PLUGINS
    scores: List[float] = []

    for fn in checks:
        try:
            res = fn(root, cfg)
        except Exception as exc:
            res = RealityCheckResult(
                name=getattr(fn, "__name__", "plugin"),
                passed=False,
                score=0.0,
                errors=[f"plugin error: {exc}"],
            )
        report.checks.append(res)
        scores.append(res.score)
        report.errors.extend([f"{res.name}: {e}" for e in res.errors])
        report.semantic_warnings.extend([f"{res.name}: {w}" for w in res.warnings])
        if not res.passed:
            report.smoke_test_valid = False

    if scores:
        report.runtime_validation_score = round(sum(scores) / len(scores), 2)
    else:
        report.runtime_validation_score = 0.0

    if report.runtime_validation_score < 60.0:
        report.smoke_test_valid = False

    report.duration_ms = int((time.perf_counter() - start) * 1000)
    return report
