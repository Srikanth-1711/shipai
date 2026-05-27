"""Sandboxed subprocess execution helpers.

Used for future isolated build/verification steps with strict safety defaults.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def run_sandboxed_command(
    command: List[str],
    *,
    cwd: str | Path | None = None,
    timeout_seconds: int = 30,
) -> SandboxResult:
    """Run command with bounded timeout and captured output.

    Safety constraints:
    - no shell=True
    - explicit cwd boundary
    - timeout enforced
    """
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
            check=False,
        )
        return SandboxResult(
            exit_code=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        return SandboxResult(
            exit_code=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"timeout after {timeout_seconds}s",
            timed_out=True,
        )


def create_temp_workspace(prefix: str = "shipai_sandbox_") -> str:
    return tempfile.mkdtemp(prefix=prefix)

