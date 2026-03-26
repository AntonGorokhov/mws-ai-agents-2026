"""Safe code execution via subprocess."""

import shutil
import subprocess
import logging
from pathlib import Path

from src.config import BLOCKED_PATTERNS, CODE_TIMEOUT_S

PYTHON = shutil.which("python3") or shutil.which("python") or "python3"

logger = logging.getLogger(__name__)


def run_sandboxed(code: str, cwd: str, timeout: int = CODE_TIMEOUT_S) -> dict:
    """Execute generated Python code in a subprocess.

    Returns {"ok": bool, "stdout": str, "stderr": str, "error": str | None}.
    """
    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            msg = f"Blocked pattern: {pattern}"
            logger.warning(msg)
            return {"ok": False, "stdout": "", "stderr": "", "error": msg}

    code_dir = Path(cwd) / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    script_path = code_dir / "agent_script.py"
    script_path.write_text(code)

    try:
        result = subprocess.run(
            [PYTHON, str(script_path)],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout[-5000:] if result.stdout else "",
            "stderr": result.stderr[-5000:] if result.stderr else "",
            "error": None if result.returncode == 0 else f"Exit code {result.returncode}",
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": "", "error": f"Timeout ({timeout}s)"}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": "", "error": str(e)}
