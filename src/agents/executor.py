"""Executor: deterministic code runner. Injects paths, strips redefinitions, runs code."""

import re
import logging
import time
from pathlib import Path

from src.safety.sandbox import run_sandboxed
from src.safety.guardrails import extract_python_code, sanitize_llm_output
from src.utils.llm import call_llm
from src.config import MODELS, MAX_CODE_FAILURES
from src.graph.state import PipelineState

logger = logging.getLogger(__name__)


def _build_prefix(state: dict) -> str:
    """Build the hardcoded variable prefix."""
    workspace = state["workspace_dir"]
    competition_dir = state["competition_dir"]
    target_column = state["target_column"]

    return (
        "# === INJECTED BY EXECUTOR (do not modify) ===\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        f"TRAIN_INPUT = '{Path(competition_dir) / 'train.csv'}'\n"
        f"TEST_INPUT = '{Path(competition_dir) / 'test.csv'}'\n"
        f"TRAIN_OUTPUT = '{Path(workspace) / 'cleaned_train.csv'}'\n"
        f"TEST_OUTPUT = '{Path(workspace) / 'cleaned_test.csv'}'\n"
        f"SUBMISSION_PATH = '{Path(workspace) / 'submission.csv'}'\n"
        f"MODEL_PATH = '{Path(workspace) / 'model.pkl'}'\n"
        f"TARGET_COLUMN = '{target_column}'\n"
        "\n"
        "# Safety helper: drop non-numeric columns before model training\n"
        "def _safe_drop_non_numeric(df):\n"
        "    return df.select_dtypes(exclude=['object', 'datetime64', 'datetime64[ns]'])\n"
    )


def _strip_redefinitions(code: str, prefix_vars: list[str]) -> str:
    """Remove lines that redefine our injected variables."""
    cleaned = []
    for line in code.split("\n"):
        stripped = line.strip()
        skip = False
        for var in prefix_vars:
            if re.match(rf"^{var}\s*=", stripped):
                skip = True
                break
        if not skip:
            cleaned.append(line)
    return "\n".join(cleaned)


def _auto_fix(code: str) -> str:
    """Fix common LLM code generation mistakes."""
    # Fix deprecated squared=False in mean_squared_error
    code = re.sub(r"mean_squared_error\(([^)]+),\s*squared\s*=\s*False\)", r"np.sqrt(mean_squared_error(\1))", code)
    # Fix LGBMRegressor.fit() verbose argument (not supported, use log_evaluation callback)
    code = re.sub(r"(lgb\.LGBMRegressor[^)]*\.fit\([^)]*),\s*verbose\s*=\s*\d+", r"\1", code)
    code = re.sub(r"(lgb\.LGBMClassifier[^)]*\.fit\([^)]*),\s*verbose\s*=\s*\d+", r"\1", code)
    # Also fix if verbose is passed as first-ish kwarg
    code = re.sub(r"\.fit\(([^)]*?)verbose\s*=\s*(?:0|False|-1)\s*,?\s*", r".fit(\1", code)
    return code


INJECTED_VARS = [
    "TRAIN_INPUT", "TEST_INPUT", "TRAIN_OUTPUT", "TEST_OUTPUT",
    "SUBMISSION_PATH", "MODEL_PATH", "TARGET_COLUMN",
]


def executor_agent(state: PipelineState) -> PipelineState:
    """Execute generated code with injected paths and retry logic."""
    workspace = state["workspace_dir"]
    code = state.get("generated_code", "")

    if not code:
        logger.error("[executor] No code to execute")
        new_state = {**state}
        new_state["errors"] = state.get("errors", []) + ["executor: no code provided"]
        new_state["phase"] = "executed"
        return new_state

    # Build prefix and prepare code
    prefix = _build_prefix(state)
    code = _strip_redefinitions(code, INJECTED_VARS)
    code = _auto_fix(code)
    full_code = prefix + "\n" + code

    # Ensure workspace dirs exist
    Path(workspace).mkdir(parents=True, exist_ok=True)
    (Path(workspace) / "code").mkdir(exist_ok=True)

    # Execute with retries
    result = None
    start = time.time()

    for attempt in range(MAX_CODE_FAILURES):
        result = run_sandboxed(full_code, cwd=workspace)
        if result["ok"]:
            break

        stderr = result.get("stderr", "")[-1500:]
        logger.warning("[executor] Attempt %d/%d failed: %s",
                       attempt + 1, MAX_CODE_FAILURES, stderr[:200])

        if attempt < MAX_CODE_FAILURES - 1:
            # Ask coder to fix
            fix_result = call_llm(
                model=MODELS["coder"],
                system=(
                    "Fix this Python script. The error is shown below.\n"
                    "Output ONLY the corrected Python code in ```python``` fences.\n"
                    "Write FLAT code, no functions. Variables TRAIN_INPUT, TEST_INPUT, "
                    "SUBMISSION_PATH, MODEL_PATH, TARGET_COLUMN are pre-defined — do NOT redefine them."
                ),
                user=f"Error:\n{stderr}\n\nOriginal code:\n{code}",
                fallback=MODELS["coder_fallback"],
            )
            raw = sanitize_llm_output(fix_result["content"])
            code = extract_python_code(raw)
            code = _strip_redefinitions(code, INJECTED_VARS)
            full_code = prefix + "\n" + code

    duration = time.time() - start

    # Build result state
    new_state = {**state}
    new_state["phase"] = "executed"
    new_state["agent_logs"] = state.get("agent_logs", []) + [{
        "agent": "executor",
        "duration_s": round(duration, 2),
        "result": "success" if result["ok"] else "failure",
    }]

    if result["ok"]:
        stdout = result.get("stdout", "")
        new_state["execution_stdout"] = stdout[-3000:]

        # Check outputs
        sub_path = str(Path(workspace) / "submission.csv")
        if Path(sub_path).exists():
            new_state["submission_path"] = sub_path

        model_path = str(Path(workspace) / "model.pkl")
        if Path(model_path).exists():
            new_state["model_path"] = model_path

        # Parse CV scores
        new_state["cv_scores"] = _parse_scores(stdout)
        logger.info("[executor] Success. CV scores: %s", new_state["cv_scores"])
    else:
        new_state["errors"] = state.get("errors", []) + [
            f"executor failed: {result.get('stderr', '')[:300]}"
        ]
        logger.error("[executor] All attempts failed")

    return new_state


def _parse_scores(stdout: str) -> list[float]:
    import re
    scores = []
    for match in re.finditer(r"RMSE[:\s]+(\d+\.?\d*)", stdout, re.IGNORECASE):
        try:
            scores.append(float(match.group(1)))
        except ValueError:
            pass
    return scores[:10]
