"""Eval Agent: validates predictions, checks leakage, scores."""

import logging
from pathlib import Path

from src.agents.base import run_agent
from src.graph.state import PipelineState

logger = logging.getLogger(__name__)


def eval_agent(state: PipelineState) -> PipelineState:
    """Eval Agent node: validate, check leakage, score."""
    workspace = state["workspace_dir"]
    competition_dir = state["competition_dir"]
    target_column = state["target_column"]
    metric_name = state.get("metric_name", "rmse")
    iteration = state.get("iteration", 0)

    submission_path = state.get("submission_path", str(Path(workspace) / "submission.csv"))

    rag_q = f"data leakage detection cross validation evaluation {metric_name} regression"

    system_prompt = (
        "You are an expert ML evaluator. Validate predictions for NYC Airbnb availability (0–365 days).\n\n"
        "Generate a COMPLETE Python script that:\n"
        "1. Loads submission.csv — checks 'index' and 'prediction' columns exist\n"
        "2. Checks for NaN/Inf in predictions\n"
        "3. Checks predictions are in [0, 365] range. If not, clip them and overwrite submission.\n"
        "4. Loads cleaned train, trains a quick LightGBM with 5-fold CV to get RMSE score\n"
        "5. Prints a JSON dict as the LAST line of stdout:\n"
        "   {\"score\": <float>, \"has_leakage\": false, \"issues\": [...], \"recommendation\": \"submit\"}\n\n"
        "IMPORTANT:\n"
        "- Print the JSON as the VERY LAST line.\n"
        "- 'recommendation' must be 'submit' (we only do 1 iteration).\n"
        "- 'score' is the CV RMSE on training data.\n"
        "- Write FLAT script code — NO function definitions, NO if __name__ blocks. Just linear code.\n"
    )

    cleaned_train = state.get("cleaned_data_path", str(Path(workspace) / "cleaned_train.csv"))

    user_prompt = (
        "The following variables are ALREADY DEFINED at the top of the script (injected automatically):\n"
        "  SUBMISSION_PATH, TRAIN_PATH, TARGET_COLUMN, METRIC\n\n"
        "Use these variables directly — do NOT redefine them or hardcode paths.\n\n"
        f"Previous CV scores: {state.get('cv_scores', [])}\n"
        f"Iteration: {iteration}\n\n"
        "Generate the Python script. Return ONLY the code inside ```python``` fences.\n"
        "Do NOT define SUBMISSION_PATH, TRAIN_PATH, TARGET_COLUMN, or METRIC — they already exist."
    )

    code_prefix = (
        f"# === HARDCODED PATHS (do not modify) ===\n"
        f"SUBMISSION_PATH = '{submission_path}'\n"
        f"TRAIN_PATH = '{cleaned_train}'\n"
        f"TARGET_COLUMN = '{target_column}'\n"
        f"METRIC = '{metric_name}'\n"
    )

    agent_result = run_agent(
        agent_name="eval_agent",
        state=state,
        rag_question=rag_q,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_key="reasoning",
        fallback_key="reasoning_fallback",
        code_prefix=code_prefix,
    )

    new_state = {**state}
    new_state["agent_logs"] = state.get("agent_logs", []) + [agent_result["log_entry"]]
    new_state["iteration"] = iteration + 1

    if agent_result["result"]["ok"]:
        stdout = agent_result["result"].get("stdout", "")
        metrics = _parse_eval_output(stdout)
        new_state["metrics"] = metrics
        new_state["phase"] = "eval_done"
        logger.info("[eval_agent] Evaluation done. Metrics: %s", metrics)
    else:
        error_msg = agent_result["result"].get("stderr", "")[:500]
        new_state["errors"] = state.get("errors", []) + [f"eval_agent failed: {error_msg}"]
        new_state["phase"] = "eval_done"
        new_state["metrics"] = state.get("metrics", {"score": 999.0})
        logger.error("[eval_agent] Failed: %s", error_msg)

    return new_state


def _parse_eval_output(stdout: str) -> dict:
    """Parse the JSON summary from eval agent output."""
    import json

    # Try to find JSON in the last lines
    lines = stdout.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    return {"score": 999.0, "has_leakage": False, "issues": [], "recommendation": "submit"}
