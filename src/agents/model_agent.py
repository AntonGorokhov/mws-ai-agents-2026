"""Model Agent: trains models, tunes hyperparams, produces submission."""

import logging
from pathlib import Path

from src.agents.base import run_agent
from src.graph.state import PipelineState

logger = logging.getLogger(__name__)


def model_agent(state: PipelineState) -> PipelineState:
    """Model Agent node: train model and generate predictions."""
    workspace = state["workspace_dir"]
    competition_dir = state["competition_dir"]
    target_column = state["target_column"]
    metric_name = state.get("metric_name", "rmse")
    iteration = state.get("iteration", 0)

    cleaned_train = state.get("cleaned_data_path", str(Path(workspace) / "cleaned_train.csv"))
    cleaned_test = str(Path(workspace) / "cleaned_test.csv")

    # If cleaned data doesn't exist, fall back to raw data
    if not Path(cleaned_train).exists():
        cleaned_train = str(Path(competition_dir) / "train.csv")
        cleaned_test = str(Path(competition_dir) / "test.csv")
        logger.warning("[model_agent] Using raw data (cleaned not found)")

    rag_q = f"model selection regression {metric_name} hyperparameter tuning ensemble"
    if iteration > 0:
        prev_scores = state.get("cv_scores", [])
        rag_q += f" improve score from {prev_scores}"

    system_prompt = (
        "You are an expert ML engineer. Train models for a regression task.\n\n"
        "DOMAIN CONTEXT:\n"
        "NYC Airbnb availability prediction. Target is integer 0–365 (days available).\n"
        "~36% of targets are 0. Correlations are weak (max ~0.22). Features are already cleaned and numeric.\n\n"
        "Generate a COMPLETE Python script that:\n"
        "1. Reads cleaned train/test CSVs. Splits X and y using TARGET_COLUMN.\n"
        "2. Fills any remaining NaN with median (numeric) or 0.\n"
        "3. Drops any remaining non-numeric columns.\n"
        "4. Trains THREE models with 5-fold CV and early stopping:\n"
        "   a) LightGBM (num_leaves=63, lr=0.05, n_estimators=2000, early_stopping=50)\n"
        "   b) XGBoost (max_depth=6, lr=0.05, n_estimators=2000, early_stopping=50)\n"
        "   c) CatBoost (depth=6, lr=0.05, iterations=2000, early_stopping=50, verbose=0)\n"
        "5. For each model, collects out-of-fold predictions on train and full predictions on test.\n"
        "6. Ensembles: weighted average of test predictions. Weights = 1/cv_rmse for each model.\n"
        "7. CLIPS final predictions to [0, 365] — this is critical!\n"
        "8. Saves submission.csv (columns: 'index', 'prediction') and model.pkl.\n"
        "9. Prints per-fold RMSE and final ensemble RMSE.\n\n"
        "CRITICAL RULES:\n"
        "- CLIP all predictions to [0, 365] before saving.\n"
        "- Use np.clip(predictions, 0, 365).\n"
        "- submission.csv: 'index' = 0-based row index, 'prediction' = float.\n"
        "- Print 'RMSE: <value>' for each model and the ensemble.\n"
        "- For CatBoost: use CatBoostRegressor with verbose=0, no cat_features (data is already encoded).\n"
        "- Write FLAT script code — NO function definitions, NO if __name__ blocks. Just linear code.\n"
        "- Import: pandas, numpy, lightgbm, xgboost, catboost, sklearn, pickle.\n"
    )

    user_prompt = (
        "The following variables are ALREADY DEFINED at the top of the script (injected automatically):\n"
        "  TRAIN_PATH, TEST_PATH, SUBMISSION_PATH, MODEL_PATH, TARGET_COLUMN\n\n"
        "Use these variables directly — do NOT redefine them or hardcode paths.\n"
        "Example: train = pd.read_csv(TRAIN_PATH)\n\n"
        "Generate the Python script. Return ONLY the code inside ```python``` fences.\n"
        "Do NOT define TRAIN_PATH, TEST_PATH, SUBMISSION_PATH, MODEL_PATH, or TARGET_COLUMN — they already exist."
    )

    submission_path = str(Path(workspace) / "submission.csv")
    model_save_path = str(Path(workspace) / "model.pkl")

    # Hardcoded prefix — LLM cannot override these
    code_prefix = (
        f"# === HARDCODED PATHS (do not modify) ===\n"
        f"TRAIN_PATH = '{cleaned_train}'\n"
        f"TEST_PATH = '{cleaned_test}'\n"
        f"SUBMISSION_PATH = '{submission_path}'\n"
        f"MODEL_PATH = '{model_save_path}'\n"
        f"TARGET_COLUMN = '{target_column}'\n"
    )

    agent_result = run_agent(
        agent_name="model_agent",
        state=state,
        rag_question=rag_q,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        code_prefix=code_prefix,
    )

    new_state = {**state}
    new_state["training_code"] = agent_result["code"]
    new_state["agent_logs"] = state.get("agent_logs", []) + [agent_result["log_entry"]]

    if agent_result["result"]["ok"]:
        stdout = agent_result["result"].get("stdout", "")
        submission_path = str(Path(workspace) / "submission.csv")
        model_path = str(Path(workspace) / "model.pkl")

        new_state["phase"] = "model_trained"
        new_state["submission_path"] = submission_path if Path(submission_path).exists() else ""
        new_state["model_path"] = model_path if Path(model_path).exists() else ""

        # Try to parse CV scores from stdout
        cv_scores = _parse_cv_scores(stdout)
        if cv_scores:
            new_state["cv_scores"] = cv_scores

        logger.info("[model_agent] Training successful. CV scores: %s", cv_scores)
    else:
        error_msg = agent_result["result"].get("stderr", "")[:500]
        new_state["errors"] = state.get("errors", []) + [f"model_agent failed: {error_msg}"]
        new_state["phase"] = "model_trained"  # proceed to eval
        logger.error("[model_agent] Failed: %s", error_msg)

    return new_state


def _parse_cv_scores(stdout: str) -> list[float]:
    """Try to extract CV scores from model output."""
    import re
    scores = []
    # Look for patterns like "RMSE: 0.1234" or "CV score: 0.5678"
    patterns = [
        r"(?:rmse|RMSE|cv|CV)[:\s]+(\d+\.?\d*)",
        r"(?:score|Score)[:\s]+(\d+\.?\d*)",
        r"(?:mean|Mean)[:\s]+(\d+\.?\d*)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, stdout)
        for m in matches:
            try:
                scores.append(float(m))
            except ValueError:
                pass
    return scores[:5]  # cap at 5
