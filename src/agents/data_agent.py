"""Data Agent: cleans data, creates features, produces cleaned CSV."""

import json
import logging
from pathlib import Path

import pandas as pd

from src.agents.base import run_agent
from src.graph.state import PipelineState

logger = logging.getLogger(__name__)


def _build_data_summary(competition_dir: str, target_column: str) -> dict:
    """Build a JSON summary of the dataset."""
    train = pd.read_csv(Path(competition_dir) / "train.csv")
    test = pd.read_csv(Path(competition_dir) / "test.csv")

    summary = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "columns": list(train.columns),
        "dtypes": {col: str(dtype) for col, dtype in train.dtypes.items()},
        "missing_pct": {col: round(train[col].isna().mean() * 100, 2) for col in train.columns if train[col].isna().any()},
        "target_stats": {
            "mean": round(float(train[target_column].mean()), 4),
            "median": round(float(train[target_column].median()), 4),
            "std": round(float(train[target_column].std()), 4),
            "min": round(float(train[target_column].min()), 4),
            "max": round(float(train[target_column].max()), 4),
        },
        "nunique": {col: int(train[col].nunique()) for col in train.columns},
        "sample_values": {col: train[col].dropna().head(3).tolist() for col in train.columns},
    }

    # Classify column types
    categoricals = [c for c in train.columns if train[c].dtype == "object"]
    numericals = [c for c in train.columns if train[c].dtype in ("int64", "float64") and c != target_column]
    datetimes = [c for c in train.columns if "dt" in c.lower() or "date" in c.lower()]

    summary["categoricals"] = categoricals
    summary["numericals"] = numericals
    summary["potential_datetimes"] = datetimes

    return summary


def data_agent(state: PipelineState) -> PipelineState:
    """Data Agent node: clean data and engineer features."""
    workspace = state["workspace_dir"]
    competition_dir = state["competition_dir"]
    target_column = state["target_column"]

    # Build data summary
    data_summary = _build_data_summary(competition_dir, target_column)
    column_types = f"categorical: {data_summary['categoricals']}, numerical: {data_summary['numericals']}, datetime: {data_summary['potential_datetimes']}"

    data_summary_json = json.dumps(data_summary, indent=2, default=str)

    # Determine RAG query based on iteration
    iteration = state.get("iteration", 0)
    if iteration == 0:
        rag_q = f"feature engineering {column_types} missing values handling"
    else:
        prev_errors = state.get("errors", [])
        rag_q = f"improve features iteration {iteration} {' '.join(prev_errors[-2:])}"

    train_path = str(Path(competition_dir) / "train.csv")
    test_path = str(Path(competition_dir) / "test.csv")
    out_train = str(Path(workspace) / "cleaned_train.csv")
    out_test = str(Path(workspace) / "cleaned_test.csv")

    system_prompt = (
        "You are an expert data scientist. Your task is to clean data and engineer features for a regression task.\n\n"
        "DOMAIN CONTEXT:\n"
        "This is NYC Airbnb data. Target is 'target' (integer 0-365, representing availability days).\n"
        "~36% of targets are 0. The column 'sum' is price per night. 'last_dt' is last review date.\n"
        "'location_cluster' is borough (5 values), 'location' is neighborhood (220 values).\n"
        "'type_house' is room type (3 values). Correlations with target are weak (max ~0.22).\n\n"
        "Generate a COMPLETE Python script that:\n"
        "1. Reads train.csv and test.csv\n"
        "2. Drops 'name' (too unique, 36K values) and '_id' (identifier)\n"
        "3. Drops 'host_name' (too many unique, not useful)\n"
        "4. Parses 'last_dt' as datetime → extract year, month, day_of_week, days_since_last_review\n"
        "5. Creates missing indicator: has_last_review = 1 if last_dt is not NaN, else 0\n"
        "6. Fills avg_reviews NaN with 0 (no reviews = 0 average)\n"
        "7. Encodes 'location_cluster' (5 values) with one-hot encoding\n"
        "8. Encodes 'type_house' (3 values) with one-hot encoding\n"
        "9. Encodes 'location' (220 neighborhoods) with frequency encoding from TRAIN only\n"
        "10. Creates interaction features:\n"
        "    - sum_per_min_days = sum / (min_days + 1)\n"
        "    - log_sum = log1p(sum)\n"
        "    - log_amt_reviews = log1p(amt_reviews)\n"
        "    - reviews_per_host = amt_reviews / (total_host + 1)\n"
        "    - is_entire_home (binary from type_house)\n"
        "11. Saves cleaned train (with target) and test to output paths\n"
        "12. Prints column list and shape to stdout\n\n"
        "CRITICAL RULES:\n"
        "- Write FLAT script code — NO function definitions, NO if __name__ blocks. Just linear code.\n"
        "- Use only pandas, numpy — no sklearn needed for this step.\n"
        "- Apply the SAME transformations to both train and test.\n"
        "- For frequency encoding: compute on train, map onto both train and test. Fill unseen with 0.\n"
        "- Do NOT drop target from cleaned_train.\n"
        "- All output columns must be numeric.\n"
    )

    user_prompt = (
        "The following variables are ALREADY DEFINED at the top of the script (injected automatically):\n"
        "  TRAIN_INPUT, TEST_INPUT, TRAIN_OUTPUT, TEST_OUTPUT, TARGET_COLUMN\n\n"
        "Use these variables directly — do NOT redefine them or hardcode paths.\n"
        "Example: train = pd.read_csv(TRAIN_INPUT)\n\n"
        "Generate the Python script. Return ONLY the code inside ```python``` fences.\n"
        "Do NOT define TRAIN_INPUT, TEST_INPUT, TRAIN_OUTPUT, TEST_OUTPUT, or TARGET_COLUMN — they already exist."
    )

    # Hardcoded prefix — LLM cannot override these
    code_prefix = (
        f"# === HARDCODED PATHS (do not modify) ===\n"
        f"TRAIN_INPUT = '{train_path}'\n"
        f"TEST_INPUT = '{test_path}'\n"
        f"TRAIN_OUTPUT = '{out_train}'\n"
        f"TEST_OUTPUT = '{out_test}'\n"
        f"TARGET_COLUMN = '{target_column}'\n"
    )

    agent_result = run_agent(
        agent_name="data_agent",
        state=state,
        rag_question=rag_q,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        code_prefix=code_prefix,
    )

    # Update state
    new_state = {**state}
    new_state["data_summary"] = data_summary_json
    new_state["column_types"] = column_types
    new_state["feature_code"] = agent_result["code"]
    new_state["agent_logs"] = state.get("agent_logs", []) + [agent_result["log_entry"]]

    if agent_result["result"]["ok"]:
        cleaned_train_path = str(Path(workspace) / "cleaned_train.csv")
        cleaned_test_path = str(Path(workspace) / "cleaned_test.csv")

        if Path(cleaned_train_path).exists():
            new_state["cleaned_data_path"] = cleaned_train_path
            new_state["phase"] = "data_ready"
            logger.info("[data_agent] Data cleaning successful")
        else:
            new_state["errors"] = state.get("errors", []) + ["Cleaned train.csv not produced"]
            new_state["phase"] = "data_ready"  # proceed anyway
            logger.warning("[data_agent] Cleaned CSV not found, proceeding anyway")
    else:
        error_msg = agent_result["result"].get("stderr", "")[:500]
        new_state["errors"] = state.get("errors", []) + [f"data_agent failed: {error_msg}"]
        new_state["phase"] = "data_ready"  # proceed to let model agent try
        logger.error("[data_agent] Failed: %s", error_msg)

    return new_state
