"""Analyst Agent: analyzes data, outputs JSON instructions for the Coder."""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.utils.llm import call_llm
from src.rag.retriever import rag_query
from src.safety.guardrails import sanitize_llm_output
from src.config import MODELS
from src.graph.state import PipelineState

logger = logging.getLogger(__name__)


def _build_data_profile(competition_dir: str, target_column: str) -> str:
    """Build a detailed data profile as text."""
    train = pd.read_csv(Path(competition_dir) / "train.csv")
    test = pd.read_csv(Path(competition_dir) / "test.csv")

    lines = [
        f"Train: {train.shape[0]} rows, {train.shape[1]} cols",
        f"Test: {test.shape[0]} rows, {test.shape[1]} cols",
        f"Target: '{target_column}' — min={train[target_column].min()}, max={train[target_column].max()}, "
        f"mean={train[target_column].mean():.2f}, median={train[target_column].median():.2f}, "
        f"std={train[target_column].std():.2f}, skew={train[target_column].skew():.2f}",
        f"Target zeros: {(train[target_column] == 0).sum()} ({(train[target_column] == 0).mean()*100:.1f}%)",
        "",
        "Columns:",
    ]

    for col in train.columns:
        dtype = train[col].dtype
        nuniq = train[col].nunique()
        null_pct = train[col].isna().mean() * 100
        sample = train[col].dropna().head(3).tolist()

        if dtype in ("int64", "float64") and col != target_column:
            corr = train[col].corr(train[target_column])
            lines.append(f"  {col}: {dtype}, nunique={nuniq}, null={null_pct:.1f}%, corr_target={corr:.3f}, sample={sample}")
        else:
            lines.append(f"  {col}: {dtype}, nunique={nuniq}, null={null_pct:.1f}%, sample={sample}")

    return "\n".join(lines)


def analyst_agent(state: PipelineState) -> PipelineState:
    """Analyst: analyze data and produce JSON instructions."""
    competition_dir = state["competition_dir"]
    target_column = state["target_column"]
    iteration = state.get("iteration", 0)

    # Build data profile
    profile = _build_data_profile(competition_dir, target_column)

    # RAG context — include zero-inflated and datetime topics
    rag_context = rag_query(
        "zero-inflated two-stage classification regression datetime feature engineering categorical missing values",
        "ml_knowledge", top_k=8
    )

    prev_scores = state.get("cv_scores", [])
    prev_errors = state.get("errors", [])
    reviewer_feedback = state.get("reviewer_feedback", "")

    system_prompt = (
        "You are a senior data scientist. Analyze the dataset and produce a JSON plan.\n\n"
        "Output ONLY valid JSON (no markdown fences, no explanation) with this structure:\n"
        "{\n"
        '  "drop_columns": ["col1", "col2"],\n'
        '  "drop_after_features": ["datetime_col"],\n'
        '  "features": [\n'
        '    {"name": "feature_name", "formula": "pandas expression", "description": "why"}\n'
        '  ],\n'
        '  "categorical_encoding": {"col": "method"},\n'
        '  "fill_na": {"col": "strategy"},\n'
        '  "two_stage": {"enabled": true/false, "zero_threshold": 0},\n'
        '  "model_params": {\n'
        '    "lgbm": {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 2000, ...},\n'
        '    "xgb": {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 2000, ...},\n'
        '    "catboost": {"depth": 6, "learning_rate": 0.05, "iterations": 2000, ...}\n'
        '  },\n'
        '  "target_transform": "none" or "log1p",\n'
        '  "clip_predictions": [min, max],\n'
        '  "ensemble_method": "weighted_average"\n'
        "}\n\n"
        "RULES:\n"
        "- drop_columns: columns to drop IMMEDIATELY (IDs, names, useless).\n"
        "- drop_after_features: datetime/string columns to drop AFTER extracting features from them.\n"
        "  NEVER put datetime columns in drop_columns if you plan to extract features from them!\n"
        "- For datetime columns: extract year, month, day_of_week, days_since_reference, has_date flag.\n"
        "- For categorical: use 'label' (low cardinality), 'frequency' (high cardinality), or 'target' encoding.\n"
        "- fill_na: 'median', 'zero', 'mode', or 'missing' (for categoricals).\n"
        "- two_stage: if target has >20% zeros, set enabled=true. This trains a classifier (zero/non-zero)\n"
        "  then a regressor on non-zero samples. Final pred = P(non_zero) * regression_pred.\n"
        "- The competition metric is MSE. Optimize for it.\n"
        "- Set clip_predictions based on target min/max.\n"
    )

    user_prompt = f"Data profile:\n{profile}\n\n"
    if rag_context:
        user_prompt += f"ML Knowledge:\n{rag_context[:3000]}\n\n"
    if iteration > 0 and reviewer_feedback:
        user_prompt += f"Previous iteration feedback:\n{reviewer_feedback}\n"
        user_prompt += f"Previous CV scores: {prev_scores}\n"
        user_prompt += f"Previous errors: {prev_errors[-3:]}\n"
    user_prompt += "\nOutput the JSON plan:"

    llm_result = call_llm(
        model=MODELS["reasoning"],
        system=system_prompt,
        user=user_prompt,
        fallback=MODELS["reasoning_fallback"],
        max_tokens=4096,
    )

    raw = sanitize_llm_output(llm_result["content"])

    # Parse JSON — try to extract from response
    plan = _parse_json_plan(raw)

    # Ensure two_stage field exists
    if "two_stage" not in plan:
        plan["two_stage"] = {"enabled": False, "zero_threshold": 0}

    # Ensure drop_after_features exists
    if "drop_after_features" not in plan:
        plan["drop_after_features"] = []

    logger.info("[analyst] Plan generated: %d features, models: %s, two_stage: %s",
                len(plan.get("features", [])),
                list(plan.get("model_params", {}).keys()),
                plan.get("two_stage", {}).get("enabled", False))

    new_state = {**state}
    new_state["analyst_plan"] = json.dumps(plan, default=str)
    new_state["data_profile"] = profile
    new_state["phase"] = "planned"
    new_state["agent_logs"] = state.get("agent_logs", []) + [{
        "agent": "analyst",
        "model_used": llm_result["model_used"],
        "tokens": llm_result["tokens"],
        "result": "success",
    }]
    return new_state


def _parse_json_plan(raw: str) -> dict:
    """Extract JSON from LLM response."""
    import re
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown fences
    match = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("[analyst] Failed to parse JSON, using defaults")
    return _default_plan()


def _default_plan() -> dict:
    return {
        "drop_columns": ["name", "_id", "host_name"],
        "drop_after_features": ["last_dt"],
        "features": [
            {"name": "last_dt_year", "formula": "pd.to_datetime(df['last_dt'], errors='coerce').dt.year", "description": "year of last review"},
            {"name": "last_dt_month", "formula": "pd.to_datetime(df['last_dt'], errors='coerce').dt.month", "description": "month of last review"},
            {"name": "last_dt_dayofweek", "formula": "pd.to_datetime(df['last_dt'], errors='coerce').dt.dayofweek", "description": "day of week"},
            {"name": "days_since_last_review", "formula": "(pd.Timestamp('2020-01-01') - pd.to_datetime(df['last_dt'], errors='coerce')).dt.days", "description": "recency"},
            {"name": "has_last_review", "formula": "df['last_dt'].notna().astype(int)", "description": "has any review"},
            {"name": "log_sum", "formula": "np.log1p(df['sum'])", "description": "log price"},
            {"name": "log_amt_reviews", "formula": "np.log1p(df['amt_reviews'])", "description": "log reviews"},
            {"name": "reviews_per_host", "formula": "df['amt_reviews'] / (df['total_host'] + 1)", "description": "reviews ratio"},
        ],
        "categorical_encoding": {
            "location_cluster": "label",
            "type_house": "label",
            "location": "frequency",
        },
        "fill_na": {
            "avg_reviews": "zero",
            "days_since_last_review": "median",
            "last_dt_year": "median",
            "last_dt_month": "median",
            "last_dt_dayofweek": "median",
        },
        "two_stage": {"enabled": True, "zero_threshold": 0},
        "model_params": {
            "lgbm": {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 2000},
            "xgb": {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 2000},
            "catboost": {"depth": 6, "learning_rate": 0.05, "iterations": 2000},
        },
        "target_transform": "none",
        "clip_predictions": [0, 365],
        "ensemble_method": "weighted_average",
    }
