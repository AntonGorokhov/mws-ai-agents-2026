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

    # RAG context
    rag_context = rag_query("feature engineering categorical numerical datetime missing values model selection regression", "ml_knowledge", top_k=5)

    prev_scores = state.get("cv_scores", [])
    prev_errors = state.get("errors", [])
    reviewer_feedback = state.get("reviewer_feedback", "")

    system_prompt = (
        "You are a senior data scientist. Analyze the dataset and produce a JSON plan.\n\n"
        "Output ONLY valid JSON (no markdown fences, no explanation) with this structure:\n"
        "{\n"
        '  "drop_columns": ["col1", "col2"],\n'
        '  "features": [\n'
        '    {"name": "feature_name", "formula": "pandas expression", "description": "why"}\n'
        '  ],\n'
        '  "categorical_encoding": {"col": "method"},\n'
        '  "fill_na": {"col": "strategy"},\n'
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
        "- Drop columns that are too unique (names, IDs) or not useful.\n"
        "- For datetime columns: extract year, month, day_of_week, days_since.\n"
        "- For categorical: use 'label' (low cardinality), 'frequency' (high cardinality), or 'onehot'.\n"
        "- fill_na: 'median', 'zero', 'mode', or 'missing' (for categoricals).\n"
        "- Consider the target distribution when choosing target_transform.\n"
        "- Set clip_predictions based on target min/max.\n"
    )

    user_prompt = f"Data profile:\n{profile}\n\n"
    if rag_context:
        user_prompt += f"ML Knowledge:\n{rag_context[:2000]}\n\n"
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

    logger.info("[analyst] Plan generated: %d features, models: %s",
                len(plan.get("features", [])),
                list(plan.get("model_params", {}).keys()))

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
        "features": [
            {"name": "log_sum", "formula": "np.log1p(df['sum'])", "description": "log price"},
            {"name": "log_amt_reviews", "formula": "np.log1p(df['amt_reviews'])", "description": "log reviews"},
            {"name": "reviews_per_host", "formula": "df['amt_reviews'] / (df['total_host'] + 1)", "description": "reviews ratio"},
        ],
        "categorical_encoding": {
            "location_cluster": "onehot",
            "type_house": "onehot",
            "location": "frequency",
        },
        "fill_na": {
            "avg_reviews": "zero",
            "last_dt": "missing",
        },
        "model_params": {
            "lgbm": {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 2000},
            "xgb": {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 2000},
            "catboost": {"depth": 6, "learning_rate": 0.05, "iterations": 2000},
        },
        "target_transform": "none",
        "clip_predictions": [0, 365],
        "ensemble_method": "weighted_average",
    }
