"""Coder Agent: takes analyst's JSON plan, generates Python code."""

import json
import logging

from src.utils.llm import call_llm
from src.safety.guardrails import extract_python_code, sanitize_llm_output
from src.config import MODELS
from src.graph.state import PipelineState

logger = logging.getLogger(__name__)


def coder_agent(state: PipelineState) -> PipelineState:
    """Coder: generate Python code from analyst's plan."""
    plan = json.loads(state.get("analyst_plan", "{}"))
    workspace = state["workspace_dir"]
    competition_dir = state["competition_dir"]
    target_column = state["target_column"]

    # Build the code generation prompt with very specific instructions
    system_prompt = (
        "You are a Python code generator. You receive a JSON plan and produce a SINGLE Python script.\n\n"
        "RULES:\n"
        "- Output ONLY Python code inside ```python``` fences. No explanations.\n"
        "- Write FLAT script code — NO function definitions, NO classes, NO if __name__ blocks.\n"
        "- The script will have variables pre-injected at the top (TRAIN_INPUT, TEST_INPUT, etc).\n"
        "  Use those variables, do NOT redefine them.\n"
        "- The script must:\n"
        "  1. Read train/test CSVs using TRAIN_INPUT and TEST_INPUT\n"
        "  2. Apply feature engineering from the plan\n"
        "  3. Train models with 5-fold CV (KFold, shuffle=True, random_state=42)\n"
        "  4. Ensemble predictions (weighted by 1/rmse)\n"
        "  5. Clip predictions using the plan's clip range\n"
        "  6. Save submission.csv (columns: index, prediction) to SUBMISSION_PATH\n"
        "  7. Save model to MODEL_PATH with pickle\n"
        "  8. Print 'RMSE: <value>' for each model and final ensemble\n\n"
        "- For early stopping: use eval_set/valid_sets with callbacks\n"
        "- For CatBoost: use verbose=0\n"
        "- Handle errors gracefully: if a feature fails, skip it and continue\n"
        "- Import everything at the top: pandas, numpy, lightgbm, xgboost, catboost, sklearn, pickle\n"
        "- CRITICAL: After feature engineering, drop ALL string/object/datetime columns before training.\n"
        "  Use: X_train = X_train.select_dtypes(exclude=['object', 'datetime64'])\n"
        "- CRITICAL: When creating new features from a column (e.g. datetime), drop the original column after.\n"
        "- For LGBMRegressor early stopping use: callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]\n"
        "- For XGBRegressor early stopping: set early_stopping_rounds in constructor, not fit()\n"
        "- CRITICAL: Use np.sqrt(mean_squared_error(y, pred)) for RMSE. Do NOT use squared=False (removed).\n"
        "- CRITICAL: Ensure train and test have identical columns before training. Use common columns.\n"
    )

    plan_str = json.dumps(plan, indent=2)

    user_prompt = (
        f"Pre-defined variables (already injected, do NOT redefine):\n"
        f"  TRAIN_INPUT, TEST_INPUT, TRAIN_OUTPUT, TEST_OUTPUT\n"
        f"  SUBMISSION_PATH, MODEL_PATH, TARGET_COLUMN\n\n"
        f"JSON Plan:\n{plan_str}\n\n"
    )

    # If there's previous code and feedback, include it
    prev_code = state.get("generated_code", "")
    reviewer_feedback = state.get("reviewer_feedback", "")
    errors = state.get("errors", [])
    if prev_code and (reviewer_feedback or errors):
        user_prompt += f"PREVIOUS CODE (had issues, fix them):\n```python\n{prev_code[-3000:]}\n```\n\n"
        if reviewer_feedback:
            user_prompt += f"REVIEWER FEEDBACK:\n{reviewer_feedback}\n\n"
        if errors:
            user_prompt += f"ERRORS FROM LAST RUN:\n{chr(10).join(errors[-3:])}\n\n"

    user_prompt += "Generate the complete Python script."

    llm_result = call_llm(
        model=MODELS["coder"],
        system=system_prompt,
        user=user_prompt,
        fallback=MODELS["coder_fallback"],
        max_tokens=8192,
    )

    raw = sanitize_llm_output(llm_result["content"])
    code = extract_python_code(raw)

    logger.info("[coder] Generated %d lines of code", code.count("\n") + 1)

    new_state = {**state}
    new_state["generated_code"] = code
    new_state["phase"] = "coded"
    new_state["agent_logs"] = state.get("agent_logs", []) + [{
        "agent": "coder",
        "model_used": llm_result["model_used"],
        "tokens": llm_result["tokens"],
        "result": "success",
    }]
    return new_state
