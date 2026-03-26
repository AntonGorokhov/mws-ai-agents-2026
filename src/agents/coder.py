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

    two_stage = plan.get("two_stage", {}).get("enabled", False)

    system_prompt = (
        "You are a Python code generator. You receive a JSON plan and produce a SINGLE Python script.\n\n"
        "Output ONLY Python code inside ```python``` fences. No explanations.\n"
        "Write FLAT script code — NO function definitions, NO classes, NO if __name__ blocks.\n"
        "Variables TRAIN_INPUT, TEST_INPUT, SUBMISSION_PATH, MODEL_PATH, TARGET_COLUMN are pre-injected.\n\n"
        "EXACT STRUCTURE (follow this order):\n"
        "1. Imports: pandas, numpy, lightgbm, xgboost, catboost, sklearn, pickle\n"
        "2. Load data: train_df = pd.read_csv(TRAIN_INPUT), test_df = pd.read_csv(TEST_INPUT)\n"
        "3. Feature engineering FIRST on BOTH train and test:\n"
        "   - Extract datetime features BEFORE dropping datetime columns\n"
        "   - Create has_date flags BEFORE filling NaN\n"
        "4. Drop 'drop_columns' (IDs, names)\n"
        "5. Drop 'drop_after_features' (datetime source columns, AFTER features extracted)\n"
        "6. Categorical encoding:\n"
        "   - For 'label': LabelEncoder\n"
        "   - For 'frequency': map(value_counts(normalize=True))\n"
        "   - For 'target': CV-based target encoding with smoothing (NO LEAKAGE!):\n"
        "     * Split train into 5 folds. For each fold, encode using OTHER folds' target mean.\n"
        "     * Apply smoothing: smooth = (count * cat_mean + smoothing * global_mean) / (count + smoothing)\n"
        "     * For test: use ALL train data to compute target means.\n"
        "     * Use smoothing=10 for location, smoothing=20 for host_name.\n"
        "     * Fill unseen categories with global_mean.\n"
        "     * IMPORTANT: do this BEFORE train/test split for CV, using only train_df and test_df.\n"
        "7. Fill NaN values\n"
        "8. Drop ALL remaining object/string columns: df.select_dtypes(exclude=['object'])\n"
        "9. Align train/test columns (sorted common columns)\n"
    )

    if two_stage:
        system_prompt += (
            "10. TWO-STAGE MODEL (plan has two_stage.enabled=true):\n"
            "   a. Stage 1 — Binary classifier: predict zero vs non-zero target\n"
            "      - y_binary = (y_train > 0).astype(int)\n"
            "      - Train LGBMClassifier with 5-fold CV, get P(non_zero) for test\n"
            "   b. Stage 2 — Regressor on non-zero samples ONLY:\n"
            "      - mask = y_train > 0\n"
            "      - Train LGBM/XGB/CatBoost ensemble on X_train[mask], y_train[mask]\n"
            "      - Get regression predictions for test\n"
            "   c. Combine: final_pred = classifier_proba * regression_pred\n"
            "   d. Clip to [0, 365]\n"
            "11. Save submission and model\n\n"
        )
    else:
        system_prompt += (
            "10. Train 3 models (LightGBM, XGBoost, CatBoost) with 5-fold CV\n"
            "11. Weighted ensemble (weights = 1/mse), clip predictions\n"
            "12. Save submission and model\n\n"
        )

    system_prompt += (
        "CRITICAL RULES:\n"
        "- Extract features from columns BEFORE dropping them!\n"
        "- mean_squared_error: use mean_squared_error(y, pred). squared=False is REMOVED.\n"
        "- LGBMRegressor/Classifier: callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]\n"
        "- XGBRegressor: early_stopping_rounds=100 in constructor, verbose=0 in fit()\n"
        "- CatBoostRegressor: verbose=0, early_stopping_rounds=100\n"
        "- Print 'MSE: <value>' for each model after CV\n"
        "- Wrap each model training in try/except — if one fails, continue with others\n"
        "- For the classifier in two-stage: use LGBMClassifier(objective='binary', n_estimators=1000)\n"
        "- SUBMISSION FORMAT: pd.DataFrame({'index': range(len(test_df)), 'prediction': final_pred}).to_csv(SUBMISSION_PATH, index=False)\n"
        "  Columns MUST be 'index' and 'prediction', NOT 'id' or 'target'!\n"
        "- If target_transform='log1p': apply y_train = np.log1p(y_train) before training, then np.expm1() on predictions BEFORE combining with classifier.\n"
        "- Weighted ensemble: weights = [1/mse_i for each model], normalize weights, then weighted_avg = sum(w_i * pred_i)\n"
        "- For target encoding columns (plan.categorical_encoding has 'target'): implement CV-based target encoding.\n"
        "  Do NOT use concat for target encoding — encode train and test SEPARATELY to avoid leakage.\n"
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

    logger.info("[coder] Generated %d lines of code (two_stage=%s)", code.count("\n") + 1, two_stage)

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
