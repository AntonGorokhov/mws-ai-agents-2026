"""Reviewer Agent: analyzes results, provides feedback for next iteration."""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.utils.llm import call_llm
from src.safety.guardrails import sanitize_llm_output
from src.config import MODELS
from src.graph.state import PipelineState

logger = logging.getLogger(__name__)


def reviewer_agent(state: PipelineState) -> PipelineState:
    """Review execution results and decide next steps."""
    workspace = state["workspace_dir"]
    competition_dir = state["competition_dir"]
    target_column = state["target_column"]
    iteration = state.get("iteration", 0)
    cv_scores = state.get("cv_scores", [])
    stdout = state.get("execution_stdout", "")
    submission_path = state.get("submission_path", "")

    # Gather facts for the reviewer
    facts = [f"Iteration: {iteration}", f"CV scores: {cv_scores}"]

    if submission_path and Path(submission_path).exists():
        sub = pd.read_csv(submission_path)
        facts.append(f"Submission: {len(sub)} rows")
        facts.append(f"Prediction range: [{sub['prediction'].min():.2f}, {sub['prediction'].max():.2f}]")
        facts.append(f"Prediction mean: {sub['prediction'].mean():.2f}")
        facts.append(f"NaN predictions: {sub['prediction'].isna().sum()}")
    else:
        facts.append("No submission produced!")

    train = pd.read_csv(Path(competition_dir) / "train.csv")
    facts.append(f"Target range: [{train[target_column].min()}, {train[target_column].max()}]")
    facts.append(f"Target mean: {train[target_column].mean():.2f}")

    errors = state.get("errors", [])
    if errors:
        facts.append(f"Errors: {errors[-3:]}")

    facts.append(f"Execution output (last 500 chars): {stdout[-500:]}")

    system_prompt = (
        "You are a senior ML reviewer. Analyze the pipeline results and give feedback.\n\n"
        "Output ONLY valid JSON (no fences, no explanation):\n"
        "{\n"
        '  "score": <best RMSE as float>,\n'
        '  "issues": ["issue1", "issue2"],\n'
        '  "feedback": "specific actionable feedback for next iteration",\n'
        '  "recommendation": "improve_features" or "tune_model" or "submit"\n'
        "}\n\n"
        "- If no submission was produced, recommend 'improve_features'.\n"
        "- If score is reasonable and improving, recommend 'submit'.\n"
        "- If prediction range doesn't match target range, flag it.\n"
    )

    user_prompt = "Results:\n" + "\n".join(facts)

    llm_result = call_llm(
        model=MODELS["reasoning"],
        system=system_prompt,
        user=user_prompt,
        fallback=MODELS["reasoning_fallback"],
        max_tokens=2048,
    )

    raw = sanitize_llm_output(llm_result["content"])
    review = _parse_review(raw)

    logger.info("[reviewer] Score: %s, Recommendation: %s",
                review.get("score"), review.get("recommendation"))

    new_state = {**state}
    new_state["metrics"] = review
    new_state["reviewer_feedback"] = review.get("feedback", "")
    new_state["iteration"] = iteration + 1
    new_state["phase"] = "reviewed"
    new_state["agent_logs"] = state.get("agent_logs", []) + [{
        "agent": "reviewer",
        "model_used": llm_result["model_used"],
        "tokens": llm_result["tokens"],
        "result": "success",
    }]
    return new_state


def _parse_review(raw: str) -> dict:
    import re
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"score": 999.0, "issues": [], "feedback": "", "recommendation": "submit"}
