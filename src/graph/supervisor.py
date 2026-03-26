"""Supervisor: pure Python router for new architecture."""

import logging

from src.graph.state import PipelineState

logger = logging.getLogger(__name__)


def route(state: PipelineState) -> str:
    """Route to next node based on phase."""
    phase = state.get("phase", "init")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    logger.info("[supervisor] Phase: %s, Iteration: %d/%d", phase, iteration, max_iterations)

    if phase == "init":
        return "analyst"

    if phase == "planned":
        return "coder"

    if phase == "coded":
        return "executor"

    if phase == "executed":
        return "reviewer"

    if phase == "reviewed":
        if iteration >= max_iterations:
            logger.info("[supervisor] Max iterations reached → submit")
            return "submit"

        rec = state.get("metrics", {}).get("recommendation", "submit")
        if rec in ("improve_features", "tune_model"):
            logger.info("[supervisor] Reviewer says '%s' → analyst", rec)
            return "analyst"

        logger.info("[supervisor] Reviewer says submit")
        return "submit"

    if phase == "complete":
        return "submit"

    logger.warning("[supervisor] Unknown phase '%s' → submit", phase)
    return "submit"
