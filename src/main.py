"""CLI entry point for the Kaggle ML Agents pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from src.config import WORKSPACE_DIR, MAX_ITERATIONS
from src.safety.validators import check_competition_dir
from src.monitoring.logger import setup_logging, save_agent_logs
from src.monitoring.report import generate_report
from src.graph.builder import build_graph
from src.graph.state import PipelineState

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Kaggle ML Agents Pipeline")
    parser.add_argument("--competition-dir", required=True, help="Path to competition data directory")
    parser.add_argument("--target-column", required=True, help="Name of the target column")
    parser.add_argument("--metric", default="rmse", help="Evaluation metric (default: rmse)")
    parser.add_argument("--task", default="", help="Task description")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS, help="Max feedback loop iterations")
    parser.add_argument("--workspace", default=str(WORKSPACE_DIR), help="Workspace directory")
    args = parser.parse_args()

    # Setup workspace
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "code").mkdir(exist_ok=True)

    # Setup logging
    setup_logging(str(workspace))

    # Validate competition directory
    logger.info("Validating competition directory: %s", args.competition_dir)
    validation = check_competition_dir(args.competition_dir, args.target_column)
    if not validation["ok"]:
        logger.error("Validation failed: %s", validation["error"])
        sys.exit(1)

    logger.info("Validation passed. Columns: %s", validation["info"].get("train_columns", []))

    # Build initial state
    initial_state: PipelineState = {
        "competition_dir": str(Path(args.competition_dir).resolve()),
        "task_description": args.task or f"Predict {args.target_column} ({args.metric})",
        "target_column": args.target_column,
        "metric_name": args.metric,
        "workspace_dir": str(workspace.resolve()),
        "data_profile": "",
        "analyst_plan": "",
        "generated_code": "",
        "execution_stdout": "",
        "reviewer_feedback": "",
        "cv_scores": [],
        "metrics": {},
        "model_path": "",
        "submission_path": "",
        "phase": "init",
        "iteration": 0,
        "max_iterations": args.max_iterations,
        "agent_logs": [],
        "errors": [],
    }

    # Build and run graph
    logger.info("Building LangGraph pipeline...")
    graph = build_graph()

    logger.info("Starting pipeline (max %d iterations)...", args.max_iterations)
    final_state = graph.invoke(initial_state)

    # Save logs and report
    save_agent_logs(final_state.get("agent_logs", []), str(workspace))
    report = generate_report(final_state)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Submission: {final_state.get('submission_path', 'N/A')}")
    print(f"Iterations: {final_state.get('iteration', 0)}/{args.max_iterations}")
    print(f"CV Scores: {final_state.get('cv_scores', [])}")
    print(f"Report: {workspace / 'report.md'}")

    if final_state.get("errors"):
        print(f"\nWarnings/Errors ({len(final_state['errors'])}):")
        for e in final_state["errors"][-5:]:
            print(f"  - {e[:100]}")

    print("=" * 60)


if __name__ == "__main__":
    main()
