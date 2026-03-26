"""Generate final pipeline report."""

import json
import logging
from pathlib import Path
from datetime import datetime

from src.graph.state import PipelineState

logger = logging.getLogger(__name__)


def generate_report(state: PipelineState) -> str:
    """Generate a markdown report of the pipeline run."""
    workspace = state.get("workspace_dir", "workspace")
    agent_logs = state.get("agent_logs", [])

    total_time = sum(log.get("duration_s", 0) for log in agent_logs)
    total_input_tokens = sum(log.get("tokens", {}).get("input", 0) for log in agent_logs)
    total_output_tokens = sum(log.get("tokens", {}).get("output", 0) for log in agent_logs)

    report_lines = [
        "# Pipeline Report",
        f"\nGenerated: {datetime.now().isoformat()}",
        f"\n## Summary",
        f"- Iterations: {state.get('iteration', 0)}/{state.get('max_iterations', 3)}",
        f"- Final phase: {state.get('phase', 'unknown')}",
        f"- Total time: {total_time:.1f}s",
        f"- Total tokens: {total_input_tokens + total_output_tokens} (in: {total_input_tokens}, out: {total_output_tokens})",
        f"- Submission: {state.get('submission_path', 'N/A')}",
    ]

    # Metrics
    metrics = state.get("metrics", {})
    if metrics:
        report_lines.append(f"\n## Metrics")
        for k, v in metrics.items():
            report_lines.append(f"- {k}: {v}")

    # CV Scores
    cv_scores = state.get("cv_scores", [])
    if cv_scores:
        report_lines.append(f"\n## CV Scores")
        for i, s in enumerate(cv_scores):
            report_lines.append(f"- Score {i+1}: {s:.4f}")

    # Agent logs
    report_lines.append(f"\n## Agent Activity")
    report_lines.append("| Agent | Model | Duration (s) | Tokens | Result |")
    report_lines.append("|-------|-------|-------------|--------|--------|")
    for log in agent_logs:
        tokens = log.get("tokens", {})
        total_t = tokens.get("input", 0) + tokens.get("output", 0)
        report_lines.append(
            f"| {log.get('agent', '?')} | {log.get('model_used', '?')} | "
            f"{log.get('duration_s', 0):.1f} | {total_t} | {log.get('result', '?')} |"
        )

    # Errors
    errors = state.get("errors", [])
    if errors:
        report_lines.append(f"\n## Errors")
        for e in errors:
            report_lines.append(f"- {e[:200]}")

    report_text = "\n".join(report_lines)

    # Save report
    report_path = Path(workspace) / "report.md"
    report_path.write_text(report_text)
    logger.info("Report saved to %s", report_path)

    return report_text
