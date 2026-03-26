"""Pipeline state definition."""

from typing import TypedDict


class PipelineState(TypedDict):
    # Input
    competition_dir: str
    task_description: str
    target_column: str
    metric_name: str

    # Working
    workspace_dir: str
    data_profile: str
    analyst_plan: str           # JSON plan from analyst
    generated_code: str         # Python code from coder
    execution_stdout: str       # stdout from executor
    reviewer_feedback: str      # text feedback from reviewer
    cv_scores: list[float]
    metrics: dict
    model_path: str
    submission_path: str

    # Control
    phase: str  # init → planned → coded → executed → reviewed → complete
    iteration: int
    max_iterations: int

    # Logs
    agent_logs: list[dict]
    errors: list[str]
