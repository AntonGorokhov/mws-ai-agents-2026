"""Build the LangGraph StateGraph pipeline — new architecture."""

import logging

from langgraph.graph import StateGraph, END

from src.graph.state import PipelineState
from src.graph.supervisor import route
from src.agents.analyst import analyst_agent
from src.agents.coder import coder_agent
from src.agents.executor import executor_agent
from src.agents.reviewer import reviewer_agent

logger = logging.getLogger(__name__)


def _submit_node(state: PipelineState) -> PipelineState:
    """Final node: mark pipeline as complete."""
    new_state = {**state}
    new_state["phase"] = "complete"
    logger.info("[submit] Pipeline complete. Submission: %s", state.get("submission_path", "N/A"))
    return new_state


def _route_conditional(state: PipelineState) -> str:
    next_node = route(state)
    return next_node


_ALL_NODES = ["analyst", "coder", "executor", "reviewer", "submit"]


def build_graph() -> StateGraph:
    """Build and compile the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("analyst", analyst_agent)
    graph.add_node("coder", coder_agent)
    graph.add_node("executor", executor_agent)
    graph.add_node("reviewer", reviewer_agent)
    graph.add_node("submit", _submit_node)

    # Entry point
    graph.set_entry_point("analyst")

    # Each node → conditional routing
    routing_map = {n: n for n in _ALL_NODES}
    for node in ["analyst", "coder", "executor", "reviewer"]:
        graph.add_conditional_edges(node, _route_conditional, routing_map)

    # Submit → END
    graph.add_edge("submit", END)

    return graph.compile()
