"""Conditional-edge routing functions for the pipeline graph.

Pure functions of :class:`PipelineState` -- no I/O, no LLM calls -- so every
branch is unit-testable. Routers deliberately check for the *presence of the
data they need* rather than ``state["error"]``: a stale error string from an
earlier stage must not poison routing decisions downstream of a stage that
succeeded.
"""

from kg_pipeline.workflow.state import PipelineState

# Maximum self-correction attempts before the graph is exported as-is with
# its critique log ("unvalidated"). Each attempt costs one corrector LLM call.
MAX_CORRECTION_RETRIES = 3


def route_after_vision(state: PipelineState) -> str:
    """no_content -> export | parse -> prune_bypasses."""
    raw_topology = state.get("raw_topology")
    if raw_topology is None or not raw_topology.raw_nodes:
        return "no_content"
    return "parse"


def route_after_normalize(state: PipelineState) -> str:
    """failed -> export | continue -> normalize_terms."""
    graph = state.get("graph")
    if graph is None or not graph.nodes:
        return "failed"
    return "continue"


def route_after_validate(state: PipelineState) -> str:
    """export when clean or the retry budget is spent; otherwise correct."""
    if state.get("validated"):
        return "export"
    if state.get("retry_count", 0) >= MAX_CORRECTION_RETRIES:
        return "export"
    return "correct"
