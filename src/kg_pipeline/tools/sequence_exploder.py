"""Tool -- ``explode_sequences``: Golden Rule 3's deterministic other half.

The normalizer *decides* what is fused into a ``properties.sequence`` (that
is judgment); this tool mechanically explodes those composites into flat,
individually typed nodes with re-wired edges and expanded group membership.
Thin registry wrapper around :func:`kg_pipeline.tools.sanitizer.sanitize_graph`,
which is idempotent -- safe to re-run after each correction loop.
"""

from kg_pipeline.workflow.state import PipelineState
from kg_pipeline.tools.sanitizer import sanitize_graph
from kg_pipeline.registry import NodeFn, PipelineDeps, register_node


@register_node(
    "explode_sequences",
    kind="tool",
    description="Explodes composite properties.sequence nodes into atomic nodes.",
)
def make_node(deps: PipelineDeps) -> NodeFn:
    def _node(state: PipelineState) -> dict:
        return {"graph": sanitize_graph(state["graph"])}

    return _node
