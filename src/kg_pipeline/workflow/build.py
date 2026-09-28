"""Pipeline topology: wires registered nodes into a LangGraph ``StateGraph``.

The flow (also rendered as a diagram by ``python main.py diagram``):

    START -> vision_extract
               |- no raw nodes ------------------> export
             prune_bypasses -> resolve_legend -> normalize
               |- no graph produced --------------> export
             attach_styles -> normalize_terms -> explode_sequences -> validate
               |- critiques & retries left -> correct -> attach_styles (loop)
               '- clean, or retry budget spent ---> export -> END

Adding a node: register it (see ``kg_pipeline.registry``), append its name to
``PIPELINE_NODES``, and wire its edges below. Nothing else changes.
"""

from langgraph.graph import END, START, StateGraph

from kg_pipeline.workflow.routing import (
    route_after_normalize,
    route_after_validate,
    route_after_vision,
)
from kg_pipeline.workflow.state import PipelineState
from kg_pipeline.registry import PipelineDeps, build_node, discover_nodes

# Every node participating in the pipeline, by registry name, in execution
# order. Whether a node is an LLM agent or a deterministic tool is recorded in
# the registry.
PIPELINE_NODES = [
    "vision_extract",
    "prune_bypasses",
    "resolve_legend",
    "normalize",
    "attach_styles",  # deterministic: raw style_id -> normalized node
    "normalize_terms",
    "explode_sequences",
    "validate",
    "correct",  # conditional: only on validation failure, loops via attach_styles
    "export",
]


def build_pipeline(deps: PipelineDeps):
    """Builds and compiles the full extraction graph from the node registry."""
    discover_nodes()

    builder = StateGraph(PipelineState)
    for name in PIPELINE_NODES:
        builder.add_node(name, build_node(name, deps))

    builder.add_edge(START, "vision_extract")
    builder.add_conditional_edges(
        "vision_extract",
        route_after_vision,
        {"no_content": "export", "parse": "prune_bypasses"},
    )
    builder.add_edge("prune_bypasses", "resolve_legend")
    builder.add_edge("resolve_legend", "normalize")
    builder.add_conditional_edges(
        "normalize",
        route_after_normalize,
        {"failed": "export", "continue": "attach_styles"},
    )
    builder.add_edge("attach_styles", "normalize_terms")
    builder.add_edge("normalize_terms", "explode_sequences")
    builder.add_edge("explode_sequences", "validate")
    builder.add_conditional_edges(
        "validate", route_after_validate, {"export": "export", "correct": "correct"}
    )
    # via attach_styles: the corrector rebuilds the graph from the LLM and
    # cannot know style_id, so the join is re-applied after every repair.
    builder.add_edge("correct", "attach_styles")
    builder.add_edge("export", END)

    return builder.compile()

