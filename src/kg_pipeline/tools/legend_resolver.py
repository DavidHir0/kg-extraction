"""Tool -- ``resolve_legend``: substitutes legend meaning into styled shapes.

Pure dict-lookup over ``legend_map`` (no judgment, so it stays out of the LLM
prompt): for every raw node tagged with a ``style_id``, append the legend's
text to its ``raw_text_elements``. The normalizer then treats it as if the
author wrote it inside the shape.

**Edges are deliberately not touched here.** A legend entry either names an
operation, which the vision prompt already turns into a node, or a connection
type, which becomes ``EdgeModel.role``. Turning legend-styled arrows into
``A -> Op -> B`` here was tried and lowered benchmark scores.
"""

from __future__ import annotations

from kg_pipeline.workflow.state import PipelineState
from kg_pipeline.registry import NodeFn, PipelineDeps, register_node
from kg_pipeline.schema.graph import RawTopology


def resolve_legend(raw: RawTopology) -> RawTopology:
    """Pure function: returns a new RawTopology with legend meaning merged in."""
    if not raw.legend_map:
        return raw

    resolved_nodes = []
    for node in raw.raw_nodes:
        legend_text = raw.legend_map.get(node.style_id or "")
        if legend_text and legend_text not in node.raw_text_elements:
            node = node.model_copy(
                update={"raw_text_elements": [*node.raw_text_elements, legend_text]}
            )
        resolved_nodes.append(node)

    return raw.model_copy(update={"raw_nodes": resolved_nodes})


@register_node(
    "resolve_legend",
    kind="tool",
    description="Merge legend_map into styled nodes.",
)
def make_node(deps: PipelineDeps) -> NodeFn:
    def _node(state: PipelineState) -> dict:
        return {"raw_topology": resolve_legend(state["raw_topology"])}

    return _node
