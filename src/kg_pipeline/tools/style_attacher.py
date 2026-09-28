"""Tool -- ``attach_styles``: gives the visual channel its own field.

A shape's legend style is *visual* information: it says what the box was drawn
like, not what is written on it. Without a field of its own, a colour could
only survive normalization as legend text pasted into the node's name, giving
names like ``"Multi-Head Attention\\nAttention Layer (Orange)"``.

This tool copies ``RawNode.style_id`` onto the corresponding ``NodeModel``, so
the channel has a home and the name can mean what it says.

**Deterministic on purpose.** The normalizer could be asked to echo the style
back, but that spends model attention on a lookup and gets it wrong sometimes;
the mapping is a pure id join and belongs in code (the agents-vs-tools split).

**Unscored.** The benchmark labels have no style field, so populating this
cannot move a benchmark number. It is a data-model correction, not an
improvement.
"""

import re

from kg_pipeline.workflow.state import PipelineState
from kg_pipeline.registry import NodeFn, PipelineDeps, register_node

# The normalizer reuses raw node ids, and ``sanitize_graph`` appends ``_<i>``
# when it explodes a fused block, so a final id maps back to its raw node by
# stripping one trailing numeric suffix. Falls back to the id itself.
_SUB_SUFFIX = re.compile(r"_\d+$")


@register_node(
    "attach_styles",
    kind="tool",
    description="Copies the raw shape's legend style_id onto the normalized node.",
)
def make_node(deps: PipelineDeps) -> NodeFn:
    def _node(state: PipelineState) -> dict:
        graph = state.get("graph")
        raw = state.get("raw_topology")
        if graph is None or raw is None:
            return {}

        styles = {n.id: n.style_id for n in raw.raw_nodes if n.style_id}
        if not styles:
            return {}

        updated = []
        for node in graph.nodes:
            if node.style_id:  # already set -- never overwrite
                updated.append(node)
                continue
            style = styles.get(node.id) or styles.get(_SUB_SUFFIX.sub("", node.id))
            updated.append(node.model_copy(update={"style_id": style}) if style else node)

        return {"graph": graph.model_copy(update={"nodes": updated})}

    return _node
