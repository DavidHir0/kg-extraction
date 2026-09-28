"""Tool -- ``prune_bypasses``: Golden Rule 4, the Visual Bypass Rule.

Purges raw nodes that are completely empty (no text, no legend style) and act
as a simple 1-in/1-out passthrough, bridging the edge straight through.
Repeats until fixpoint so chains of empty boxes collapse. Also drops fully
isolated empty nodes (no text, no style, zero edges) -- stray decorative
pictograms the vision pass emitted despite its icon rules. Both criteria are
purely structural, which is why they live here as code and not in an LLM
prompt. Runs *before* the normalizer so no tokens are wasted on junk boxes.

Kept deliberately conservative: *connected* endpoints (0-in or 0-out with at
least one edge) and N-to-1 merge points survive -- classifying those as
DATA/JUNCTION is real judgment and belongs to the normalizer (its Golden
Rule 4 exceptions).
"""

from kg_pipeline.workflow.state import PipelineState
from kg_pipeline.registry import NodeFn, PipelineDeps, register_node
from kg_pipeline.schema.graph import RawEdge, RawTopology


def prune_bypasses(raw: RawTopology) -> RawTopology:
    """Pure function: returns a new RawTopology with bypass nodes bridged out."""
    nodes = {node.id: node for node in raw.raw_nodes}
    edges = list(raw.raw_edges)

    changed = True
    while changed:
        changed = False
        for node_id, node in list(nodes.items()):
            if node.raw_text_elements or node.style_id:
                continue
            incoming = [e for e in edges if e.target == node_id]
            outgoing = [e for e in edges if e.source == node_id]
            if len(incoming) != 1 or len(outgoing) != 1:
                continue
            in_edge, out_edge = incoming[0], outgoing[0]
            if in_edge.source == node_id or out_edge.target == node_id:
                continue  # self-loop; nothing sane to bridge
            edges.remove(in_edge)
            edges.remove(out_edge)
            edges.append(
                RawEdge(source=in_edge.source, target=out_edge.target, style=in_edge.style)
            )
            del nodes[node_id]
            changed = True

    # Decoration net: an empty node with no edges at all is visual noise
    # (e.g. an unlabeled pictogram); nothing downstream could classify it.
    for node_id, node in list(nodes.items()):
        if node.raw_text_elements or node.style_id:
            continue
        if any(edge.source == node_id or edge.target == node_id for edge in edges):
            continue
        del nodes[node_id]

    pruned_groups = [
        group.model_copy(
            update={"member_node_ids": [m for m in group.member_node_ids if m in nodes]}
        )
        for group in raw.raw_groups
    ]
    return raw.model_copy(
        update={
            "raw_nodes": list(nodes.values()),
            "raw_edges": edges,
            "raw_groups": pruned_groups,
        }
    )


@register_node(
    "prune_bypasses",
    kind="tool",
    description="Purges empty 1-in/1-out passthrough boxes and bridges their edges.",
)
def make_node(deps: PipelineDeps) -> NodeFn:
    def _node(state: PipelineState) -> dict:
        return {"raw_topology": prune_bypasses(state["raw_topology"])}

    return _node
