from pydantic import ValidationError

from kg_pipeline.schema.graph import ArchitectureGraph, EdgeModel, NodeModel

# Framework-default hyperparameters the LLM emits from background knowledge
# rather than from the figure (stride/dilation/groups of 1). Figures never
# label defaults and annotators never record them, so keeping these costs
# precision with zero recall: run_122b had dilation 252/264 and groups 276/278
# as the literal default, with 0 true positives. Non-default values ("/2",
# groups="C") are real figure content and pass through.
_DEFAULTABLE_KEYS = ("stride", "dilation", "groups")


def _is_default(value) -> bool:
    if isinstance(value, list):
        return bool(value) and all(_is_default(v) for v in value)
    return value in (1, "1")


def _drop_default_properties(node: NodeModel) -> NodeModel:
    cleared = {k: None for k in _DEFAULTABLE_KEYS if _is_default(getattr(node.properties, k))}
    if not cleared:
        return node
    return node.model_copy(update={"properties": node.properties.model_copy(update=cleared)})


def sanitize_graph(graph: ArchitectureGraph) -> ArchitectureGraph:
    """Pass 3: deterministic, no LLM call.

    Explodes composite nodes (``properties.sequence``) into flat, individually
    typed atomic nodes, re-wires edges to enter/leave at the sequence's
    boundary nodes, and expands group membership to include every exploded
    node. Port of ``schema_enforcer._explode_composite_nodes``.

    Unlike the original, this doesn't need a separate null-stripping step or
    an explicit re-validation pass: Pydantic validates each ``NodeModel`` on
    construction, and ``model_dump(exclude_none=True)`` at export time drops
    unset fields.
    """
    node_boundary: dict[str, tuple[str, str]] = {}
    node_members: dict[str, list[str]] = {}
    flat_nodes: list[NodeModel] = []
    injected_edges: list[EdgeModel] = []

    for node in graph.nodes:
        sequence = node.properties.sequence

        if not sequence:
            flat_nodes.append(node)
            node_boundary[node.id] = (node.id, node.id)
            node_members[node.id] = [node.id]
            continue

        sub_ids = [f"{node.id}_{i}" for i in range(len(sequence))]
        for step, (sub_id, element) in enumerate(zip(sub_ids, sequence), start=1):
            # Prefer the step's OWN operation name when the normalizer supplied
            # it, so an exploded conv/BN/relu reads "Conv 3x3" rather than
            # "E1 (Step 1)", which fails text matching. Falls back to the
            # parent-derived text when no label is present.
            raw_text = element.label or f"{node.raw_text} (Step {step})"
            # A sequence step is typed loosely on ArchitectureGraph, so a model
            # can emit one that the graph accepts and NodeModel rejects (e.g.
            # `kernel: "3x3"` where the schema wants `[3, 3]`). Before this
            # guard a single such step raised out of the whole evaluation, so
            # one bad node cost all 73 figures. Keep the node, drop only the
            # properties that will not parse.
            def _step_node(props):
                return NodeModel.model_validate(
                    {
                        "id": sub_id,
                        "class": element.class_,
                        "sub_type": element.sub_type,
                        "raw_text": raw_text,
                        # Every step of a fused block was drawn in the parent's
                        # style, so the visual attribution survives explosion.
                        "style_id": node.style_id,
                        "properties": props,
                    }
                )

            try:
                flat_nodes.append(_step_node(element.properties))
            except ValidationError:
                flat_nodes.append(_step_node({}))
        for source, target in zip(sub_ids, sub_ids[1:]):
            injected_edges.append(EdgeModel(source=source, target=target, role="forward"))

        node_boundary[node.id] = (sub_ids[0], sub_ids[-1])
        node_members[node.id] = sub_ids

    rewired_edges = [
        edge.model_copy(
            update={
                "source": node_boundary.get(edge.source, (edge.source, edge.source))[1],
                "target": node_boundary.get(edge.target, (edge.target, edge.target))[0],
            }
        )
        for edge in [*graph.edges, *injected_edges]
    ]

    expanded_groups = [
        group.model_copy(
            update={
                "member_node_ids": [
                    sub_id
                    for member_id in group.member_node_ids
                    for sub_id in node_members.get(member_id, [member_id])
                ]
            }
        )
        for group in graph.groups
    ]

    return graph.model_copy(
        update={
            "nodes": [_drop_default_properties(n) for n in flat_nodes],
            "edges": rewired_edges,
            "groups": expanded_groups,
        }
    )
