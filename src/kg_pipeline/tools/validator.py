"""Tool -- ``validate``: the sole author of ``critique_log``. Zero LLM calls.

Pydantic already guarantees shape (edge keys, role literals, group types) at
structured-output parse time; this validator adds the *semantic* checks the
type system can't express: referential integrity and the per-``sub_type``
allowed-property tables from ``prompts/schema_v5.8.md`` section 2.

Every critique names the offending id, so the corrector agent can repair
surgically. Node ids are checked for uniqueness but NOT for literal UUID
format -- vision-assigned ids like ``node_3`` are stable and traceable back
to the raw extraction, which matters more than the format.
"""

from kg_pipeline.workflow.state import PipelineState
from kg_pipeline.registry import NodeFn, PipelineDeps, register_node
from kg_pipeline.schema.graph import ArchitectureGraph

# Allowed `properties` keys per (class, sub_type), transcribed from
# prompts/schema_v5.8.md section 2. Update BOTH when the schema evolves.
GLOBAL_PROPERTIES = {"name", "symbol", "note", "sequence"}

_DATA_PROPERTIES = {"dimensions", "dtype", "modality"}

ALLOWED_PROPERTIES: dict[tuple[str, str], set[str]] = {
    ("DATA", "input"): _DATA_PROPERTIES,
    ("DATA", "output"): _DATA_PROPERTIES,
    ("DATA", "intermediate"): _DATA_PROPERTIES,
    ("DATA", "constant"): _DATA_PROPERTIES,
    ("LAYER", "convolution"): {"kernel", "channels", "stride", "dilation", "groups"},
    ("LAYER", "transposed_convolution"): {"kernel", "channels", "stride"},
    ("LAYER", "linear"): {"units", "features"},
    ("LAYER", "attention"): {"heads", "dim"},
    ("LAYER", "recurrent"): {"hidden_size", "direction"},
    ("LAYER", "embedding"): {"vocab_size", "dim"},
    ("LAYER", "custom"): {"algorithm"},
    ("MODIFIER", "activation"): {"algorithm"},
    ("MODIFIER", "normalization"): {"algorithm"},
    ("MODIFIER", "downsample"): {"algorithm", "kernel", "stride"},
    ("MODIFIER", "upsample"): {"algorithm"},
    ("MODIFIER", "reshape"): {"algorithm"},
    ("MODIFIER", "dropout"): {"rate"},
    ("MODIFIER", "mask"): {"algorithm"},
    ("MODIFIER", "element_wise"): {"algorithm"},
    ("MODIFIER", "positional_encoding"): {"algorithm"},
    ("MODIFIER", "loss"): {"algorithm"},
    ("MODIFIER", "custom"): {"algorithm"},
    ("JUNCTION", "add"): set(),
    ("JUNCTION", "average"): set(),
    ("JUNCTION", "multiply"): {"mechanism"},
    ("JUNCTION", "concat"): {"dim"},
    ("JUNCTION", "gating"): {"mechanism"},
    ("MACRO", "block"): {"repetition", "derived_from_definition"},
    ("MACRO", "network"): set(),
}


def validate_graph(graph: ArchitectureGraph) -> list[str]:
    """Pure function: returns a list of critiques; empty means valid."""
    critiques: list[str] = []

    node_ids: set[str] = set()
    for node in graph.nodes:
        if node.id in node_ids:
            critiques.append(f"node '{node.id}': duplicate node id")
        node_ids.add(node.id)

        key = (node.class_, node.sub_type)
        if key not in ALLOWED_PROPERTIES:
            valid = sorted(s for c, s in ALLOWED_PROPERTIES if c == node.class_)
            critiques.append(
                f"node '{node.id}': unknown sub_type '{node.sub_type}' for class "
                f"'{node.class_}' (valid: {', '.join(valid)})"
            )
            continue

        allowed = ALLOWED_PROPERTIES[key] | GLOBAL_PROPERTIES
        present = set(node.properties.model_dump(exclude_none=True))
        for extra in sorted(present - allowed):
            critiques.append(
                f"node '{node.id}': property '{extra}' is not allowed for "
                f"{node.class_}/{node.sub_type} (allowed: {', '.join(sorted(allowed))})"
            )

    group_ids: set[str] = set()
    for group in graph.groups:
        if group.id in group_ids:
            critiques.append(f"group '{group.id}': duplicate group id")
        group_ids.add(group.id)
        for member in group.member_node_ids:
            if member not in node_ids:
                critiques.append(
                    f"group '{group.id}': member '{member}' is not an existing node id"
                )

    for index, edge in enumerate(graph.edges):
        for endpoint, value in (("source", edge.source), ("target", edge.target)):
            if value not in node_ids:
                critiques.append(
                    f"edge #{index} ({edge.source} -> {edge.target}): {endpoint} "
                    f"'{value}' is not an existing node id"
                )

    for node in graph.nodes:
        target = node.properties.derived_from_definition
        if target is not None and target not in group_ids:
            critiques.append(
                f"node '{node.id}': derived_from_definition '{target}' is not an "
                "existing group id"
            )

    return critiques


@register_node(
    "validate",
    kind="tool",
    description="Checks referential integrity and per-sub_type property rules (v5.8).",
)
def make_node(deps: PipelineDeps) -> NodeFn:
    def _node(state: PipelineState) -> dict:
        critiques = validate_graph(state["graph"])
        return {"critique_log": critiques, "validated": not critiques}

    return _node
