from kg_pipeline.schema.graph import ArchitectureGraph
from kg_pipeline.tools.validator import validate_graph


def _graph(**overrides) -> ArchitectureGraph:
    payload = {
        "architecture_name": "net",
        "nodes": [
            {"id": "in", "class": "DATA", "sub_type": "input", "raw_text": "Input"},
            {
                "id": "conv",
                "class": "LAYER",
                "sub_type": "convolution",
                "raw_text": "3x3 Conv",
                "properties": {"kernel": [3, 3], "name": "Stem"},
            },
        ],
        "edges": [{"source": "in", "target": "conv", "role": "forward"}],
        "groups": [],
    }
    payload.update(overrides)
    return ArchitectureGraph.model_validate(payload)


def test_valid_graph_yields_no_critiques():
    assert validate_graph(_graph()) == []


def test_dangling_edge_endpoint_is_critiqued():
    graph = _graph(edges=[{"source": "in", "target": "ghost", "role": "forward"}])

    critiques = validate_graph(graph)

    assert len(critiques) == 1
    assert "ghost" in critiques[0] and "target" in critiques[0]


def test_disallowed_property_for_sub_type_is_critiqued():
    graph = _graph(
        nodes=[
            {
                "id": "act",
                "class": "MODIFIER",
                "sub_type": "activation",
                "raw_text": "ReLU",
                # kernel is a convolution property, not an activation one
                "properties": {"kernel": [3, 3]},
            }
        ],
        edges=[],
    )

    critiques = validate_graph(graph)

    assert len(critiques) == 1
    assert "'kernel'" in critiques[0] and "act" in critiques[0]


def test_unknown_sub_type_is_critiqued_with_valid_options():
    graph = _graph(
        nodes=[
            {"id": "x", "class": "JUNCTION", "sub_type": "blend", "raw_text": ""},
        ],
        edges=[],
    )

    critiques = validate_graph(graph)

    assert len(critiques) == 1
    assert "blend" in critiques[0] and "concat" in critiques[0]


def test_duplicate_node_id_is_critiqued():
    graph = _graph(
        nodes=[
            {"id": "in", "class": "DATA", "sub_type": "input", "raw_text": "A"},
            {"id": "in", "class": "DATA", "sub_type": "output", "raw_text": "B"},
        ],
        edges=[],
    )

    assert any("duplicate node id" in c for c in validate_graph(graph))


def test_group_membership_and_derived_from_definition_are_checked():
    graph = _graph(
        nodes=[
            {
                "id": "macro",
                "class": "MACRO",
                "sub_type": "block",
                "raw_text": "Residual Block",
                "properties": {"derived_from_definition": "missing_group"},
            }
        ],
        edges=[],
        groups=[
            {
                "id": "g1",
                "type": "subgraph_definition",
                "label": "Residual Block",
                "boundary_type": "explicit_box",
                "member_node_ids": ["macro", "ghost"],
            }
        ],
    )

    critiques = validate_graph(graph)

    assert any("'ghost'" in c for c in critiques)
    assert any("missing_group" in c for c in critiques)
