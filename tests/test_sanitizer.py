from kg_pipeline.tools.sanitizer import sanitize_graph
from kg_pipeline.schema.graph import ArchitectureGraph


def test_atomic_nodes_pass_through_unchanged():
    graph = ArchitectureGraph.model_validate(
        {
            "architecture_name": "net",
            "nodes": [
                {"id": "a", "class": "DATA", "sub_type": "input", "raw_text": "Input"},
                {"id": "b", "class": "LAYER", "sub_type": "convolution", "raw_text": "Conv"},
            ],
            "edges": [{"source": "a", "target": "b", "role": "forward"}],
        }
    )

    sanitized = sanitize_graph(graph)

    assert [n.id for n in sanitized.nodes] == ["a", "b"]
    assert sanitized.edges[0].source == "a"
    assert sanitized.edges[0].target == "b"


def test_composite_sequence_is_exploded_and_edges_rewired():
    graph = ArchitectureGraph.model_validate(
        {
            "architecture_name": "net",
            "nodes": [
                {"id": "in", "class": "DATA", "sub_type": "input", "raw_text": "Input"},
                {
                    "id": "block",
                    "class": "LAYER",
                    "sub_type": "convolution",
                    "raw_text": "Conv+BN+ReLU",
                    "properties": {
                        "sequence": [
                            {"class": "LAYER", "sub_type": "convolution", "properties": {}},
                            {"class": "MODIFIER", "sub_type": "normalization", "properties": {}},
                            {"class": "MODIFIER", "sub_type": "activation", "properties": {}},
                        ]
                    },
                },
                {"id": "out", "class": "DATA", "sub_type": "output", "raw_text": "Output"},
            ],
            "edges": [
                {"source": "in", "target": "block", "role": "forward"},
                {"source": "block", "target": "out", "role": "forward"},
            ],
            "groups": [
                {
                    "id": "g1",
                    "type": "container",
                    "label": "Stem",
                    "boundary_type": "explicit_box",
                    "member_node_ids": ["in", "block"],
                }
            ],
        }
    )

    sanitized = sanitize_graph(graph)

    # "block" explodes into 3 atomic nodes; "in"/"out" pass through untouched.
    node_ids = [n.id for n in sanitized.nodes]
    assert node_ids == ["in", "block_0", "block_1", "block_2", "out"]
    assert [n.sub_type for n in sanitized.nodes if n.id.startswith("block_")] == [
        "convolution",
        "normalization",
        "activation",
    ]

    # Internal chain edges are injected, and the original in/out edges are
    # rewired to the sequence's boundary nodes.
    edge_pairs = {(e.source, e.target) for e in sanitized.edges}
    assert edge_pairs == {
        ("in", "block_0"),
        ("block_0", "block_1"),
        ("block_1", "block_2"),
        ("block_2", "out"),
    }

    # Group membership expands to every exploded node.
    assert sanitized.groups[0].member_node_ids == ["in", "block_0", "block_1", "block_2"]


def test_framework_default_properties_are_dropped():
    # Figures never label stride/dilation/groups defaults; the LLM emits them
    # from background knowledge. Real (non-default) values must survive.
    graph = ArchitectureGraph.model_validate(
        {
            "nodes": [
                {"id": "a", "class": "LAYER", "sub_type": "convolution", "raw_text": "Conv",
                 "properties": {"kernel": [3, 3], "stride": [1, 1], "dilation": 1, "groups": 1}},
                {"id": "b", "class": "LAYER", "sub_type": "convolution", "raw_text": "Conv /2",
                 "properties": {"stride": 2, "dilation": [1, 2, 4], "groups": "C"}},
            ],
        }
    )

    nodes = {n.id: n for n in sanitize_graph(graph).nodes}

    assert nodes["a"].properties.stride is None
    assert nodes["a"].properties.dilation is None
    assert nodes["a"].properties.groups is None
    assert nodes["a"].properties.kernel == [3, 3]  # non-defaultable key untouched
    assert nodes["b"].properties.stride == 2
    assert nodes["b"].properties.dilation == [1, 2, 4]
    assert nodes["b"].properties.groups == "C"


def test_default_drop_applies_inside_exploded_sequences():
    graph = ArchitectureGraph.model_validate(
        {
            "nodes": [
                {"id": "block", "class": "LAYER", "sub_type": "convolution", "raw_text": "Conv+BN",
                 "properties": {"sequence": [
                     {"class": "LAYER", "sub_type": "convolution",
                      "properties": {"stride": [1, 1], "kernel": [3, 3]}},
                     {"class": "MODIFIER", "sub_type": "normalization", "properties": {}},
                 ]}},
            ],
        }
    )

    exploded = {n.id: n for n in sanitize_graph(graph).nodes}

    assert exploded["block_0"].properties.stride is None
    assert exploded["block_0"].properties.kernel == [3, 3]


def test_sequence_label_used_as_step_raw_text_when_present():
    """A labelled sequence step is named by its own
    operation, not '<parent> (Step N)'. Unlabelled steps keep the fallback."""
    graph = ArchitectureGraph.model_validate(
        {
            "architecture_name": "net",
            "nodes": [
                {
                    "id": "E1",
                    "class": "LAYER",
                    "sub_type": "convolution",
                    "raw_text": "E1",
                    "properties": {
                        "sequence": [
                            {"class": "LAYER", "sub_type": "convolution",
                             "label": "Conv 3x3", "properties": {}},
                            {"class": "MODIFIER", "sub_type": "normalization",
                             "label": "Batch Norm", "properties": {}},
                            {"class": "MODIFIER", "sub_type": "activation",
                             "properties": {}},  # no label -> fallback
                        ]
                    },
                },
            ],
            "edges": [],
        }
    )
    out = sanitize_graph(graph)
    texts = [n.raw_text for n in out.nodes if n.id.startswith("E1_")]
    assert texts == ["Conv 3x3", "Batch Norm", "E1 (Step 3)"]
