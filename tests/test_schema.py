from kg_pipeline.schema.graph import ArchitectureGraph, RawTopology


def test_raw_topology_defaults_to_empty():
    raw = RawTopology.model_validate({})
    assert raw.raw_nodes == []
    assert raw.legend_map == {}


def test_raw_topology_round_trip():
    payload = {
        "legend_map": {"style_1": "Conv"},
        "raw_nodes": [{"id": "node_0", "raw_text_elements": ["Conv"], "style_id": "style_1"}],
        "raw_edges": [{"source": "node_0", "target": "node_0", "style": "solid"}],
        "raw_groups": [],
    }
    raw = RawTopology.model_validate(payload)
    assert raw.raw_nodes[0].style_id == "style_1"


def test_architecture_graph_round_trip_uses_class_alias():
    payload = {
        "architecture_name": "test_net",
        "nodes": [
            {"id": "n1", "class": "LAYER", "sub_type": "convolution", "raw_text": "3x3 Conv", "properties": {"kernel": [3, 3]}},
        ],
        "edges": [
            {"source": "n1", "target": "n1", "role": "forward", "properties": {"labels": []}},
        ],
    }

    graph = ArchitectureGraph.model_validate(payload)
    assert graph.nodes[0].class_ == "LAYER"
    assert graph.nodes[0].properties.kernel == [3, 3]

    dumped = graph.model_dump(by_alias=True, exclude_none=True)
    assert dumped["nodes"][0]["class"] == "LAYER"
    assert "class_" not in dumped["nodes"][0]
    # exclude_none drops unset optional properties instead of emitting nulls
    assert "dtype" not in dumped["nodes"][0]["properties"]


def test_symbolic_and_unknown_property_values_validate():
    # Regression: benchmark run_122b lost 3 figures to these exact payloads --
    # symbolic group counts ("C", "k") and a null for an unbound dimension
    # axis (R^{N x D} -> [null, "D"]) are legitimate normalizer output.
    payload = {
        "nodes": [
            {
                "id": "n1",
                "class": "LAYER",
                "sub_type": "convolution",
                "raw_text": "Shared Conv1D",
                "properties": {"kernel": [1, 1], "channels": "C", "groups": "C"},
            },
            {
                "id": "n2",
                "class": "DATA",
                "sub_type": "input",
                "raw_text": "X in R^{N x D}",
                "properties": {"dimensions": [None, "D"]},
            },
        ],
    }

    graph = ArchitectureGraph.model_validate(payload)
    assert graph.nodes[0].properties.groups == "C"
    assert graph.nodes[1].properties.dimensions == [None, "D"]
