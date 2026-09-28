from kg_pipeline.schema.graph import ArchitectureGraph
from kg_pipeline.tools.dictionary_normalizer import (
    load_lookup_table,
    normalize_terms,
    strip_latex,
)


def test_strip_latex():
    assert strip_latex("$3 \\times 3$") == "3 x 3"
    assert strip_latex("$\\alpha$") == "alpha"
    assert strip_latex("$H \\times W \\times C$") == "H x W x C"
    assert strip_latex("plain text") == "plain text"


def test_lookup_table_parses_all_dictionary_sections():
    table = load_lookup_table()

    # Section 1 (5-column): MODIFIER shorthands
    assert table[("activation", "algorithm")]["relu"] == "Relu"
    assert table[("normalization", "algorithm")]["bn"] == "BatchNormalization"
    # LaTeX shorthand resolves too
    assert table[("activation", "algorithm")]["sigma"] == "Sigmoid"
    # Section 2: JUNCTION
    assert table[("multiply", "mechanism")]["matmul"] == "MatMul"
    # Section 3 (6-column, "`name` (Global key)" target): LAYER/MACRO
    assert table[("recurrent", "name")]["lstm"] == "LSTM"
    assert table[("block", "name")]["se block"] == "SqueezeAndExcitation"


def _graph(nodes) -> ArchitectureGraph:
    return ArchitectureGraph.model_validate({"architecture_name": "net", "nodes": nodes})


def test_variant_terms_are_canonicalized():
    graph = _graph(
        [
            {
                "id": "n1",
                "class": "MODIFIER",
                "sub_type": "activation",
                "raw_text": "ReLu",
                "properties": {"algorithm": "ReLu"},
            }
        ]
    )

    result = normalize_terms(graph, load_lookup_table())

    assert result.nodes[0].properties.algorithm == "Relu"


def test_latex_is_stripped_from_value_fields_but_raw_text_kept():
    graph = _graph(
        [
            {
                "id": "n1",
                "class": "LAYER",
                "sub_type": "convolution",
                "raw_text": "$3 \\times 3$ Conv",
                "properties": {"kernel": ["$3$", "$3$"], "symbol": "$\\alpha$"},
            }
        ]
    )

    result = normalize_terms(graph, load_lookup_table())

    assert result.nodes[0].raw_text == "$3 \\times 3$ Conv"
    assert result.nodes[0].properties.kernel == ["3", "3"]
    assert result.nodes[0].properties.symbol == "alpha"


def test_sequence_elements_are_normalized_too():
    graph = _graph(
        [
            {
                "id": "n1",
                "class": "LAYER",
                "sub_type": "convolution",
                "raw_text": "Conv+BN+ReLU",
                "properties": {
                    "sequence": [
                        {"class": "LAYER", "sub_type": "convolution", "properties": {}},
                        {
                            "class": "MODIFIER",
                            "sub_type": "normalization",
                            "properties": {"algorithm": "batch norm"},
                        },
                        {
                            "class": "MODIFIER",
                            "sub_type": "activation",
                            "properties": {"algorithm": "RELU"},
                        },
                    ]
                },
            }
        ]
    )

    result = normalize_terms(graph, load_lookup_table())

    sequence = result.nodes[0].properties.sequence
    assert sequence[1].properties["algorithm"] == "BatchNormalization"
    assert sequence[2].properties["algorithm"] == "Relu"
