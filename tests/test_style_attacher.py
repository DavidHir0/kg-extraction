"""Tests for ``attach_styles`` -- the visual channel's own field."""

from kg_pipeline.tools.sanitizer import sanitize_graph
from kg_pipeline.registry import build_node, build_deps, discover_nodes
from kg_pipeline.schema.graph import ArchitectureGraph, RawTopology
from kg_pipeline.settings import get_settings


def _node():
    discover_nodes()
    return build_node("attach_styles", build_deps(get_settings(), require_keys=False))


def _state(**overrides):
    state = {
        "raw_topology": RawTopology.model_validate(
            {
                "legend_map": {"style_1": "Attention Layer (Orange)"},
                "raw_nodes": [
                    {"id": "a", "raw_text_elements": ["MHSA"], "style_id": "style_1"},
                    {"id": "b", "raw_text_elements": ["Input"]},
                ],
                "raw_edges": [{"source": "a", "target": "b"}],
            }
        ),
        "graph": ArchitectureGraph.model_validate(
            {
                "architecture_name": "net",
                "nodes": [
                    {"id": "a", "class": "LAYER", "sub_type": "attention", "raw_text": "MHSA"},
                    {"id": "b", "class": "DATA", "sub_type": "input", "raw_text": "Input"},
                ],
                "edges": [{"source": "a", "target": "b", "role": "forward"}],
            }
        ),
    }
    state.update(overrides)
    return state


def test_style_id_is_copied_onto_the_matching_node():
    result = _node()(_state())
    nodes = {n.id: n for n in result["graph"].nodes}
    assert nodes["a"].style_id == "style_1"
    assert nodes["b"].style_id is None  # unstyled shapes stay unstyled


def test_exploded_substep_inherits_the_parent_style():
    """``sanitize_graph`` appends ``_<i>``; the join must still resolve."""
    state = _state(
        graph=ArchitectureGraph.model_validate(
            {
                "architecture_name": "net",
                "nodes": [
                    {
                        "id": "a_0",
                        "class": "LAYER",
                        "sub_type": "attention",
                        "raw_text": "MHSA (Step 1)",
                    }
                ],
                "edges": [],
            }
        )
    )
    result = _node()(state)
    assert result["graph"].nodes[0].style_id == "style_1"


def test_explosion_carries_style_to_every_step():
    graph = ArchitectureGraph.model_validate(
        {
            "architecture_name": "net",
            "nodes": [
                {
                    "id": "fused",
                    "class": "JUNCTION",
                    "sub_type": "add",
                    "raw_text": "Add & Norm",
                    "style_id": "style_1",
                    "properties": {
                        "sequence": [
                            {"class": "JUNCTION", "sub_type": "add"},
                            {"class": "MODIFIER", "sub_type": "normalization"},
                        ]
                    },
                }
            ],
            "edges": [],
        }
    )
    exploded = sanitize_graph(graph)
    assert [n.style_id for n in exploded.nodes] == ["style_1", "style_1"]


def test_no_styles_and_missing_state_are_no_ops():
    bare = _state(
        raw_topology=RawTopology.model_validate(
            {"raw_nodes": [{"id": "a", "raw_text_elements": ["MHSA"]}], "raw_edges": []}
        )
    )
    assert _node()(bare) == {}
    assert _node()({"graph": None, "raw_topology": None}) == {}


def test_an_existing_style_is_never_overwritten():
    state = _state()
    state["graph"].nodes[0].style_id = "already_set"
    result = _node()(state)
    assert result["graph"].nodes[0].style_id == "already_set"
