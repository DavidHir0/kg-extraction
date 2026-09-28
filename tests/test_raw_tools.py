"""Tests for the deterministic tools operating on the raw topology."""

from kg_pipeline.schema.graph import RawEdge, RawTopology
from kg_pipeline.tools.bypass_pruner import prune_bypasses
from kg_pipeline.tools.legend_resolver import resolve_legend


def _topology(**overrides) -> RawTopology:
    payload = {
        "raw_nodes": [
            {"id": "a", "raw_text_elements": ["Input"]},
            {"id": "empty", "raw_text_elements": []},
            {"id": "b", "raw_text_elements": ["Conv"]},
        ],
        "raw_edges": [
            {"source": "a", "target": "empty"},
            {"source": "empty", "target": "b"},
        ],
    }
    payload.update(overrides)
    return RawTopology.model_validate(payload)


def test_empty_passthrough_node_is_pruned_and_edge_bridged():
    pruned = prune_bypasses(_topology())

    assert [n.id for n in pruned.raw_nodes] == ["a", "b"]
    assert [(e.source, e.target) for e in pruned.raw_edges] == [("a", "b")]


def test_chain_of_empty_nodes_collapses_to_single_edge():
    pruned = prune_bypasses(
        _topology(
            raw_nodes=[
                {"id": "a", "raw_text_elements": ["Input"]},
                {"id": "e1", "raw_text_elements": []},
                {"id": "e2", "raw_text_elements": []},
                {"id": "b", "raw_text_elements": ["Conv"]},
            ],
            raw_edges=[
                {"source": "a", "target": "e1"},
                {"source": "e1", "target": "e2"},
                {"source": "e2", "target": "b"},
            ],
        )
    )

    assert [n.id for n in pruned.raw_nodes] == ["a", "b"]
    assert [(e.source, e.target) for e in pruned.raw_edges] == [("a", "b")]


def test_junctions_endpoints_and_styled_nodes_survive():
    topology = _topology(
        raw_nodes=[
            {"id": "a", "raw_text_elements": ["A"]},
            {"id": "b", "raw_text_elements": ["B"]},
            {"id": "merge", "raw_text_elements": []},  # 2-in/1-out: junction
            {"id": "styled", "raw_text_elements": [], "style_id": "style_1"},
            {"id": "sink", "raw_text_elements": []},  # 1-in/0-out: endpoint
        ],
        raw_edges=[
            {"source": "a", "target": "merge"},
            {"source": "b", "target": "merge"},
            {"source": "merge", "target": "styled"},
            {"source": "styled", "target": "sink"},
        ],
    )

    pruned = prune_bypasses(topology)

    assert {n.id for n in pruned.raw_nodes} == {"a", "b", "merge", "styled", "sink"}


def test_isolated_empty_node_is_dropped_but_isolated_text_survives():
    topology = _topology(
        raw_nodes=[
            {"id": "a", "raw_text_elements": ["Input"]},
            {"id": "b", "raw_text_elements": ["Conv"]},
            {"id": "icon", "raw_text_elements": []},  # stray pictogram: drop
            {"id": "caption", "raw_text_elements": ["Encoder"]},  # keep: has text
            {"id": "styled", "raw_text_elements": [], "style_id": "s1"},  # keep: legend
        ],
        raw_edges=[{"source": "a", "target": "b"}],
    )

    pruned = prune_bypasses(topology)

    assert {n.id for n in pruned.raw_nodes} == {"a", "b", "caption", "styled"}


def test_pruned_nodes_are_dropped_from_group_membership():
    topology = _topology(
        raw_groups=[
            {
                "id": "g1",
                "label": "Stem",
                "boundary_type": "explicit_box",
                "member_node_ids": ["a", "empty", "b"],
            }
        ]
    )

    pruned = prune_bypasses(topology)

    assert pruned.raw_groups[0].member_node_ids == ["a", "b"]


def test_legend_text_appended_to_styled_nodes():
    topology = _topology(
        legend_map={"style_1": "Conv Layer"},
        raw_nodes=[
            {"id": "a", "raw_text_elements": [], "style_id": "style_1"},
            {"id": "b", "raw_text_elements": ["Pool"]},
        ],
        raw_edges=[],
    )

    resolved = resolve_legend(topology)

    assert resolved.raw_nodes[0].raw_text_elements == ["Conv Layer"]
    assert resolved.raw_nodes[1].raw_text_elements == ["Pool"]


def test_legend_resolution_is_idempotent():
    topology = _topology(
        legend_map={"style_1": "Conv Layer"},
        raw_nodes=[{"id": "a", "raw_text_elements": ["Conv Layer"], "style_id": "style_1"}],
        raw_edges=[],
    )

    resolved = resolve_legend(topology)

    assert resolved.raw_nodes[0].raw_text_elements == ["Conv Layer"]


# --- the vision model's output schema ---


def test_raw_edge_keeps_the_benchmarked_shape():
    """`RawEdge` is part of the schema the vision model is asked to fill, so its
    fields are pinned to the shape the benchmark numbers were measured with."""
    assert "style_id" in RawEdge.model_fields
    assert RawEdge(source="a", target="b").style == "solid"
    assert RawEdge(source="a", target="b", style="dashed").style == "dashed"


def test_raw_edge_carries_no_docstring():
    """Pydantic copies a model's docstring into `model_json_schema()`, and
    `with_structured_output` sends that schema to the vision model. Notes about
    this class belong in `#` comments, which never reach the model."""
    assert not (RawEdge.__doc__ or "").strip(), (
        "RawEdge has a docstring; it will be sent to the vision model. "
        "Use a `#` comment instead."
    )
    assert "description" not in RawEdge.model_json_schema()


def test_legend_resolution_never_touches_edges():
    """resolve_legend is node-side only: no A -> Op -> B rewiring."""
    topo = _topology(
        legend_map={"style_pink": "MaxPool 2x2", "style_dash": "Skip Connection"},
        raw_nodes=[
            {"id": "a", "raw_text_elements": ["E1"]},
            {"id": "b", "raw_text_elements": ["E2"]},
        ],
        raw_edges=[
            {"source": "a", "target": "b", "style": "solid"},
            {"source": "a", "target": "b", "style": "dashed"},
        ],
    )
    resolved = resolve_legend(topo)
    assert len(resolved.raw_nodes) == 2, "no operation node may be minted here"
    assert [e.style for e in resolved.raw_edges] == [
        "solid",
        "dashed",
    ], "edges pass through untouched, style preserved"
    assert all(e.style_id is None for e in resolved.raw_edges), (
        "resolve_legend must never populate RawEdge.style_id"
    )


def test_monolith_carries_the_legend_arrow_rule():
    """Pins the vision prompt's legend-arrow rule, so an edit cannot drop it silently."""
    from kg_pipeline.prompts.loader import load_agent_prompt

    text = load_agent_prompt("vision_extractor")
    assert "Legend-Styled Arrows ARE Operations" in text
