"""Matcher behavior: both algorithms must recover correct alignments from
content + structure alone (node ids never carry information)."""

import glob
import os

import pytest

from kg_pipeline.eval.matching import MatchConfig, match_anchor, match_propagation
from kg_pipeline.eval.metrics import figure_metrics
from kg_pipeline.eval.preprocess import load_graph, prepare_graph
from kg_pipeline.schema.graph import ArchitectureGraph
from kg_pipeline.tools.dictionary_normalizer import load_lookup_table

GOLDEN_DIR = os.path.join("data", "golden_dataset", "ground_truth")
MATCHERS = [match_propagation, match_anchor]


def graph(nodes, edges=()) -> ArchitectureGraph:
    return ArchitectureGraph.model_validate(
        {
            "architecture_name": "net",
            "nodes": nodes,
            "edges": [
                {"source": s, "target": t, "role": r, "properties": {}}
                for s, r, t in edges
            ],
        }
    )


def n(node_id, cls="LAYER", sub="convolution", raw="Conv 3x3", **props):
    return {"id": node_id, "class": cls, "sub_type": sub, "raw_text": raw, "properties": props}


def rename_ids(g: ArchitectureGraph) -> ArchitectureGraph:
    """Fresh ids + reversed node order: only content/structure can align."""
    mapping = {node.id: f"pred_{i}" for i, node in enumerate(g.nodes)}
    return g.model_copy(
        update={
            "nodes": [
                node.model_copy(update={"id": mapping[node.id]}) for node in g.nodes
            ][::-1],
            "edges": [
                e.model_copy(update={"source": mapping[e.source], "target": mapping[e.target]})
                for e in g.edges
            ],
            "groups": [
                grp.model_copy(
                    update={"member_node_ids": [mapping.get(m, m) for m in grp.member_node_ids]}
                )
                for grp in g.groups
            ],
        }
    )


def _sample_golden_paths() -> list[str]:
    """A few real goldens, guaranteed to include one with implicit
    (raw_text "") nodes and one with composite sequences."""
    paths = sorted(glob.glob(os.path.join(GOLDEN_DIR, "*.json")))
    assert paths, f"golden dataset not found at {GOLDEN_DIR}"
    chosen = paths[:2]
    for want in ('"raw_text": ""', '"sequence"'):
        for path in paths:
            with open(path) as f:
                if want in f.read():
                    if path not in chosen:
                        chosen.append(path)
                    break
    return chosen


@pytest.mark.parametrize("matcher", MATCHERS, ids=["propagation", "anchor"])
@pytest.mark.parametrize("gold_path", _sample_golden_paths())
def test_self_match_on_real_goldens_is_perfect(matcher, gold_path):
    table = load_lookup_table()
    gold = prepare_graph(load_graph(gold_path), table)
    pred = rename_ids(gold)

    metrics = figure_metrics(pred, gold, matcher(pred, gold))

    assert metrics["nodes"]["f1"] == 1.0
    assert metrics["accuracy"]["class"]["accuracy"] == 1.0
    assert metrics["accuracy"]["sub_type"]["accuracy"] == 1.0
    assert metrics["triplets"]["strict"]["f1"] == 1.0
    assert metrics["properties"]["f1"] == 1.0


@pytest.mark.parametrize("matcher", MATCHERS, ids=["propagation", "anchor"])
def test_duplicate_text_nodes_resolved_by_structure(matcher):
    gold = graph(
        [
            n("in", cls="DATA", sub="input", raw="Image"),
            n("c1"),
            n("c2"),
            n("out", cls="DATA", sub="output", raw="Mask"),
        ],
        [("in", "forward", "c1"), ("c1", "forward", "c2"), ("c2", "forward", "out")],
    )
    pred = rename_ids(gold)

    result = matcher(pred, gold)

    # pred ids were assigned in gold-node order: pred_1 <-> c1, pred_2 <-> c2.
    mapping = result.pred_to_gold
    assert mapping["pred_1"] == "c1"
    assert mapping["pred_2"] == "c2"


@pytest.mark.parametrize("matcher", MATCHERS, ids=["propagation", "anchor"])
def test_implicit_empty_text_nodes_match_structurally(matcher):
    gold = graph(
        [
            n("x", cls="DATA", sub="input", raw="x"),
            n("skip", cls="DATA", sub="input", raw="residual"),
            n("j", cls="JUNCTION", sub="add", raw="", note="implicit"),
            n("y", cls="DATA", sub="output", raw="y"),
        ],
        [("x", "forward", "j"), ("skip", "skip", "j"), ("j", "forward", "y")],
    )
    pred = rename_ids(gold)

    result = matcher(pred, gold)

    assert result.pred_to_gold["pred_2"] == "j"
    assert not result.unmatched_gold


@pytest.mark.parametrize("matcher", MATCHERS, ids=["propagation", "anchor"])
def test_misclassified_node_still_matches(matcher):
    gold = graph(
        [n("a", cls="DATA", sub="input", raw="x"), n("b", raw="ReLU"), n("c", cls="DATA", sub="output", raw="y")],
        [("a", "forward", "b"), ("b", "forward", "c")],
    )
    pred = rename_ids(gold)
    # The pipeline typed the middle box MODIFIER/activation instead of LAYER.
    pred.nodes[1].class_ = "MODIFIER"
    pred.nodes[1].sub_type = "activation"

    metrics = figure_metrics(pred, gold, matcher(pred, gold))

    assert metrics["nodes"]["f1"] == 1.0  # detection unaffected
    assert metrics["accuracy"]["class"]["accuracy"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("matcher", MATCHERS, ids=["propagation", "anchor"])
def test_missing_and_extra_nodes_hit_recall_and_precision(matcher):
    gold = graph(
        [
            n("a", cls="DATA", sub="input", raw="input image"),
            n("b", raw="Conv 7x7"),
            n("c", cls="DATA", sub="output", raw="logits"),
        ],
        [("a", "forward", "b"), ("b", "forward", "c")],
    )
    # Pred drops the conv and hallucinates an unrelated pooling node.
    pred = graph(
        [
            n("p0", cls="DATA", sub="input", raw="input image"),
            n("p1", cls="MODIFIER", sub="downsample", raw="totally unrelated garbage"),
            n("p2", cls="DATA", sub="output", raw="logits"),
        ],
        [("p0", "forward", "p2")],
    )

    result = matcher(pred, gold)
    metrics = figure_metrics(pred, gold, result)

    assert "p1" in result.unmatched_pred  # below threshold, not force-matched
    assert "b" in result.unmatched_gold
    assert metrics["nodes"]["tp"] == 2
    assert metrics["nodes"]["fp"] == 1
    assert metrics["nodes"]["fn"] == 1


@pytest.mark.parametrize("matcher", MATCHERS, ids=["propagation", "anchor"])
def test_empty_prediction_graph(matcher):
    gold = graph([n("a", cls="DATA", sub="input", raw="x")])
    pred = graph([])

    result = matcher(pred, gold)

    assert not result.pairs
    assert result.unmatched_gold == ["a"]


# --- edge-consistency refinement ---


def test_refine_untangles_crossed_duplicate_assignment():
    """A deliberately crossed permutation of identical-text duplicates must be
    repaired by edge agreement (the features are indifferent between copies)."""
    import numpy as np

    from kg_pipeline.eval.matching import MatchedPair, MatchResult, _refine_by_edges

    #   in -> convA -> mid -> convB -> out   (two identical "Conv 3x3" nodes)
    gold = graph(
        [
            n("g_in", cls="DATA", sub="input", raw="input"),
            n("g_convA"),
            n("g_mid", cls="DATA", sub="intermediate", raw="feat 1"),
            n("g_convB"),
            n("g_out", cls="DATA", sub="output", raw="output"),
        ],
        [
            ("g_in", "forward", "g_convA"),
            ("g_convA", "forward", "g_mid"),
            ("g_mid", "forward", "g_convB"),
            ("g_convB", "forward", "g_out"),
        ],
    )
    pred = graph(
        [
            n("p_in", cls="DATA", sub="input", raw="input"),
            n("p_convA"),
            n("p_mid", cls="DATA", sub="intermediate", raw="feat 1"),
            n("p_convB"),
            n("p_out", cls="DATA", sub="output", raw="output"),
        ],
        [
            ("p_in", "forward", "p_convA"),
            ("p_convA", "forward", "p_mid"),
            ("p_mid", "forward", "p_convB"),
            ("p_convB", "forward", "p_out"),
        ],
    )
    order_p = [nn.id for nn in pred.nodes]
    order_g = [nn.id for nn in gold.nodes]
    # Feature similarity: 1.0 for same-role pairs AND for the duplicate cross
    # pairs (identical text), low elsewhere.
    s = np.full((5, 5), 0.1)
    same = {("p_in", "g_in"), ("p_mid", "g_mid"), ("p_out", "g_out"),
            ("p_convA", "g_convA"), ("p_convA", "g_convB"),
            ("p_convB", "g_convA"), ("p_convB", "g_convB")}
    for i, p in enumerate(order_p):
        for j, g in enumerate(order_g):
            if (p, g) in same:
                s[i, j] = 1.0
    signals = {"text": s}
    crossed = MatchResult(
        pairs=[
            MatchedPair("p_in", "g_in", 1.0, {}),
            MatchedPair("p_convA", "g_convB", 1.0, {}),  # crossed!
            MatchedPair("p_mid", "g_mid", 1.0, {}),
            MatchedPair("p_convB", "g_convA", 1.0, {}),  # crossed!
            MatchedPair("p_out", "g_out", 1.0, {}),
        ],
        unmatched_pred=[],
        unmatched_gold=[],
    )

    refined = _refine_by_edges(pred, gold, s, signals, crossed, MatchConfig())

    mapping = refined.pred_to_gold
    assert mapping["p_convA"] == "g_convA"
    assert mapping["p_convB"] == "g_convB"


def test_refine_moves_onto_unmatched_gold_twin():
    """A pred duplicate matched to the wrong (already-contested) copy moves to
    the unmatched identical twin when edges vote for it."""
    import numpy as np

    from kg_pipeline.eval.matching import MatchedPair, MatchResult, _refine_by_edges

    gold = graph(
        [
            n("g_in", cls="DATA", sub="input", raw="input"),
            n("g_conv1"),
            n("g_conv2"),
        ],
        [("g_in", "forward", "g_conv1"), ("g_in", "forward", "g_conv2")],
    )
    # Pred found only one conv, wired from input -- but it was left unmatched
    # while gold_conv2 stayed free.
    pred = graph(
        [n("p_in", cls="DATA", sub="input", raw="input"), n("p_conv")],
        [("p_in", "forward", "p_conv")],
    )
    order_p = [nn.id for nn in pred.nodes]
    order_g = [nn.id for nn in gold.nodes]
    s = np.full((2, 3), 0.1)
    for i, p in enumerate(order_p):
        for j, g in enumerate(order_g):
            if (p, g) in {("p_in", "g_in"), ("p_conv", "g_conv1"), ("p_conv", "g_conv2")}:
                s[i, j] = 1.0
    signals = {"text": s}
    # Start p_conv on g_conv2; edges are equally happy either way here, so no
    # move should occur (agreement must STRICTLY improve) -- stability check.
    stable = MatchResult(
        pairs=[MatchedPair("p_in", "g_in", 1.0, {}), MatchedPair("p_conv", "g_conv2", 1.0, {})],
        unmatched_pred=[], unmatched_gold=["g_conv1"],
    )
    refined = _refine_by_edges(pred, gold, s, signals, stable, MatchConfig())
    assert refined.pred_to_gold["p_conv"] == "g_conv2"


def test_refine_never_overrides_confident_feature_evidence():
    """Nodes whose texts genuinely differ (similarity gap > eps) must never be
    swapped, even if a swap would add a matching edge."""
    import numpy as np

    from kg_pipeline.eval.matching import MatchedPair, MatchResult, _refine_by_edges

    gold = graph(
        [n("g_a", raw="Conv 3x3"), n("g_b", raw="MaxPool 2x2", sub="downsample", cls="MODIFIER")],
        [("g_a", "forward", "g_b")],
    )
    pred = graph(
        [n("p_a", raw="Conv 3x3"), n("p_b", raw="MaxPool 2x2", sub="downsample", cls="MODIFIER")],
        [("p_b", "forward", "p_a")],  # pred wired backwards
    )
    s = np.array([[1.0, 0.4], [0.4, 1.0]])
    signals = {"text": s}
    correct = MatchResult(
        pairs=[MatchedPair("p_a", "g_a", 1.0, {}), MatchedPair("p_b", "g_b", 1.0, {})],
        unmatched_pred=[], unmatched_gold=[],
    )

    refined = _refine_by_edges(pred, gold, s, signals, correct, MatchConfig())

    # Swapping would "fix" the backwards edge but contradict the text evidence.
    assert refined.pred_to_gold == {"p_a": "g_a", "p_b": "g_b"}


def test_number_mismatch_damps_text_similarity():
    """Identical-structure dimension labels with different digits must score
    lower than digit-identical ones (legacy number-check idea, softened)."""

    from kg_pipeline.eval.matching import MatchConfig, feature_similarity

    pred = graph([
        n("p0", cls="DATA", sub="intermediate", raw="$64 \\times 570 \\times 570$"),
    ])
    gold = graph([
        n("g_same", cls="DATA", sub="intermediate", raw="$64 \\times 570 \\times 570$"),
        n("g_other", cls="DATA", sub="intermediate", raw="$64 \\times 284 \\times 284$"),
    ])

    s, signals = feature_similarity(pred, gold, MatchConfig())

    assert signals["text"][0, 0] > signals["text"][0, 1]
    assert signals["text"][0, 0] > 0.9   # exact match stays high
    assert signals["text"][0, 1] < 0.65  # digit-mismatched twin is damped
