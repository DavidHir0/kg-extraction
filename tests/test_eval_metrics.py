"""Metric definitions and dataset aggregation, independent of any matcher:
these tests hand-build the MatchResult so only the counting is under test."""

import glob
import json
import os
import shutil

from kg_pipeline.eval.evaluate import evaluate_run
from kg_pipeline.eval.matching import MatchedPair, MatchResult
from kg_pipeline.eval.metrics import aggregate, figure_metrics, missing_figure_metrics, prf
from kg_pipeline.schema.graph import ArchitectureGraph

GOLDEN_DIR = os.path.join("data", "golden_dataset", "ground_truth")


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


def pairs(*id_pairs) -> MatchResult:
    return MatchResult(
        pairs=[MatchedPair(p, g, 1.0, {}) for p, g in id_pairs],
        unmatched_pred=[],
        unmatched_gold=[],
    )


def test_prf_math():
    # Vacuously empty scores perfect, not zero.
    assert prf(0, 0, 0) == {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert prf(0, 1, 0)["f1"] == 0.0
    result = prf(2, 1, 1)
    assert result["precision"] == 2 / 3
    assert result["recall"] == 2 / 3
    assert result["f1"] == 2 / 3


def test_role_flip_hits_strict_but_not_relaxed():
    gold = graph(
        [n("a", raw="A"), n("b", raw="B"), n("c", raw="C")],
        [("a", "forward", "b"), ("a", "skip", "c")],
    )
    pred = graph(
        [n("pa", raw="A"), n("pb", raw="B"), n("pc", raw="C")],
        [("pa", "forward", "pb"), ("pa", "forward", "pc")],  # skip flipped to forward
    )

    metrics = figure_metrics(pred, gold, pairs(("pa", "a"), ("pb", "b"), ("pc", "c")))

    assert metrics["triplets"]["strict"]["f1"] == 0.5  # 1 TP, 1 FP, 1 FN
    assert metrics["triplets"]["relaxed"]["f1"] == 1.0


def test_edge_with_unmatched_endpoint_is_false_positive():
    gold = graph([n("a", raw="A"), n("b", raw="B")], [("a", "forward", "b")])
    pred = graph(
        [n("pa", raw="A"), n("pb", raw="B"), n("px", raw="ghost")],
        [("pa", "forward", "pb"), ("pa", "forward", "px")],
    )
    result = MatchResult(
        pairs=[MatchedPair("pa", "a", 1.0, {}), MatchedPair("pb", "b", 1.0, {})],
        unmatched_pred=["px"],
        unmatched_gold=[],
    )

    metrics = figure_metrics(pred, gold, result)

    assert metrics["triplets"]["strict"]["tp"] == 1
    assert metrics["triplets"]["strict"]["fp"] == 1
    assert metrics["triplets"]["relaxed"]["fp"] == 1


def test_property_scoring_on_matched_nodes():
    gold = graph([n("a", kernel=["3", "3"], channels=64)])
    pred = graph([n("pa", kernel=["3", "3"], channels=128)])  # channels wrong

    metrics = figure_metrics(pred, gold, pairs(("pa", "a")))

    # kernel[0], kernel[1] agree; channels disagrees on both sides.
    assert metrics["properties"]["tp"] == 2
    assert metrics["properties"]["fp"] == 1
    assert metrics["properties"]["fn"] == 1


def test_note_and_name_are_not_scored_as_properties():
    gold = graph([n("a", raw="", note="implicit", name="Add")])
    pred = graph([n("pa", raw="")])

    metrics = figure_metrics(pred, gold, pairs(("pa", "a")))

    assert metrics["properties"] == prf(0, 0, 0)


def test_fused_vs_exploded_granularity_mismatch():
    # Golden exploded a "Conv+BN+ReLU" box into 3 nodes; the pipeline kept one
    # fused node. Accepted behavior: 1 match + 2 golden FNs.
    gold = graph(
        [n("g0", raw="Conv+BN+ReLU (Step 1)"), n("g1", raw="Conv+BN+ReLU (Step 2)"), n("g2", raw="Conv+BN+ReLU (Step 3)")],
        [("g0", "forward", "g1"), ("g1", "forward", "g2")],
    )
    pred = graph([n("p0", raw="Conv+BN+ReLU")])
    result = MatchResult(
        pairs=[MatchedPair("p0", "g0", 0.9, {})],
        unmatched_pred=[],
        unmatched_gold=["g1", "g2"],
    )

    metrics = figure_metrics(pred, gold, result)

    assert metrics["nodes"]["tp"] == 1
    assert metrics["nodes"]["fn"] == 2


def test_missing_figure_is_all_false_negatives():
    gold = graph(
        [n("a", cls="DATA", sub="input", raw="x", dimensions=["224", "224"]), n("b", raw="Conv")],
        [("a", "forward", "b")],
    )

    metrics = missing_figure_metrics(gold)

    assert metrics["nodes"] == prf(0, 0, 2)
    assert metrics["triplets"]["strict"] == prf(0, 0, 1)
    assert metrics["properties"]["fn"] == 2  # dimensions[0], dimensions[1]
    assert metrics["missing_prediction"] is True


def test_role_confusion_and_error_analysis():
    from kg_pipeline.eval.analysis import analyze_figures
    from kg_pipeline.eval.report import match_audit
    from kg_pipeline.eval.matching import MatchResult, MatchedPair

    gold = graph(
        [n("a", raw="A"), n("b", raw="B"), n("c", raw="C", channels=64)],
        [("a", "forward", "b"), ("a", "skip", "c")],
    )
    pred = graph(
        [n("pa", raw="A"), n("pb", raw="B", cls="MODIFIER", sub="activation"), n("pc", raw="C", channels=32)],
        [("pa", "forward", "pb"), ("pa", "forward", "pc")],
    )
    result = MatchResult(
        pairs=[MatchedPair(p, g, 1.0, {}) for p, g in (("pa", "a"), ("pb", "b"), ("pc", "c"))],
        unmatched_pred=[],
        unmatched_gold=[],
    )
    metrics = figure_metrics(pred, gold, result)
    metrics["match"] = match_audit(result, pred, gold)

    assert metrics["role_confusion"] == {"forward": {"forward": 1}, "skip": {"forward": 1}}
    assert metrics["property_errors"]["channels"] == {"tp": 0, "fp": 1, "fn": 1}

    analysis = analyze_figures({"fig": metrics})
    assert analysis["role_confusion"]["skip"]["forward"] == 1
    assert analysis["class_confusion"]["LAYER"] == {"LAYER": 2, "MODIFIER": 1}
    assert analysis["property_errors"]["channels"]["recall"] == 0.0
    assert analysis["figures_ranked_worst_first"][0]["figure"] == "fig"


def test_aggregate_micro_pools_counts_and_macro_averages():
    gold = graph([n("a", raw="A"), n("b", raw="B")], [("a", "forward", "b")])
    perfect = figure_metrics(
        graph([n("pa", raw="A"), n("pb", raw="B")], [("pa", "forward", "pb")]),
        gold,
        pairs(("pa", "a"), ("pb", "b")),
    )
    empty = missing_figure_metrics(gold)

    summary = aggregate({"fig1": perfect, "fig2": empty})

    assert summary["figures"] == 2
    assert summary["missing_predictions"] == 1
    # micro: 2 TP + 2 FN nodes -> P=1, R=0.5
    assert summary["micro"]["nodes"]["precision"] == 1.0
    assert summary["micro"]["nodes"]["recall"] == 0.5
    # macro: mean of per-figure F1s (1.0 and 0.0)
    assert summary["macro"]["nodes_f1"] == 0.5
    assert summary["macro"]["triplets_strict_f1"] == 0.5


def test_evaluate_run_end_to_end(tmp_path):
    """Two real goldens as their own predictions (one missing) through the
    full evaluate_run + report-writing path."""
    golden_paths = sorted(glob.glob(os.path.join(GOLDEN_DIR, "*.json")))[:2]
    assert len(golden_paths) == 2, f"golden dataset not found at {GOLDEN_DIR}"

    gold_dir = tmp_path / "gold"
    pred_dir = tmp_path / "run" / "final_graphs"
    gold_dir.mkdir()
    pred_dir.mkdir(parents=True)
    for path in golden_paths:
        shutil.copy(path, gold_dir / os.path.basename(path))
    shutil.copy(golden_paths[0], pred_dir / os.path.basename(golden_paths[0]))

    out_dir = tmp_path / "eval"
    summaries = evaluate_run(
        str(tmp_path / "run"),
        golden_dir=str(gold_dir),
        matchers=("propagation", "anchor"),
        out_dir=str(out_dir),
        run_meta={"vision model": "test"},
    )

    for name in ("propagation", "anchor"):
        summary = summaries[name]
        assert summary["figures"] == 2
        assert summary["missing_predictions"] == 1
        # The present figure is a verbatim copy: perfect scores on it.
        assert summary["micro"]["nodes"]["precision"] == 1.0
        assert summary["macro"]["nodes_f1"] == 0.5

    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "matcher_disagreement.json").is_file()
    assert (out_dir / "error_analysis.json").is_file()
    report = (out_dir / "REPORT.md").read_text()
    assert "vision model" in report and "Headline metrics" in report
    stem = os.path.splitext(os.path.basename(golden_paths[0]))[0]
    audit = json.loads((out_dir / "figures" / "propagation" / f"{stem}.json").read_text())
    assert audit["match"]["pairs"], "audit must list matched pairs with signals"
    assert {"class", "sub_type", "raw_text"} <= set(audit["match"]["pairs"][0]["pred"])


def test_conditional_wiring_metric_ignores_unmatched_endpoints():
    """The conditional metric scores wiring only among matched nodes: a missed
    node's edges must not count against it (but do hit strict/relaxed)."""
    from kg_pipeline.eval.matching import MatchedPair, MatchResult
    from kg_pipeline.eval.metrics import figure_metrics
    from kg_pipeline.schema.graph import ArchitectureGraph

    def g(nodes, edges):
        return ArchitectureGraph.model_validate({
            "nodes": [{"id": i, "class": "LAYER", "sub_type": "convolution", "raw_text": i}
                      for i in nodes],
            "edges": [{"source": s, "target": t, "role": "forward", "properties": {}}
                      for s, t in edges],
        })

    gold = g(["a", "b", "c"], [("a", "b"), ("b", "c")])
    pred = g(["pa", "pb"], [("pa", "pb")])  # node c missed entirely
    result = MatchResult(
        pairs=[MatchedPair("pa", "a", 1.0, {}), MatchedPair("pb", "b", 1.0, {})],
        unmatched_pred=[], unmatched_gold=["c"],
    )

    m = figure_metrics(pred, gold, result)

    # Strict: 1 tp, 1 fn (b->c lost with node c). Conditional: perfect wiring.
    assert m["triplets"]["strict"]["tp"] == 1
    assert m["triplets"]["strict"]["fn"] == 1
    assert m["triplets"]["conditional"]["f1"] == 1.0
