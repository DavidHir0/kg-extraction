"""Serializes evaluation results to disk and formats the console summary.

Layout under the output directory (one sibling set per matcher when several
are compared):

- ``summary.json``                    -- aggregate metrics per matcher
- ``figures/<matcher>/<figure>.json`` -- per-figure metrics + match audit
- ``matcher_disagreement.json``       -- only when >1 matcher ran
"""

import json
import os

from kg_pipeline.eval.matching import MatchResult
from kg_pipeline.schema.graph import ArchitectureGraph, NodeModel


def _node_info(node: NodeModel) -> dict:
    return {
        "id": node.id,
        "class": node.class_,
        "sub_type": node.sub_type,
        "raw_text": node.raw_text,
    }


def match_audit(result: MatchResult, pred: ArchitectureGraph, gold: ArchitectureGraph) -> dict:
    """Who matched whom, with enough node content to analyze errors without
    reopening the graph files."""
    p_by_id = {n.id: n for n in pred.nodes}
    g_by_id = {n.id: n for n in gold.nodes}
    return {
        "pairs": [
            {
                "pred": _node_info(p_by_id[p.pred_id]),
                "gold": _node_info(g_by_id[p.gold_id]),
                "score": round(p.score, 4),
                "signals": {k: round(v, 4) for k, v in p.signals.items()},
            }
            for p in sorted(result.pairs, key=lambda p: -p.score)
        ],
        "unmatched_pred": [_node_info(p_by_id[i]) for i in result.unmatched_pred],
        "unmatched_gold": [_node_info(g_by_id[i]) for i in result.unmatched_gold],
    }


def matcher_disagreement(
    matches: dict[str, dict[str, MatchResult]], summaries: dict[str, dict]
) -> dict:
    """Per figure: pred nodes the matchers assign differently, plus the
    headline metric deltas between matchers."""
    names = sorted(matches)
    figures = {}
    for figure in sorted(set().union(*(matches[n].keys() for n in names))):
        mappings = {n: matches[n][figure].pred_to_gold for n in names if figure in matches[n]}
        pred_ids = sorted(set().union(*(m.keys() for m in mappings.values())))
        diffs = [
            {"pred_id": pid, **{n: mappings[n].get(pid) for n in names}}
            for pid in pred_ids
            if len({mappings[n].get(pid) for n in names}) > 1
        ]
        if diffs:
            figures[figure] = diffs

    return {
        "matchers": names,
        "figures_with_disagreement": len(figures),
        "summary_deltas": {
            metric: {n: summaries[n]["macro"][metric] for n in names}
            for metric in ("nodes_f1", "triplets_strict_f1", "triplets_relaxed_f1")
        },
        "figures": figures,
    }


def write_reports(
    out_dir: str,
    figures: dict[str, dict[str, dict]],
    summaries: dict[str, dict],
    disagreement: dict | None,
) -> None:
    """``figures`` is {matcher: {figure: metrics-with-audit}}."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    for matcher, per_figure in figures.items():
        fig_dir = os.path.join(out_dir, "figures", matcher)
        os.makedirs(fig_dir, exist_ok=True)
        for figure, metrics in per_figure.items():
            with open(os.path.join(fig_dir, f"{figure}.json"), "w") as f:
                json.dump(metrics, f, indent=2)

    if disagreement is not None:
        with open(os.path.join(out_dir, "matcher_disagreement.json"), "w") as f:
            json.dump(disagreement, f, indent=2)


def format_summary_table(summaries: dict[str, dict]) -> str:
    """Compact micro/macro table, one row pair per matcher."""
    header = (
        "| matcher | level | node F1 | triplet F1 (strict) | triplet F1 (relaxed) "
        "| wiring F1 (cond.) | property F1 | class acc | sub_type acc |"
    )
    rule = "|---|---|---|---|---|---|---|---|---|"
    rows = []
    for matcher, summary in sorted(summaries.items()):
        micro, macro = summary["micro"], summary["macro"]
        cond_micro = micro.get("triplets_conditional", {}).get("f1")
        cond_macro = macro.get("triplets_conditional_f1")
        fmt = lambda v: f"{v:.3f}" if v is not None else "-"  # noqa: E731
        rows.append(
            f"| {matcher} | micro | {micro['nodes']['f1']:.3f} "
            f"| {micro['triplets_strict']['f1']:.3f} | {micro['triplets_relaxed']['f1']:.3f} "
            f"| {fmt(cond_micro)} "
            f"| {micro['properties']['f1']:.3f} | {micro['accuracy']['class']['accuracy']:.3f} "
            f"| {micro['accuracy']['sub_type']['accuracy']:.3f} |"
        )
        rows.append(
            f"| {matcher} | macro | {macro['nodes_f1']:.3f} "
            f"| {macro['triplets_strict_f1']:.3f} | {macro['triplets_relaxed_f1']:.3f} "
            f"| {fmt(cond_macro)} "
            f"| {macro['properties_f1']:.3f} | {macro['class_accuracy']:.3f} "
            f"| {macro['sub_type_accuracy']:.3f} |"
        )
    return "\n".join([header, rule, *rows])
