"""Orchestrates one evaluation: a directory of predicted graphs vs the
golden dataset, paired by figure stem (the pipeline names its exports after
the input image stem, which is exactly how the goldens are named).

Golden figures with no prediction file count as all-false-negatives -- a
figure the pipeline crashed on or emitted empty must hurt the aggregate, not
silently vanish from it.
"""

import glob
import json
import os

from kg_pipeline.eval.analysis import analyze_figures, build_report
from kg_pipeline.eval.matching import MATCHERS, MatchConfig
from kg_pipeline.eval.metrics import aggregate, figure_metrics, missing_figure_metrics
from kg_pipeline.eval.preprocess import load_graph, prepare_graph
from kg_pipeline.eval.report import (
    format_summary_table,
    match_audit,
    matcher_disagreement,
    write_reports,
)
from kg_pipeline.tools.dictionary_normalizer import load_lookup_table

DEFAULT_GOLDEN_DIR = os.path.join("data", "golden_dataset", "ground_truth")


def resolve_pred_dir(run_dir: str) -> str:
    """Accepts either a run directory (uses its ``final_graphs/``) or a
    directory that holds the graph JSONs directly."""
    final = os.path.join(run_dir, "final_graphs")
    return final if os.path.isdir(final) else run_dir


def evaluate_run(
    run_dir: str,
    golden_dir: str = DEFAULT_GOLDEN_DIR,
    matchers: tuple[str, ...] = ("propagation",),
    cfg: MatchConfig | None = None,
    out_dir: str | None = None,
    run_meta: dict | None = None,
) -> dict:
    """Returns {matcher: aggregate summary}; writes full reports to out_dir.

    ``run_meta`` (models, timings, pipeline statuses...) is echoed verbatim
    into the REPORT.md header so a report is self-describing later.
    """
    cfg = cfg or MatchConfig()
    table = load_lookup_table()
    pred_dir = resolve_pred_dir(run_dir)

    golden_paths = sorted(glob.glob(os.path.join(golden_dir, "*.json")))
    if not golden_paths:
        raise FileNotFoundError(f"No golden graphs found in {golden_dir}")

    figures: dict[str, dict[str, dict]] = {name: {} for name in matchers}
    matches: dict[str, dict[str, object]] = {name: {} for name in matchers}

    for gold_path in golden_paths:
        figure = os.path.splitext(os.path.basename(gold_path))[0]
        gold = prepare_graph(load_graph(gold_path), table)

        pred_path = os.path.join(pred_dir, f"{figure}.json")
        if not os.path.isfile(pred_path):
            for name in matchers:
                figures[name][figure] = missing_figure_metrics(gold)
            continue
        pred = prepare_graph(load_graph(pred_path), table)

        for name in matchers:
            result = MATCHERS[name](pred, gold, cfg)
            metrics = figure_metrics(pred, gold, result)
            metrics["match"] = match_audit(result, pred, gold)
            figures[name][figure] = metrics
            matches[name][figure] = result

    summaries = {name: aggregate(figures[name]) for name in matchers}
    disagreement = (
        matcher_disagreement(matches, summaries) if len(matchers) > 1 else None
    )

    # Error analysis on the trusted matcher (propagation won the comparison).
    primary = "propagation" if "propagation" in matchers else matchers[0]
    analysis = analyze_figures(figures[primary])

    if out_dir:
        write_reports(out_dir, figures, summaries, disagreement)
        with open(os.path.join(out_dir, "error_analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2)
        report = build_report(
            summaries, analysis, primary, format_summary_table(summaries), run_meta
        )
        with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
            f.write(report)

    return summaries
