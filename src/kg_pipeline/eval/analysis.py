"""Error analysis over a finished evaluation: WHERE does the pipeline fail.

Consumes the per-figure metric dicts (with their enriched match audits) and
aggregates the recurring error patterns a human needs for improvement work:

- class confusion matrix (matched pairs) + missed/hallucinated per class,
- sub_type confusions within a correctly classified class,
- edge role confusion (forward mistaken for skip, ...),
- property error rates per property key (kernel vs channels vs dimensions...),
- the most frequently missed golden nodes and hallucinated predictions,
  grouped by (class, sub_type) with concrete figure/raw_text examples,
- a worst-figures ranking to direct visual debugging.

Everything lands in ``error_analysis.json`` plus a human-readable
``REPORT.md`` built by :func:`build_report`.
"""

from kg_pipeline.eval.metrics import NODE_CLASSES

MISSED = "MISSED"          # golden node with no matched prediction
HALLUCINATED = "HALLUCINATED"  # predicted node with no golden counterpart
MAX_EXAMPLES = 8


def _bump(table: dict, *keys, by: int = 1) -> None:
    for key in keys[:-1]:
        table = table.setdefault(key, {})
    table[keys[-1]] = table.get(keys[-1], 0) + by


def analyze_figures(figures: dict[str, dict]) -> dict:
    """Aggregate error patterns from per-figure metrics (one matcher)."""
    class_confusion: dict[str, dict[str, int]] = {}
    sub_type_confusion: dict[str, int] = {}
    role_confusion: dict[str, dict[str, int]] = {}
    property_errors: dict[str, dict[str, int]] = {}
    missed: dict[str, dict] = {}
    hallucinated: dict[str, dict] = {}
    missing_figures = []

    for figure, metrics in sorted(figures.items()):
        if metrics.get("missing_prediction"):
            missing_figures.append(figure)
            continue

        audit = metrics.get("match", {})
        for pair in audit.get("pairs", []):
            p, g = pair["pred"], pair["gold"]
            _bump(class_confusion, g["class"], p["class"])
            if p["class"] == g["class"] and p["sub_type"] != g["sub_type"]:
                _bump(sub_type_confusion, f"{g['class']}: {g['sub_type']} -> {p['sub_type']}")

        for info, table in ((audit.get("unmatched_gold", []), missed),
                            (audit.get("unmatched_pred", []), hallucinated)):
            for node in info:
                key = f"{node['class']}/{node['sub_type']}"
                entry = table.setdefault(key, {"count": 0, "examples": []})
                entry["count"] += 1
                if len(entry["examples"]) < MAX_EXAMPLES:
                    entry["examples"].append({"figure": figure, "raw_text": node["raw_text"]})
                _bump(class_confusion, node["class"], MISSED) if table is missed else _bump(
                    class_confusion, HALLUCINATED, node["class"]
                )

        for gold_role, counts in metrics.get("role_confusion", {}).items():
            for pred_role, count in counts.items():
                _bump(role_confusion, gold_role, pred_role, by=count)

        for key, counts in metrics.get("property_errors", {}).items():
            for bucket, count in counts.items():
                _bump(property_errors, key, bucket, by=count)

    worst_figures = sorted(
        (
            {
                "figure": figure,
                "node_f1": round(m["nodes"]["f1"], 4),
                "triplet_strict_f1": round(m["triplets"]["strict"]["f1"], 4),
                "triplet_relaxed_f1": round(m["triplets"]["relaxed"]["f1"], 4),
                "property_f1": round(m["properties"]["f1"], 4),
                "missing_prediction": bool(m.get("missing_prediction")),
            }
            for figure, m in figures.items()
        ),
        key=lambda row: (row["node_f1"], row["triplet_strict_f1"]),
    )

    return {
        "missing_figures": missing_figures,
        "class_confusion": class_confusion,
        "sub_type_confusion": dict(
            sorted(sub_type_confusion.items(), key=lambda kv: -kv[1])
        ),
        "role_confusion": role_confusion,
        "property_errors": {
            key: {**counts, "recall": counts.get("tp", 0) / max(counts.get("tp", 0) + counts.get("fn", 0), 1)}
            for key, counts in sorted(property_errors.items())
        },
        "missed_nodes": dict(sorted(missed.items(), key=lambda kv: -kv[1]["count"])),
        "hallucinated_nodes": dict(sorted(hallucinated.items(), key=lambda kv: -kv[1]["count"])),
        "figures_ranked_worst_first": worst_figures,
    }


# --- markdown report ---


def _prf_row(label: str, block: dict) -> str:
    return (
        f"| {label} | {block['precision']:.3f} | {block['recall']:.3f} "
        f"| {block['f1']:.3f} | {block['tp']}/{block['fp']}/{block['fn']} |"
    )


def _confusion_table(confusion: dict[str, dict[str, int]], columns: list[str]) -> list[str]:
    lines = ["| gold \\ pred | " + " | ".join(columns) + " |",
             "|---|" + "---|" * len(columns)]
    for row_label in [*NODE_CLASSES, HALLUCINATED]:
        if row_label not in confusion:
            continue
        row = confusion[row_label]
        lines.append(
            f"| {row_label} | " + " | ".join(str(row.get(c, 0)) for c in columns) + " |"
        )
    return lines


def build_report(
    summaries: dict[str, dict],
    analysis: dict,
    primary_matcher: str,
    summary_table: str,
    run_meta: dict | None = None,
) -> str:
    primary = summaries[primary_matcher]
    micro = primary["micro"]
    lines = ["# Golden-Dataset Benchmark Report", ""]

    if run_meta:
        lines += ["## Run", ""]
        lines += [f"- **{key}**: {value}" for key, value in run_meta.items()]
        lines += [""]

    lines += [
        "## Headline metrics",
        "",
        summary_table,
        "",
        f"Error analysis below uses the **{primary_matcher}** matcher.",
        "",
        "| metric | precision | recall | F1 | tp/fp/fn |",
        "|---|---|---|---|---|",
        _prf_row("nodes", micro["nodes"]),
        _prf_row("triplets (strict)", micro["triplets_strict"]),
        _prf_row("triplets (relaxed)", micro["triplets_relaxed"]),
        _prf_row("properties", micro["properties"]),
        "",
    ]

    if analysis["missing_figures"]:
        lines += [
            f"## Failed figures ({len(analysis['missing_figures'])}, counted as all-FN)",
            "",
            *[f"- `{f}`" for f in analysis["missing_figures"]],
            "",
        ]

    lines += ["## Class confusion (matched pairs + missed/hallucinated)", ""]
    lines += _confusion_table(analysis["class_confusion"], [*NODE_CLASSES, MISSED])
    lines += [""]

    if analysis["sub_type_confusion"]:
        lines += ["## Top sub_type confusions (class correct, sub_type wrong)", ""]
        lines += [
            f"- {count}x {name}"
            for name, count in list(analysis["sub_type_confusion"].items())[:20]
        ]
        lines += [""]

    if analysis["role_confusion"]:
        roles = ["forward", "skip", "condition", "backward"]
        lines += ["## Edge role confusion (endpoint-matched edges)", ""]
        lines += ["| gold \\ pred | " + " | ".join(roles) + " |", "|---|" + "---|" * len(roles)]
        for gold_role in roles:
            if gold_role in analysis["role_confusion"]:
                row = analysis["role_confusion"][gold_role]
                lines.append(f"| {gold_role} | " + " | ".join(str(row.get(r, 0)) for r in roles) + " |")
        lines += [""]

    if analysis["property_errors"]:
        lines += [
            "## Property errors by key",
            "",
            "| property | tp | fp | fn | recall |",
            "|---|---|---|---|---|",
        ]
        for key, counts in analysis["property_errors"].items():
            lines.append(
                f"| {key} | {counts.get('tp', 0)} | {counts.get('fp', 0)} "
                f"| {counts.get('fn', 0)} | {counts['recall']:.3f} |"
            )
        lines += [""]

    for title, table in (
        ("Most-missed golden nodes", analysis["missed_nodes"]),
        ("Most-hallucinated predictions", analysis["hallucinated_nodes"]),
    ):
        if not table:
            continue
        lines += [f"## {title}", ""]
        for key, entry in list(table.items())[:15]:
            examples = "; ".join(
                f"`{e['raw_text'][:40]}` ({e['figure'][:40]})" for e in entry["examples"][:3]
            )
            lines.append(f"- **{key}** x{entry['count']} — e.g. {examples}")
        lines += [""]

    lines += ["## Figures ranked worst-first", "",
              "| figure | node F1 | strict F1 | relaxed F1 | property F1 |",
              "|---|---|---|---|---|"]
    for row in analysis["figures_ranked_worst_first"]:
        flag = " (MISSING)" if row["missing_prediction"] else ""
        lines.append(
            f"| {row['figure']}{flag} | {row['node_f1']:.3f} | {row['triplet_strict_f1']:.3f} "
            f"| {row['triplet_relaxed_f1']:.3f} | {row['property_f1']:.3f} |"
        )
    lines += [""]
    return "\n".join(lines)
