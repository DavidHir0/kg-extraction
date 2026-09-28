"""Metrics over one matched (predicted, golden) graph pair, plus dataset
aggregation.

Everything downstream of matching is deliberately dumb counting:

- node detection P/R/F1 (matched vs the two node totals),
- class / sub_type accuracy over matched pairs (detection and classification
  are separated on purpose -- a misclassified box counts as detected),
- triplet P/R/F1, strict (edge role must match) and relaxed (role ignored),
- property P/R/F1 over normalized ``key=value`` pairs on matched nodes,
- per-class detection/accuracy breakdown.

Groups are reported as bare counts only (annotation of groups is the least
consistent part of any figure; they stay out of the headline score).
"""

from kg_pipeline.eval.matching import MatchResult
from kg_pipeline.eval.preprocess import property_pairs
from kg_pipeline.schema.graph import ArchitectureGraph

NODE_CLASSES = ["DATA", "LAYER", "MODIFIER", "JUNCTION", "MACRO"]


def prf(tp: int, fp: int, fn: int) -> dict:
    if tp + fp + fn == 0:
        # Vacuously perfect: nothing to find, nothing claimed (e.g. property
        # F1 on a figure whose golden carries no properties at all).
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _ratio(hits: int, total: int) -> dict:
    return {"correct": hits, "total": total, "accuracy": hits / total if total else 0.0}


def _triplets(graph: ArchitectureGraph) -> set[tuple[str, str, str]]:
    """Deduplicated (source, role, target) set."""
    return {(e.source, e.role, e.target) for e in graph.edges}


def figure_metrics(
    pred: ArchitectureGraph, gold: ArchitectureGraph, result: MatchResult
) -> dict:
    """All metrics + the audit detail (triplet TP/FP/FN lists) for one figure."""
    p_by_id = {n.id: n for n in pred.nodes}
    g_by_id = {n.id: n for n in gold.nodes}
    mapping = result.pred_to_gold

    # --- node detection & classification ---
    nodes = prf(len(result.pairs), len(result.unmatched_pred), len(result.unmatched_gold))

    matched = [(p_by_id[m.pred_id], g_by_id[m.gold_id]) for m in result.pairs]
    class_hits = sum(1 for p, g in matched if p.class_ == g.class_)
    sub_type_hits = sum(1 for p, g in matched if p.sub_type == g.sub_type)
    both_hits = sum(1 for p, g in matched if p.class_ == g.class_ and p.sub_type == g.sub_type)
    accuracy = {
        "class": _ratio(class_hits, len(matched)),
        "sub_type": _ratio(sub_type_hits, len(matched)),
        "class_and_sub_type": _ratio(both_hits, len(matched)),
    }

    # --- triplets ---
    gold_strict = _triplets(gold)
    gold_relaxed = {(s, t) for s, _, t in gold_strict}
    pred_strict = _triplets(pred)

    strict_tp, strict_fp = [], []
    for source, role, target in sorted(pred_strict):
        mapped = (mapping.get(source), role, mapping.get(target))
        entry = {"pred": [source, role, target], "mapped": list(mapped)}
        (strict_tp if mapped in gold_strict else strict_fp).append(entry)
    strict_fn = [
        list(t)
        for t in sorted(gold_strict - {(mapping.get(s), r, mapping.get(t)) for s, r, t in pred_strict})
    ]

    # Relaxed scoring deduplicates on the PRED side (two roles between the
    # same endpoints are one relaxed claim) and maps each claim through the
    # match; hits are counted against the golden relaxed set.
    pred_relaxed = {(s, t) for s, _, t in pred_strict}
    relaxed_hits = {
        (mapping.get(s), mapping.get(t))
        for s, t in pred_relaxed
        if (mapping.get(s), mapping.get(t)) in gold_relaxed
    }
    relaxed_fp_count = sum(
        1 for s, t in pred_relaxed if (mapping.get(s), mapping.get(t)) not in gold_relaxed
    )
    relaxed_fn = [list(t) for t in sorted(gold_relaxed - relaxed_hits)]

    # Conditional (endpoint-matched-only) wiring quality, after the legacy
    # GraphEvaluator's "isolated" metric: restrict both sides to edges whose
    # endpoints were matched, so node-recall damage doesn't drown out the
    # wiring signal. Diagnostic ONLY -- it forgives every missed/hallucinated
    # node; headline claims stay on strict/relaxed.
    matched_pred_ids = {m.pred_id for m in result.pairs}
    matched_gold_ids = {m.gold_id for m in result.pairs}
    cond_pred = {(s, t) for s, t in pred_relaxed if s in matched_pred_ids and t in matched_pred_ids}
    cond_gold = {(s, t) for s, t in gold_relaxed if s in matched_gold_ids and t in matched_gold_ids}
    cond_hits = sum(1 for s, t in cond_pred if (mapping.get(s), mapping.get(t)) in cond_gold)

    triplets = {
        "strict": prf(len(strict_tp), len(strict_fp), len(strict_fn)),
        "relaxed": prf(len(relaxed_hits), relaxed_fp_count, len(relaxed_fn)),
        "conditional": prf(cond_hits, len(cond_pred) - cond_hits, len(cond_gold) - cond_hits),
    }

    # Role confusion: for endpoint-matched pred edges whose counterpart edge
    # exists in the gold (any role), count gold_role -> pred_role. The
    # diagonal is strict-correct; off-diagonal is pure role confusion.
    gold_roles: dict[tuple[str, str], set[str]] = {}
    for s, r, t in gold_strict:
        gold_roles.setdefault((s, t), set()).add(r)
    role_confusion: dict[str, dict[str, int]] = {}
    for source, role, target in pred_strict:
        for gold_role in gold_roles.get((mapping.get(source), mapping.get(target)), ()):
            counts = role_confusion.setdefault(gold_role, {})
            counts[role] = counts.get(role, 0) + 1

    # --- properties (over matched nodes only), plus a per-key breakdown ---
    prop_tp = prop_fp = prop_fn = 0
    property_errors: dict[str, dict[str, int]] = {}

    def _prop_key(pair: str) -> str:
        return pair.split("=", 1)[0].split("[", 1)[0]

    for p, g in matched:
        pp, gp = property_pairs(p), property_pairs(g)
        prop_tp += len(pp & gp)
        prop_fp += len(pp - gp)
        prop_fn += len(gp - pp)
        for bucket, pairs_ in (("tp", pp & gp), ("fp", pp - gp), ("fn", gp - pp)):
            for pair in pairs_:
                entry = property_errors.setdefault(_prop_key(pair), {"tp": 0, "fp": 0, "fn": 0})
                entry[bucket] += 1
    properties = prf(prop_tp, prop_fp, prop_fn)

    # --- per-class breakdown (from the golden side: how well is each golden
    # class detected and typed; plus pred-side FP counts per class) ---
    matched_gold_ids = {m.gold_id for m in result.pairs}
    per_class = {}
    for cls in NODE_CLASSES:
        gold_ids = [n.id for n in gold.nodes if n.class_ == cls]
        detected = [g for g in gold_ids if g in matched_gold_ids]
        typed = sum(
            1 for p, g in matched if g.class_ == cls and p.class_ == g.class_ and p.sub_type == g.sub_type
        )
        per_class[cls] = {
            "gold_total": len(gold_ids),
            "detected": len(detected),
            "recall": len(detected) / len(gold_ids) if gold_ids else 0.0,
            "typed_correct": typed,
            "unmatched_pred": sum(1 for i in result.unmatched_pred if p_by_id[i].class_ == cls),
        }

    return {
        "nodes": nodes,
        "accuracy": accuracy,
        "triplets": triplets,
        "properties": properties,
        "per_class": per_class,
        "role_confusion": role_confusion,
        "property_errors": property_errors,
        "groups": {"pred": len(pred.groups), "gold": len(gold.groups)},
        "audit": {
            "triplet_tp_strict": strict_tp,
            "triplet_fp_strict": strict_fp,
            "triplet_fn_strict": strict_fn,
            "triplet_fn_relaxed": relaxed_fn,
        },
    }


def missing_figure_metrics(gold: ArchitectureGraph) -> dict:
    """All-FN placeholder for a golden figure with no prediction file."""
    gold_strict = _triplets(gold)
    gold_relaxed = {(s, t) for s, _, t in gold_strict}
    empty_matched = _ratio(0, 0)
    return {
        "nodes": prf(0, 0, len(gold.nodes)),
        "accuracy": {"class": empty_matched, "sub_type": empty_matched, "class_and_sub_type": empty_matched},
        "triplets": {
            "strict": prf(0, 0, len(gold_strict)),
            "relaxed": prf(0, 0, len(gold_relaxed)),
            # No matched endpoints at all -> the conditional universe is empty.
            "conditional": prf(0, 0, 0),
        },
        "properties": prf(0, 0, sum(len(property_pairs(n)) for n in gold.nodes)),
        "role_confusion": {},
        "property_errors": {},
        "per_class": {
            cls: {
                "gold_total": sum(1 for n in gold.nodes if n.class_ == cls),
                "detected": 0,
                "recall": 0.0,
                "typed_correct": 0,
                "unmatched_pred": 0,
            }
            for cls in NODE_CLASSES
        },
        "groups": {"pred": 0, "gold": len(gold.groups)},
        "missing_prediction": True,
        "audit": {},
    }


def aggregate(figures: dict[str, dict]) -> dict:
    """Micro (pooled counts) and macro (mean of per-figure values) over a run."""
    if not figures:
        return {}

    def micro_prf(path: tuple[str, ...]) -> dict:
        tp = fp = fn = 0
        for fig in figures.values():
            section = fig
            for key in path:
                section = section[key]
            tp, fp, fn = tp + section["tp"], fp + section["fp"], fn + section["fn"]
        return prf(tp, fp, fn)

    def micro_ratio(name: str) -> dict:
        hits = sum(f["accuracy"][name]["correct"] for f in figures.values())
        total = sum(f["accuracy"][name]["total"] for f in figures.values())
        return _ratio(hits, total)

    def macro(fn_get) -> float:
        return sum(fn_get(f) for f in figures.values()) / len(figures)

    per_class = {}
    for cls in NODE_CLASSES:
        gold_total = sum(f["per_class"][cls]["gold_total"] for f in figures.values())
        detected = sum(f["per_class"][cls]["detected"] for f in figures.values())
        typed = sum(f["per_class"][cls]["typed_correct"] for f in figures.values())
        per_class[cls] = {
            "gold_total": gold_total,
            "recall": detected / gold_total if gold_total else 0.0,
            "typed_recall": typed / gold_total if gold_total else 0.0,
            "unmatched_pred": sum(f["per_class"][cls]["unmatched_pred"] for f in figures.values()),
        }

    return {
        "figures": len(figures),
        "missing_predictions": sum(1 for f in figures.values() if f.get("missing_prediction")),
        "micro": {
            "nodes": micro_prf(("nodes",)),
            "triplets_strict": micro_prf(("triplets", "strict")),
            "triplets_relaxed": micro_prf(("triplets", "relaxed")),
            "triplets_conditional": micro_prf(("triplets", "conditional")),
            "properties": micro_prf(("properties",)),
            "accuracy": {name: micro_ratio(name) for name in ("class", "sub_type", "class_and_sub_type")},
        },
        "macro": {
            "nodes_f1": macro(lambda f: f["nodes"]["f1"]),
            "triplets_strict_f1": macro(lambda f: f["triplets"]["strict"]["f1"]),
            "triplets_relaxed_f1": macro(lambda f: f["triplets"]["relaxed"]["f1"]),
            "triplets_conditional_f1": macro(lambda f: f["triplets"]["conditional"]["f1"]),
            "properties_f1": macro(lambda f: f["properties"]["f1"]),
            "class_accuracy": macro(lambda f: f["accuracy"]["class"]["accuracy"]),
            "sub_type_accuracy": macro(lambda f: f["accuracy"]["sub_type"]["accuracy"]),
        },
        "per_class": per_class,
    }
