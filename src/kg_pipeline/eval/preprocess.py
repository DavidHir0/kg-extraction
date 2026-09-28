"""Shared canonicalization for evaluation.

Golden graphs are stored in composite (pre-explode) form while the pipeline
exports already-sanitized graphs, so both sides are pushed through the same
``sanitize_graph`` + ``normalize_terms`` pair before any comparison. All text
comparison (raw_text similarity, property values) goes through
:func:`norm_text` so that LaTeX, casing, and whitespace never count as
differences.
"""

import json
import re

from kg_pipeline.tools.sanitizer import sanitize_graph
from kg_pipeline.schema.graph import ArchitectureGraph, NodeModel
from kg_pipeline.tools.dictionary_normalizer import LookupTable, normalize_terms, strip_latex

# Property keys excluded from both matching and property scoring: free-form
# human text (`name`, `note` -- and `note: "implicit"` must not create
# similarity between unrelated implicit nodes) and `sequence`, which is gone
# after sanitize_graph anyway.
EXCLUDED_PROPERTIES = {"name", "note", "sequence"}


def norm_text(text: str) -> str:
    """Canonical comparison form: LaTeX-stripped, lowercased, single-spaced."""
    return re.sub(r"\s+", " ", strip_latex(text).lower()).strip()


def _norm_value(value) -> str:
    return norm_text(value) if isinstance(value, str) else str(value)


def property_pairs(node: NodeModel) -> set[str]:
    """A node's scoreable properties as normalized ``key=value`` strings.

    List values keep their position (``dimensions[0]=512``) so reordered
    dimensions don't silently count as equal.
    """
    pairs: set[str] = set()
    for key, value in node.properties.model_dump(exclude_none=True).items():
        if key in EXCLUDED_PROPERTIES:
            continue
        if isinstance(value, list):
            pairs.update(f"{key}[{i}]={_norm_value(v)}" for i, v in enumerate(value))
        else:
            pairs.add(f"{key}={_norm_value(value)}")
    return pairs


def load_graph(path: str) -> ArchitectureGraph:
    with open(path) as f:
        return ArchitectureGraph.model_validate(json.load(f))


def prepare_graph(graph: ArchitectureGraph, table: LookupTable) -> ArchitectureGraph:
    """Explodes composites and canonicalizes property vocabulary.

    Idempotent on pipeline output (already exploded/normalized), required on
    goldens (stored composite).
    """
    return normalize_terms(sanitize_graph(graph), table)
