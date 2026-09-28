"""Evaluation harness: pipeline outputs vs the golden dataset.

Pairs predicted and golden graphs by figure stem, aligns their nodes (no
shared ids) with a similarity-plus-structure matcher, and scores node,
triplet, and property precision/recall/F1. See ``matching.py`` for the two
matcher implementations and ``metrics.py`` for the metric definitions.
"""

from kg_pipeline.eval.evaluate import DEFAULT_GOLDEN_DIR, evaluate_run
from kg_pipeline.eval.matching import (
    MATCHERS,
    MatchConfig,
    match_anchor,
    match_propagation,
)

__all__ = [
    "DEFAULT_GOLDEN_DIR",
    "MATCHERS",
    "MatchConfig",
    "evaluate_run",
    "match_anchor",
    "match_propagation",
]
