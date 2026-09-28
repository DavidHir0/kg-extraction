"""The shared state flowing through every node of the extraction pipeline.

Each node reads the keys it needs and returns a partial update (a plain
``dict`` with only the keys it changed); LangGraph merges the update into the
state. ``total=False`` means every key is optional -- a run starts with just
``image_path`` and the rest fills in as nodes execute.
"""

import os
from typing import TypedDict

from kg_pipeline.schema.graph import ArchitectureGraph, RawTopology


class PipelineState(TypedDict, total=False):
    # --- Inputs (provided at invoke time) ---
    image_path: str
    output_dir: str  # where the exporter writes results; defaults to outputs/

    # --- Vision stage ---
    raw_topology: RawTopology

    # --- Normalization stage ---
    # The working graph. Written by `normalize`, then transformed in place by
    # the deterministic tools and (on validation failure) by `correct`.
    graph: ArchitectureGraph

    # --- Validation / self-correction loop ---
    critique_log: list[str]  # machine-generated errors from the validator
    retry_count: int  # correction attempts so far (capped by routing)
    validated: bool  # True once the validator passes with zero critiques

    # --- Bookkeeping ---
    status: str  # final outcome set by the exporter: success | unvalidated | empty | failed
    export_path: str  # where the final graph JSON was written
    error: str  # last node exception, captured by the registry's guard wrapper


def figure_id(state: PipelineState) -> str:
    """THE identifier of a figure, everywhere: file name without extension.

    Stamped into the exported graph as ``architecture_name`` and used to name
    all output artifacts. There is deliberately no separate name input --
    figure files are already named ``<paper>_fig<N>``-style by the extraction
    stage, so the stem is the most meaningful identifier available.
    """
    return os.path.splitext(os.path.basename(state.get("image_path", "")))[0]
