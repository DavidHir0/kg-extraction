"""Tool -- ``export``: the single terminal node of the pipeline.

Every path through the graph (success, validation-exhausted, empty vision
result, node failure) ends here, so there is exactly one place that decides
the run's final ``status`` and persists artifacts (``<figure>`` is the
pipeline-wide identifier from :func:`kg_pipeline.workflow.state.figure_id`):

- ``final_graphs/<figure>.json``          -- the sanitized final graph
- ``final_graphs/<figure>.critique.json`` -- critique log, only when unvalidated
- ``debug_outputs/<figure>_raw_topology.json`` -- the raw vision extraction

To ship graphs to a database instead, add a writer here (e.g. a Neo4j
``MERGE`` per node/edge using a driver passed in via ``PipelineDeps``) --
the rest of the pipeline stays untouched.
"""

import json
import os

from kg_pipeline.workflow.state import PipelineState, figure_id
from kg_pipeline.registry import NodeFn, PipelineDeps, register_node

# Fallback for direct invokes that set no output_dir (e.g. LangGraph Studio);
# the CLI and web server always pass an explicit per-run/per-job directory.
DEFAULT_OUTPUT_DIR = os.path.join("outputs", "adhoc")


@register_node(
    "export",
    kind="tool",
    description="Writes final graph + debug artifacts to disk and sets the run status.",
)
def make_node(deps: PipelineDeps) -> NodeFn:
    def _node(state: PipelineState) -> dict:
        output_dir = state.get("output_dir") or DEFAULT_OUTPUT_DIR
        final_dir = os.path.join(output_dir, "final_graphs")
        debug_dir = os.path.join(output_dir, "debug_outputs")
        os.makedirs(final_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)

        filename = figure_id(state) or "unknown"

        raw_topology = state.get("raw_topology")
        if raw_topology is not None:
            debug_path = os.path.join(debug_dir, f"{filename}_raw_topology.json")
            with open(debug_path, "w") as f:
                json.dump(raw_topology.model_dump(), f, indent=2)

        graph = state.get("graph")
        if graph is None or not graph.nodes:
            return {"status": "failed" if state.get("error") else "empty", "export_path": ""}

        export_path = os.path.join(final_dir, f"{filename}.json")
        with open(export_path, "w") as f:
            json.dump(graph.model_dump(by_alias=True, exclude_none=True), f, indent=2)

        if state.get("validated"):
            status = "success"
        else:
            # Retry budget exhausted (or the corrector kept failing): keep the
            # best-effort graph but flag it and persist the open critiques.
            status = "unvalidated"
            critique_path = os.path.join(final_dir, f"{filename}.critique.json")
            with open(critique_path, "w") as f:
                json.dump(
                    {
                        "critique_log": state.get("critique_log", []),
                        "retry_count": state.get("retry_count", 0),
                        "error": state.get("error", ""),
                    },
                    f,
                    indent=2,
                )

        return {"status": status, "export_path": export_path}

    return _node
