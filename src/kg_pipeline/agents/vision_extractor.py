"""Agent 1 -- ``vision_extract``: the only node that sees the image.

A "dumb" perception engine: literal transcription of shapes, text, arrows,
legends, and group headings into :class:`RawTopology` -- no schema-fitting,
no classification. Headings/captions over boxed or zoomed-in regions land in
``raw_groups`` (with their text as ``label``), which the normalizer resolves
into containers, subgraph definitions, and MACRO blocks.
"""

import json
import os

from kg_pipeline.workflow.state import PipelineState, figure_id
from kg_pipeline.llm.factory import with_retrying_structured_output
from kg_pipeline.llm.vision import build_messages
from kg_pipeline.prompts.loader import load_agent_prompt
from kg_pipeline.registry import NodeFn, PipelineDeps, register_node
from kg_pipeline.schema.graph import RawTopology

USER_PROMPT = (
    "Extract all raw structural nodes, text strings, and arrows from this diagram."
)


@register_node(
    "vision_extract",
    kind="agent",
    description="Literal visual transcription of the diagram into a raw topology.",
)
def make_node(deps: PipelineDeps) -> NodeFn:
    system_prompt = load_agent_prompt("vision_extractor")
    model = with_retrying_structured_output(deps.vision_model, RawTopology)

    def _snapshot(topo: RawTopology, state: PipelineState) -> None:
        """Writes the vision output to ``debug_outputs/<figure>_vision_raw.json``
        before any later step edits it. The exporter's ``raw_topology.json`` is
        written last, after the bypass and legend tools have changed it."""
        out = state.get("output_dir")
        if not out:
            return
        debug = os.path.join(out, "debug_outputs")
        os.makedirs(debug, exist_ok=True)
        path = os.path.join(debug, f"{figure_id(state) or 'unknown'}_vision_raw.json")
        with open(path, "w") as f:
            json.dump(topo.model_dump(), f, indent=2)

    def _node(state: PipelineState) -> dict:
        messages = build_messages(system_prompt, USER_PROMPT, state["image_path"])
        topo = model.invoke(messages)
        _snapshot(topo, state)
        return {"raw_topology": topo}

    return _node
