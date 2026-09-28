"""Agent 3 -- ``normalize``: text-only schema-fitting. Never sees the image.

The one big semantic pass: maps the raw topology onto the Universal Neural
Architecture Graph Schema (v5.8) -- class/sub_type assignment, composite
``properties.sequence`` construction (Golden Rule 3), MACRO/subgraph linking,
edge-role taxonomy, floating-math absorption (Rule 2), and on-arrow text
resolution (Rule 5). Kept as ONE call by design: splitting it into several
LLM passes multiplies cost and lets each pass corrupt fields it wasn't
focused on. The deterministic parts of its old job (legend resolution, bypass
purging, dictionary lookup, sequence explosion) live in ``kg_pipeline.tools``.
"""

import json

from kg_pipeline.workflow.state import PipelineState, figure_id
from kg_pipeline.llm.factory import with_retrying_structured_output
from kg_pipeline.prompts.loader import load_agent_prompt
from kg_pipeline.registry import NodeFn, PipelineDeps, register_node
from kg_pipeline.schema.graph import ArchitectureGraph
from langchain_core.messages import HumanMessage, SystemMessage


@register_node(
    "normalize",
    kind="agent",
    description="Fits the raw topology onto the v5.8 schema (text-only LLM pass).",
)
def make_node(deps: PipelineDeps) -> NodeFn:
    system_prompt = load_agent_prompt("normalizer", with_rulebook=True)
    model = with_retrying_structured_output(deps.text_model, ArchitectureGraph)

    def _node(state: PipelineState) -> dict:
        raw = state["raw_topology"]
        user_prompt = (
            "Normalize the following raw extraction dataset:\n"
            f"{json.dumps(raw.model_dump(), indent=2)}"
        )
        result = model.invoke([SystemMessage(system_prompt), HumanMessage(user_prompt)])

        # architecture_name/extraction_version are assigned deterministically
        # rather than trusting the LLM to echo them back correctly.
        result.architecture_name = figure_id(state)
        result.extraction_version = "5.8"
        return {"graph": result}

    return _node
