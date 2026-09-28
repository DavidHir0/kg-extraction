"""Agent 4 -- ``correct``: surgical repair in the validation loop.

Runs ONLY when the deterministic validator produced critiques and the retry
budget isn't exhausted (routed in ``workflow/routing.py``). Receives the failing
graph plus the machine-generated ``critique_log`` and is instructed to fix
exactly those violations and nothing else. Increments ``retry_count``, which
the router caps at ``MAX_CORRECTION_RETRIES``.
"""

import json

from kg_pipeline.workflow.state import PipelineState, figure_id
from kg_pipeline.llm.factory import with_retrying_structured_output
from kg_pipeline.prompts.loader import load_agent_prompt
from kg_pipeline.registry import NodeFn, PipelineDeps, register_node
from kg_pipeline.schema.graph import ArchitectureGraph
from langchain_core.messages import HumanMessage, SystemMessage


@register_node(
    "correct",
    kind="agent",
    description="Repairs exactly the validator-cited schema violations.",
)
def make_node(deps: PipelineDeps) -> NodeFn:
    system_prompt = load_agent_prompt("corrector", with_rulebook=True)
    model = with_retrying_structured_output(deps.text_model, ArchitectureGraph)

    def _node(state: PipelineState) -> dict:
        graph = state["graph"]
        critiques = state.get("critique_log", [])
        retries = state.get("retry_count", 0) + 1
        user_prompt = (
            "CRITIQUE LOG (fix ONLY these violations):\n- "
            + "\n- ".join(critiques)
            + "\n\nGRAPH:\n"
            + json.dumps(graph.model_dump(by_alias=True, exclude_none=True), indent=2)
        )
        # A failed correction attempt must still consume its retry: if the
        # exception escaped to the registry guard, retry_count would stay
        # frozen and validate -> correct would loop until the recursion limit
        # killed the whole figure (lost 2/73 in run_122b). Keep the previous
        # graph and let the router exhaust the budget into an export instead.
        try:
            result = model.invoke([SystemMessage(system_prompt), HumanMessage(user_prompt)])
        except Exception as e:  # noqa: BLE001 -- same contract as the registry guard
            return {"retry_count": retries, "error": f"[correct] {type(e).__name__}: {e}"}

        result.architecture_name = figure_id(state) or graph.architecture_name
        result.extraction_version = "5.8"
        return {"graph": result, "retry_count": retries}

    return _node
