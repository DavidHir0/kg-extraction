"""The whole pipeline on one figure, with fake models.

The other test files check each step on its own. These run the compiled
pipeline from image to exported file, so they catch mistakes in how the steps
are connected: the wrong prompt sent, the graph saved under the wrong name, an
invalid graph reported as a success, or a step skipped inside the correction
loop. No API key and no network: every model call returns a prepared answer.
"""

import json

import PIL.Image
import pytest
from langchain_core.runnables import RunnableLambda

from kg_pipeline.prompts.loader import load_agent_prompt
from kg_pipeline.registry import build_deps, discover_nodes
from kg_pipeline.schema.graph import ArchitectureGraph, RawTopology
from kg_pipeline.settings import Settings
from kg_pipeline.workflow.build import build_pipeline

FIGURE = "net_fig1"

RAW = {
    "legend_map": {"style_conv": "Convolution"},
    "raw_nodes": [
        {"id": "in", "raw_text_elements": ["Input"]},
        {"id": "gap", "raw_text_elements": []},
        {"id": "conv", "raw_text_elements": ["3x3"], "style_id": "style_conv"},
    ],
    "raw_edges": [
        {"source": "in", "target": "gap"},
        {"source": "gap", "target": "conv"},
    ],
}


def _graph(target="conv"):
    return ArchitectureGraph.model_validate(
        {
            # Deliberately not the figure name: the pipeline must stamp it.
            "architecture_name": "whatever-the-model-said",
            "nodes": [
                {"id": "in", "class": "DATA", "sub_type": "input", "raw_text": "Input"},
                {"id": "conv", "class": "LAYER", "sub_type": "convolution", "raw_text": "3x3"},
            ],
            "edges": [{"source": "in", "target": target, "role": "forward"}],
            "groups": [],
        }
    )


VALID = _graph()
INVALID = _graph(target="ghost")  # edge to a node that does not exist


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Returns run(normalizer_answer, corrector_answers) -> (state, calls, out_dir)."""
    for var in ("KG_VISION_PROVIDER", "KG_TEXT_PROVIDER", "KG_VISION_MODEL", "KG_TEXT_MODEL"):
        monkeypatch.delenv(var, raising=False)

    image = tmp_path / f"{FIGURE}.png"
    PIL.Image.new("RGB", (64, 64), "white").save(image)

    def _run(normalizer_answer, corrector_answers=()):
        answers = {
            "vision": [RawTopology.model_validate(RAW)],
            "normalizer": [normalizer_answer],
            "corrector": list(corrector_answers),
        }
        calls = []

        def fake_structured(chat_model, schema, *args, **kwargs):
            def invoke(messages, config=None):
                system = messages[0].content
                if system == load_agent_prompt("vision_extractor"):
                    agent = "vision"
                elif system == load_agent_prompt("normalizer", with_rulebook=True):
                    agent = "normalizer"
                elif system == load_agent_prompt("corrector", with_rulebook=True):
                    agent = "corrector"
                else:
                    agent = "unknown prompt"
                calls.append((agent, schema.__name__))
                return answers[agent].pop(0).model_copy(deep=True)

            return RunnableLambda(invoke)

        discover_nodes()
        for module in ("vision_extractor", "normalizer", "corrector"):
            monkeypatch.setattr(
                f"kg_pipeline.agents.{module}.with_retrying_structured_output", fake_structured
            )
        pipeline = build_pipeline(build_deps(Settings(_env_file=None), require_keys=False))
        out_dir = tmp_path / "out"
        state = pipeline.invoke({"image_path": str(image), "output_dir": str(out_dir)})
        return state, calls, out_dir

    return _run


def _exported(out_dir):
    return json.loads((out_dir / "final_graphs" / f"{FIGURE}.json").read_text())


def test_clean_figure_is_exported_under_its_name_with_styles(run):
    state, calls, out_dir = run(VALID)

    assert calls == [("vision", "RawTopology"), ("normalizer", "ArchitectureGraph")]
    assert state["status"] == "success"
    graph = _exported(out_dir)
    assert graph["architecture_name"] == FIGURE
    styles = {n["id"]: n.get("style_id") for n in graph["nodes"]}
    assert styles == {"in": None, "conv": "style_conv"}
    assert not (out_dir / "final_graphs" / f"{FIGURE}.critique.json").exists()
    assert (out_dir / "debug_outputs" / f"{FIGURE}_vision_raw.json").exists()


def test_invalid_graph_is_corrected_and_keeps_its_styles(run):
    state, calls, out_dir = run(INVALID, corrector_answers=[VALID])

    assert [agent for agent, _ in calls] == ["vision", "normalizer", "corrector"]
    assert state["status"] == "success"
    assert state["retry_count"] == 1
    graph = _exported(out_dir)
    assert graph["architecture_name"] == FIGURE
    assert {n["id"]: n.get("style_id") for n in graph["nodes"]}["conv"] == "style_conv"


def test_graph_that_stays_invalid_is_flagged_not_reported_as_success(run):
    state, calls, out_dir = run(INVALID, corrector_answers=[INVALID] * 3)

    assert [agent for agent, _ in calls].count("corrector") == 3
    assert state["status"] == "unvalidated"
    critique = json.loads((out_dir / "final_graphs" / f"{FIGURE}.critique.json").read_text())
    assert critique["retry_count"] == 3
    assert any("ghost" in c for c in critique["critique_log"])


def test_empty_vision_output_exports_nothing(run, monkeypatch):
    monkeypatch.setitem(RAW, "raw_nodes", [])
    monkeypatch.setitem(RAW, "raw_edges", [])
    state, calls, out_dir = run(VALID)

    assert calls == [("vision", "RawTopology")]
    assert state["status"] == "empty"
    assert not (out_dir / "final_graphs" / f"{FIGURE}.json").exists()
