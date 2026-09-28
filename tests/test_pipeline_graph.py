"""Tests for the registry, routing functions, and compiled pipeline topology.

The pipeline is compiled with placeholder API keys (``require_keys=False``):
chat models are constructed but never invoked, so no network access happens.
"""

import os

from kg_pipeline.workflow.build import PIPELINE_NODES, build_pipeline
from kg_pipeline.workflow.routing import (
    MAX_CORRECTION_RETRIES,
    route_after_normalize,
    route_after_validate,
    route_after_vision,
)
from kg_pipeline.registry import build_deps, discover_nodes
from kg_pipeline.schema.graph import ArchitectureGraph, RawTopology
from kg_pipeline.settings import Settings


def _deps():
    settings = Settings(_env_file=None)
    return build_deps(settings, require_keys=False)


def test_registry_discovers_every_pipeline_node():
    registry = discover_nodes()

    assert set(PIPELINE_NODES) <= set(registry)
    assert {registry[n].kind for n in ("vision_extract", "normalize", "correct")} == {"agent"}
    assert {registry[n].kind for n in ("prune_bypasses", "resolve_legend", "normalize_terms", "explode_sequences", "validate", "export")} == {"tool"}


def test_pipeline_compiles_and_contains_all_nodes():
    compiled = build_pipeline(_deps())

    drawable = compiled.get_graph()
    assert set(PIPELINE_NODES) <= set(drawable.nodes)
    # The Studio/diagram surface renders without error.
    assert "vision_extract" in drawable.draw_mermaid()


def _raw(nodes=({"id": "a", "raw_text_elements": ["Conv"]},)):
    return RawTopology.model_validate({"raw_nodes": list(nodes)})


def test_route_after_vision():
    assert route_after_vision({}) == "no_content"
    assert route_after_vision({"raw_topology": _raw(nodes=())}) == "no_content"
    assert route_after_vision({"raw_topology": _raw()}) == "parse"


def test_route_after_normalize():
    assert route_after_normalize({}) == "failed"
    empty = ArchitectureGraph()
    assert route_after_normalize({"graph": empty}) == "failed"
    graph = ArchitectureGraph.model_validate(
        {"nodes": [{"id": "a", "class": "DATA", "sub_type": "input", "raw_text": ""}]}
    )
    assert route_after_normalize({"graph": graph}) == "continue"


def test_route_after_validate_caps_retries():
    assert route_after_validate({"validated": True}) == "export"
    assert route_after_validate({"validated": False, "retry_count": 0}) == "correct"
    assert (
        route_after_validate({"validated": False, "retry_count": MAX_CORRECTION_RETRIES})
        == "export"
    )


def test_failed_correction_still_consumes_a_retry(monkeypatch):
    """Regression: an exception inside `correct` must not leave retry_count
    frozen -- that made validate -> correct loop until the recursion limit
    killed the figure (2/73 in benchmark run_122b)."""
    import kg_pipeline.agents.corrector as corrector

    class ExplodingModel:
        def invoke(self, messages):
            raise RuntimeError("LLM down")

    monkeypatch.setattr(
        corrector, "with_retrying_structured_output", lambda model, schema: ExplodingModel()
    )
    node = corrector.make_node(_deps())

    graph = ArchitectureGraph.model_validate(
        {"nodes": [{"id": "a", "class": "DATA", "sub_type": "input", "raw_text": ""}]}
    )
    state = {"image_path": "fig.png", "graph": graph, "critique_log": ["bad edge"], "retry_count": 1}
    update = node(state)

    assert update["retry_count"] == 2
    assert "[correct]" in update["error"]
    assert "graph" not in update  # previous graph is kept, not clobbered


def test_run_batch_parallel_runs_all_and_skips_done(tmp_path, monkeypatch):
    """Parallel runner: every pending figure runs exactly once, already-done
    figures are skipped, and failures are captured per figure."""
    import threading

    from kg_pipeline import runner

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        (inputs / name).write_bytes(b"x")
    final = tmp_path / "final_graphs"
    final.mkdir()
    (final / "a.json").write_text("{}")  # already done -> must be skipped

    seen, lock = [], threading.Lock()

    class Graph:
        def invoke(self, state, config=None):
            with lock:
                seen.append(os.path.basename(state["image_path"]))
            if state["image_path"].endswith("c.png"):
                raise RuntimeError("boom")
            return {"status": "success", "retry_count": 0, "graph": None}

    perf = runner.run_batch_parallel(Graph(), str(inputs), str(tmp_path), workers=3)

    assert sorted(seen) == ["b.png", "c.png"]  # a.png skipped
    assert len(perf) == 2
    by_name = {p["filename"]: p for p in perf}
    assert by_name["b.png"]["status"] == "success"
    assert by_name["c.png"]["status"] == "failed"
    assert "boom" in by_name["c.png"]["error"]
