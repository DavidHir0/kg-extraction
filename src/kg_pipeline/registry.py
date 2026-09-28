"""Plug-and-play node registry: the backbone of the modular pipeline.

Every pipeline node -- LLM **agent** or deterministic **tool** -- registers
itself here with the :func:`register_node` decorator. The graph builder
(``kg_pipeline.workflow.build``) then looks nodes up *by name*, so adding a new
node never requires touching existing node code:

1. Drop a new module into ``kg_pipeline/agents/`` (LLM call) or
   ``kg_pipeline/tools/`` (pure Python). Both packages auto-import their
   submodules, so the ``@register_node`` decorator runs on package import.
2. Decorate a factory ``(PipelineDeps) -> NodeFn`` with
   ``@register_node("my_node", kind="agent"|"tool", description="...")``.
   The factory receives shared dependencies (chat models, settings) once at
   build time and returns the actual node function.
3. Wire the node into the topology in ``workflow/build.py`` (one ``add_node``
   name in ``PIPELINE_NODES`` plus its edges).

Node functions take the :class:`~kg_pipeline.workflow.state.PipelineState` and
return a partial state update. Exceptions are caught by a guard wrapper and
recorded in ``state["error"]`` instead of crashing the whole run, so a batch
over hundreds of figures survives one bad image.
"""

from dataclasses import dataclass
from typing import Callable, Literal

from langchain_core.language_models.chat_models import BaseChatModel

from kg_pipeline.workflow.state import PipelineState
from kg_pipeline.llm.factory import get_chat_model
from kg_pipeline.settings import Settings, get_settings

NodeFn = Callable[[PipelineState], dict]
NodeKind = Literal["agent", "tool"]


@dataclass(frozen=True)
class PipelineDeps:
    """Shared dependencies handed to every node factory at build time.

    ``vision_model`` serves the image-facing agents, ``text_model`` the
    text-only ones; deterministic tools ignore both. Extend this dataclass
    (e.g. with a Neo4j driver) when a new node needs a shared resource.
    """

    settings: Settings
    vision_model: BaseChatModel
    text_model: BaseChatModel


@dataclass(frozen=True)
class NodeSpec:
    name: str
    kind: NodeKind
    description: str
    factory: Callable[[PipelineDeps], NodeFn]


_REGISTRY: dict[str, NodeSpec] = {}


def register_node(name: str, kind: NodeKind, description: str = ""):
    """Decorator registering a node factory under a unique pipeline-wide name."""

    def decorate(factory: Callable[[PipelineDeps], NodeFn]):
        if name in _REGISTRY:
            raise ValueError(f"Duplicate node registration: {name!r}")
        _REGISTRY[name] = NodeSpec(name=name, kind=kind, description=description, factory=factory)
        return factory

    return decorate


def discover_nodes() -> dict[str, NodeSpec]:
    """Imports the agent/tool packages (triggering registration) and returns
    the full registry. Call this before looking nodes up."""
    import kg_pipeline.agents  # noqa: F401  (auto-imports register all agents)
    import kg_pipeline.tools  # noqa: F401  (auto-imports register all tools)

    return dict(_REGISTRY)


def build_node(name: str, deps: PipelineDeps) -> NodeFn:
    """Instantiates a registered node and wraps it in the error guard."""
    try:
        spec = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none — did discover_nodes() run?>"
        raise KeyError(f"Unknown node {name!r}. Registered nodes: {known}") from None
    return _guarded(name, spec.factory(deps))


def _guarded(name: str, fn: NodeFn) -> NodeFn:
    """Converts node exceptions into a state update instead of a crash.

    Routing functions never branch on ``error`` (a stale error from an earlier
    node would poison later routing); they check for the presence of the data
    they need. ``error`` is informational and ends up in the perf log.
    """

    def wrapped(state: PipelineState) -> dict:
        try:
            return fn(state)
        except Exception as e:  # noqa: BLE001 -- batch runs must survive any node failure
            return {"error": f"[{name}] {type(e).__name__}: {e}"}

    wrapped.__name__ = name
    return wrapped


def build_deps(settings: Settings | None = None, *, require_keys: bool = True) -> PipelineDeps:
    """Builds the shared dependency bundle from settings/env.

    ``require_keys=False`` constructs the chat models with placeholder API
    keys -- enough to compile and visualize the graph without credentials.
    """
    settings = settings or get_settings()
    allow_missing = not require_keys
    vision_model = get_chat_model(
        settings.vision_provider,
        settings,
        model=settings.vision_model,
        allow_missing_key=allow_missing,
    )
    text_model = get_chat_model(
        settings.text_provider,
        settings,
        model=settings.text_model,
        allow_missing_key=allow_missing,
        role="text",
    )
    return PipelineDeps(settings=settings, vision_model=vision_model, text_model=text_model)
