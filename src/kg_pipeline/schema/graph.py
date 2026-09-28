"""Pydantic schemas shared by every node in the KG-extraction pipeline.

The vision agents output :class:`RawTopology` -- an unstructured, literal
transcription of what's on the page. The normalization/correction agents fit
that into :class:`ArchitectureGraph`, the Universal Neural Architecture Graph
Schema (v5.8, see ``prompts/schema_v5.8.md``). The deterministic tools
(``kg_pipeline.tools``) transform these models without any LLM call.

``ArchitectureGraph`` and friends are a close port of the Pydantic models in
the original ``kg_generation/utils/schema_enforcer.py``, which already
defined the schema in a structured-output-ready shape.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Pass 1: raw, unstructured vision extraction ---


class RawNode(BaseModel):
    id: str
    raw_text_elements: list[str] = Field(default_factory=list)
    style_id: str | None = None
    # Ordered legend styles when this shape is a STACK of legend-coloured slabs
    # (e.g. a U-Net block drawn as green Conv + purple BN + cyan ReLu). Nothing
    # downstream reads it; it stays because it is part of the output format the
    # vision model is asked to fill, and removing it would change that request.
    style_ids: list[str] = Field(default_factory=list)


class RawEdge(BaseModel):
    source: str
    target: str
    style: Literal["solid", "dashed"] = "solid"
    # Legend style id of the ARROW itself (e.g. "style_pink_arrow"). Nothing
    # downstream reads it; like `RawNode.style_ids` it stays because it is part
    # of the output format the vision model is asked to fill.
    #
    # Keep notes about these classes as `#` comments, never docstrings: Pydantic
    # copies docstrings into the JSON schema, so they would be sent to the model
    # on every call and change what it is asked.
    style_id: str | None = None


class RawGroup(BaseModel):
    id: str
    label: str
    boundary_type: Literal["explicit_box", "implicit_header", "shaded_region"]
    member_node_ids: list[str] = Field(default_factory=list)


class RawTopology(BaseModel):
    legend_map: dict[str, str] = Field(default_factory=dict)
    raw_nodes: list[RawNode] = Field(default_factory=list)
    raw_edges: list[RawEdge] = Field(default_factory=list)
    raw_groups: list[RawGroup] = Field(default_factory=list)


# --- Pass 2/3: the normalized, schema-conformant Architecture Graph (v5.8) ---

NodeClass = Literal["DATA", "LAYER", "MODIFIER", "JUNCTION", "MACRO"]


class SequenceElement(BaseModel):
    """A single step inside a fused composite node's `properties.sequence`."""

    model_config = ConfigDict(populate_by_name=True)

    class_: NodeClass = Field(alias="class")
    sub_type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    # The operation's own written name ("Conv 3x3", "ReLu"). When present, the
    # sanitizer uses it as the exploded step's raw_text instead of the parent
    # block's text + "(Step N)". Optional: absent -> the parent-derived naming.
    label: str | None = None


class NodeProperties(BaseModel):
    # Global
    name: str | None = None
    symbol: str | None = None
    note: str | None = None

    # DATA
    # Item type includes None: models legitimately emit ``null`` for an
    # unknown axis (e.g. R^{N x D} with unbound N -> [null, "D"]).
    dimensions: list[int | str | None] | None = None
    dtype: str | None = None
    modality: str | None = None

    # LAYER
    kernel: list[int | str] | None = None
    channels: int | str | None = None
    stride: int | list[int | str] | str | None = None
    dilation: int | list[int | str] | str | None = None
    groups: int | str | None = None  # str for symbolic counts, e.g. "C"
    units: int | str | None = None
    features: int | str | None = None
    heads: int | str | None = None
    dim: int | str | None = None
    hidden_size: int | str | None = None
    direction: str | None = None
    vocab_size: int | str | None = None
    algorithm: str | None = None  # shared with MODIFIER

    # MODIFIER
    rate: float | None = None

    # JUNCTION & MACRO
    mechanism: str | None = None
    repetition: int | str | None = None
    derived_from_definition: str | None = None

    # The composite sequence array (exploded by pass 3's sanitizer)
    sequence: list[SequenceElement] | None = None


class NodeModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    class_: NodeClass = Field(alias="class")
    sub_type: str
    raw_text: str
    # The legend style this shape was drawn in, carried through from
    # ``RawNode.style_id``, so a colour does not have to be pasted into the
    # node's name to survive normalization. Populated deterministically by the
    # ``attach_styles`` tool; unscored, since the benchmark labels have no style
    # field.
    style_id: str | None = None
    properties: NodeProperties = Field(default_factory=NodeProperties)


class EdgeProperties(BaseModel):
    labels: list[str] = Field(default_factory=list)


class EdgeModel(BaseModel):
    source: str
    target: str
    role: Literal["forward", "skip", "condition", "backward"]
    properties: EdgeProperties = Field(default_factory=EdgeProperties)


class GroupModel(BaseModel):
    id: str
    type: Literal["container", "subgraph_definition"]
    label: str
    boundary_type: Literal["explicit_box", "implicit_header"]
    member_node_ids: list[str] = Field(default_factory=list)


class ArchitectureGraph(BaseModel):
    architecture_name: str = ""
    extraction_version: str = "5.8"
    nodes: list[NodeModel] = Field(default_factory=list)
    edges: list[EdgeModel] = Field(default_factory=list)
    groups: list[GroupModel] = Field(default_factory=list)
