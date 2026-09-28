"""Tool -- ``normalize_terms``: taxonomy lookup + the deterministic half of
Golden Rule 7 (LaTeX stripping).

Two responsibilities, both post-normalizer safety nets over what the LLM
wrote into ``properties``:

1. **Terminology**: maps variant spellings to the canonical values defined in
   ``prompts/normalization_dict.md`` ("relu" -> "Relu", "bn" ->
   "BatchNormalization", ...). The markdown file is the single source of
   truth -- it is parsed here at build time AND appended verbatim to the
   normalizer/corrector prompts, so extending the taxonomy means editing one
   file and touching no code.
2. **LaTeX stripping**: property *values* must be machine-readable, so LaTeX
   markup is stripped from value-carrying fields ("$\\alpha$" -> "alpha").
   ``raw_text`` is never touched (it lives outside ``properties``), and the
   human-readable ``name``/``note`` fields are left alone.
"""

import re

from kg_pipeline.workflow.state import PipelineState
from kg_pipeline.prompts.loader import load_normalization_dict
from kg_pipeline.registry import NodeFn, PipelineDeps, register_node
from kg_pipeline.schema.graph import ArchitectureGraph, NodeModel

# Value-carrying property fields that get LaTeX-stripped. Deliberately
# excludes `name` and `note` (free-form human text).
STRIP_FIELDS = {
    "symbol", "algorithm", "mechanism", "dimensions", "kernel", "stride",
    "dilation", "channels", "units", "features", "heads", "dim",
    "hidden_size", "vocab_size", "repetition", "direction", "modality",
    "dtype",
}

# (sub_type, property) -> {variant_lowercase: canonical}
LookupTable = dict[tuple[str, str], dict[str, str]]


def strip_latex(value: str) -> str:
    """"$3 \\times 3$" -> "3 x 3"; "$\\alpha$" -> "alpha"."""
    value = value.replace("$", "")
    value = re.sub(r"\\times", "x", value)
    value = re.sub(r"\\(\w+)", r"\1", value)
    value = value.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", value).strip()


def parse_normalization_dict(markdown: str) -> LookupTable:
    """Parses the markdown tables of ``normalization_dict.md``.

    Handles both table layouts in the file: 5-column
    ``| Subtype | Target Property | Normalized Value | Shorthands | Notes |``
    and 6-column (same, with a leading ``Class`` column).
    """
    table: LookupTable = {}
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip().strip("`").strip() for c in stripped.strip("|").split("|")]
        if len(cells) == 5:
            sub_type, prop, canonical, shorthands = cells[0], cells[1], cells[2], cells[3]
        elif len(cells) == 6:
            sub_type, prop, canonical, shorthands = cells[1], cells[2], cells[3], cells[4]
        else:
            continue
        if not canonical or "---" in sub_type or sub_type in ("Subtype", "Class"):
            continue
        prop = prop.split("(")[0].strip().strip("`")  # "`name` (Global key)" -> "name"
        variants = table.setdefault((sub_type, prop), {})
        variants[strip_latex(canonical).lower()] = canonical
        for shorthand in shorthands.split(","):
            shorthand = strip_latex(shorthand).lower()
            if shorthand:
                variants[shorthand] = canonical
    return table


def load_lookup_table() -> LookupTable:
    return parse_normalization_dict(load_normalization_dict())


def _normalize_properties(sub_type: str, properties: dict, table: LookupTable) -> dict:
    for key, value in properties.items():
        if key == "sequence" and isinstance(value, list):
            for element in value:
                if isinstance(element, dict):
                    element["properties"] = _normalize_properties(
                        element.get("sub_type", ""), element.get("properties", {}), table
                    )
            continue
        if isinstance(value, str):
            cleaned = strip_latex(value) if key in STRIP_FIELDS else value
            canonical = table.get((sub_type, key), {}).get(strip_latex(value).lower())
            properties[key] = canonical or cleaned
        elif isinstance(value, list) and key in STRIP_FIELDS:
            properties[key] = [strip_latex(v) if isinstance(v, str) else v for v in value]
    return properties


def normalize_terms(graph: ArchitectureGraph, table: LookupTable) -> ArchitectureGraph:
    """Pure function: returns a new graph with canonicalized property values."""
    normalized_nodes = []
    for node in graph.nodes:
        properties = _normalize_properties(
            node.sub_type, node.properties.model_dump(exclude_none=True), table
        )
        normalized_nodes.append(
            NodeModel.model_validate(
                {
                    "id": node.id,
                    "class": node.class_,
                    "sub_type": node.sub_type,
                    "raw_text": node.raw_text,
                    # Rebuilt from scratch rather than copied, so every
                    # non-property field must be listed here or it is silently
                    # dropped (this ate ``style_id`` on its first outing).
                    "style_id": node.style_id,
                    "properties": properties,
                }
            )
        )
    return graph.model_copy(update={"nodes": normalized_nodes})


@register_node(
    "normalize_terms",
    kind="tool",
    description="Canonicalizes property values via normalization_dict.md and strips LaTeX.",
)
def make_node(deps: PipelineDeps) -> NodeFn:
    table = load_lookup_table()

    def _node(state: PipelineState) -> dict:
        return {"graph": normalize_terms(state["graph"], table)}

    return _node
