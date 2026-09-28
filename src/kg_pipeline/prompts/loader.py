"""Loads agent system prompts from the top-level ``prompts/`` directory.

Layout:
    prompts/
      schema_v5.8.md          -- the Universal Neural Architecture Graph Schema
      normalization_dict.md   -- terminology lookup table (also parsed by the
                                 deterministic ``normalize_terms`` tool)
      agents/<name>.txt       -- one system prompt per LLM agent

Agents whose output must conform to the v5.8 schema (``normalize``,
``correct``) load their prompt with ``with_rulebook=True``, which appends the
schema and dictionary.
"""

from pathlib import Path

from kg_pipeline.settings import PROMPTS_DIR

AGENTS_DIR = PROMPTS_DIR / "agents"
SCHEMA_FILE = PROMPTS_DIR / "schema_v5.8.md"
DICTIONARY_FILE = PROMPTS_DIR / "normalization_dict.md"


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def load_agent_prompt(name: str, *, with_rulebook: bool = False) -> str:
    """Loads ``prompts/agents/<name>.txt``, optionally appending the schema
    definition and normalization dictionary as a rulebook.
    """
    base = _read(AGENTS_DIR / f"{name}.txt")
    if not with_rulebook:
        return base
    return f"{base}\n\n{_read(SCHEMA_FILE)}\n\n{_read(DICTIONARY_FILE)}"


def load_normalization_dict() -> str:
    """Raw markdown of the normalization dictionary, for deterministic parsing."""
    return _read(DICTIONARY_FILE)
