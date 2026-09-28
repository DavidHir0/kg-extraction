"""Deterministic tool nodes (pure Python, no LLM calls -- free and testable).

Every module in this package is auto-imported below, so a ``@register_node``
decorator at module level is all it takes to make a new tool available to
the graph builder -- drop a file in, no imports to maintain by hand.
"""

import importlib
import pkgutil

for _module in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module.name}")
