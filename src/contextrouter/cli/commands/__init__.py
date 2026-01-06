"""Automatic CLI command discovery.

This module scans the parent `cli` package for modules and packages that register
commands in the CLI registry.
"""

from __future__ import annotations

import importlib
import pkgutil

# To avoid circular imports, we only import from contextrouter.cli.*
# but we need to know WHICH modules to import.


def discover_builtin_commands() -> None:
    """Discover and import all modules in the cli package to trigger registration."""
    # The 'cli' package is the parent of this one.
    import contextrouter.cli as cli_pkg

    package_path = cli_pkg.__path__
    prefix = cli_pkg.__name__ + "."

    for _, modname, ispkg in pkgutil.iter_modules(package_path, prefix):
        # Skip app, registry, and this package itself to avoid cycles/noise
        if modname.endswith((".app", ".registry", ".commands", ".__main__")):
            continue

        # Importing triggers @register_command at module level
        importlib.import_module(modname)


# Trigger discovery on import
discover_builtin_commands()
