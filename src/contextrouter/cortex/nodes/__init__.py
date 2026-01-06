"""LangGraph nodes package.

In agent-mode, nodes are registered classes (`BaseAgent`) in submodules.
In direct-mode, pure function steps live under `contextrouter.cortex.steps`.

We intentionally avoid importing submodules here to keep startup lazy.
"""

from __future__ import annotations

__all__ = []
