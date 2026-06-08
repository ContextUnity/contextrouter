"""Tests for RegisterManifest default_graph validation."""

from __future__ import annotations

import pytest
from contextunity.core.exceptions import ConfigurationError

from contextunity.router.service.mixins.registration import _validate_default_graph
from contextunity.router.service.payloads import GraphEntry

_GRAPH = GraphEntry.model_validate({"template": "yaml:retrieval_augmented"})


def test_multi_graph_requires_default_graph():
    entries = {"a": _GRAPH, "b": _GRAPH}

    with pytest.raises(ConfigurationError, match="must declare a valid default_graph"):
        _validate_default_graph("", entries)


def test_default_graph_must_exist_in_map():
    entries = {"a": _GRAPH}

    with pytest.raises(ConfigurationError, match="not present in graph map"):
        _validate_default_graph("missing", entries)
