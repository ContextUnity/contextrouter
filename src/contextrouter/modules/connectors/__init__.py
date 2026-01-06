"""Connectors (sources)."""

from __future__ import annotations

from .api import APIConnector
from .file import FileConnector
from .rss import RSSConnector
from .web import WebScraperConnector, WebSearchConnector

__all__ = [
    "APIConnector",
    "FileConnector",
    "RSSConnector",
    "WebSearchConnector",
    "WebScraperConnector",
]
