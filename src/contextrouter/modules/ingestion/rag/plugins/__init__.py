"""Ingestion plugins for various content types."""

from .book.plugin import BookPlugin
from .qa import QAPlugin
from .text import TextPlugin
from .video import VideoPlugin
from .web import WebPlugin

__all__ = [
    "VideoPlugin",
    "BookPlugin",
    "QAPlugin",
    "WebPlugin",
    "TextPlugin",
]
