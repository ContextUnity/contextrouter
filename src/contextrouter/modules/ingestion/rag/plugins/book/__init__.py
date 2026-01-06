"""Book plugin components."""

from .analyzer import BookAnalyzer
from .extractor import PDFExtractor
from .normalizer import ContentNormalizer

# Import the main plugin class
from .plugin import BookPlugin
from .transformer import BookTransformer

__all__ = [
    "BookAnalyzer",
    "PDFExtractor",
    "ContentNormalizer",
    "BookTransformer",
    "BookPlugin",
]
