"""QA plugin components."""

from contextrouter.core.config import Config
from contextrouter.modules.ingestion.rag.core.plugins import IngestionPlugin
from contextrouter.modules.ingestion.rag.core.registry import register_plugin
from contextrouter.modules.ingestion.rag.core.types import (
    IngestionMetadata,
    RawData,
    ShadowRecord,
)

from .analyzer import QuestionAnalyzer
from .speaker import SpeakerProcessor
from .taxonomy_mapper import TaxonomyMapper
from .transformer import QATransformer


@register_plugin("qa")
class QAPlugin(IngestionPlugin):
    """QA ingestion plugin using refactored components."""

    def __init__(self):
        """Initialize the QA plugin."""
        self._initialized = False
        self._speaker_processor = None
        self._question_analyzer = None
        self._taxonomy_mapper = None
        self._transformer = None

    def _ensure_initialized(self, config: Config):
        """Lazy initialization of components."""
        if not self._initialized:
            self._speaker_processor = SpeakerProcessor(config)
            self._question_analyzer = QuestionAnalyzer(config)
            self._taxonomy_mapper = TaxonomyMapper()
            self._transformer = QATransformer(config)
            self._initialized = True

    @property
    def speaker_processor(self):
        if self._speaker_processor is None:
            raise RuntimeError("Plugin not initialized. Call _ensure_initialized() first.")
        return self._speaker_processor

    @property
    def question_analyzer(self):
        if self._question_analyzer is None:
            raise RuntimeError("Plugin not initialized. Call _ensure_initialized() first.")
        return self._question_analyzer

    @property
    def taxonomy_mapper(self):
        if self._taxonomy_mapper is None:
            raise RuntimeError("Plugin not initialized. Call _ensure_initialized() first.")
        return self._taxonomy_mapper

    @property
    def transformer(self):
        if self._transformer is None:
            raise RuntimeError("Plugin not initialized. Call _ensure_initialized() first.")
        return self._transformer

    @property
    def source_type(self) -> str:
        """Return the source type this plugin handles."""
        return "qa"

    def transform(self, raw_data: RawData, config: Config) -> list[ShadowRecord]:
        """Transform QA data into shadow records."""
        self._ensure_initialized(config)
        # For now, delegate to the parent class implementation
        # TODO: Implement proper QA transformation using refactored components
        return super().transform(raw_data, config)

    def load(self, assets_path: str) -> list[RawData]:
        """Load QA data from assets path."""
        from contextrouter.core.config import get_core_config

        config = get_core_config()
        self._ensure_initialized(config)
        # For now, delegate to the parent class implementation
        # TODO: Implement proper QA loading using refactored components
        return super().load(assets_path)


__all__ = [
    "QuestionAnalyzer",
    "SpeakerProcessor",
    "TaxonomyMapper",
    "QATransformer",
    "QAPlugin",
]
