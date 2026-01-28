"""
news_engine - AI-powered news processing graph.

Provides four main capabilities:
- harvest: Fetch news via Perplexity/Serper
- archivist: Filter, validate, deduplicate news
- showrunner: Plan editorial content
- agents: Generate posts with persona voices

Use intent="full_pipeline" to run all steps in sequence.
"""

from .graph import build_news_engine_graph

__all__ = ["build_news_engine_graph"]
