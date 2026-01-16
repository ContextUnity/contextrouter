"""Shared CLI utilities for ingestion commands."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from contextrouter.core import Config, get_core_config
from contextrouter.modules.ingestion.rag.settings import RagIngestionConfig

ALL_TYPES = ["video", "book", "qa", "web", "knowledge"]

logger = logging.getLogger(__name__)


def load_core_cfg(config: RagIngestionConfig | None = None) -> Config:
    """Load contextrouter core config for LLM-backed ingestion steps.

    CLI owns env/TOML loading. Library ingestion modules must not read env directly.
    """
    core_cfg = get_core_config()
    if config is not None:
        _apply_ingestion_model_overrides(core_cfg, config)
        _validate_required_models(core_cfg)
    return core_cfg


def _apply_ingestion_model_overrides(core_cfg: Config, cfg: RagIngestionConfig) -> None:
    models = cfg.models
    if models.ingestion_taxonomy_model:
        core_cfg.models.ingestion.taxonomy.model = models.ingestion_taxonomy_model
    if models.ingestion_preprocess_model:
        core_cfg.models.ingestion.preprocess.model = models.ingestion_preprocess_model
    if models.ingestion_graph_model:
        core_cfg.models.ingestion.graph.model = models.ingestion_graph_model
    if models.ingestion_persona_model:
        core_cfg.models.ingestion.persona.model = models.ingestion_persona_model
    if models.ingestion_json_model:
        core_cfg.models.ingestion.json_model.model = models.ingestion_json_model
    if models.ingestion_ner_model and not cfg.enrichment.ner.model:
        cfg.enrichment.ner.model = models.ingestion_ner_model
    if models.ingestion_keyphrases_model and not cfg.enrichment.keyphrases.model:
        cfg.enrichment.keyphrases.model = models.ingestion_keyphrases_model
    if cfg.local.vllm_base_url:
        core_cfg.local.vllm_base_url = cfg.local.vllm_base_url
    if cfg.local.ollama_base_url:
        core_cfg.local.ollama_base_url = cfg.local.ollama_base_url


def _validate_required_models(core_cfg: Config) -> None:
    if not core_cfg.models.ingestion.json_model.model.strip():
        raise click.ClickException("models.ingestion.json_model.model is required")


def suppress_noisy_loggers() -> None:
    """Suppress spammy third-party loggers."""
    # Make ingestion CLI output deterministic: Click prints progress; Python logging stays quiet.
    logging.getLogger().setLevel(logging.WARNING)

    # But keep our ingestion progress visible (file-level / batch-level logs).
    logging.getLogger("contextrouter.modules.ingestion.rag").setLevel(logging.INFO)

    # Google GenAI SDK AFC spam: "INFO AFC is enabled with max remote calls: 10."
    for name in [
        "google",
        "google.genai",
        "google.genai._api_client",
        "google.genai.types",
        "google.api_core",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)
    # httpx noise
    logging.getLogger("httpx").setLevel(logging.WARNING)


def stage_banner(name: str) -> None:
    """Print a clear stage banner."""
    click.echo(click.style(f"\n{'─' * 60}", fg="cyan"))
    click.echo(click.style(f"  STAGE: {name}", fg="cyan", bold=True))
    click.echo(click.style(f"{'─' * 60}", fg="cyan"))


def coerce_types(only_types: tuple[str, ...] | None) -> list[str]:
    """Convert tuple of types to list, or return all types if None."""
    if only_types:
        return [t.lower() for t in only_types]
    return list(ALL_TYPES)


def missing(msg: str, *, hint: str) -> None:
    """Raise ClickException with helpful hint (formatted by Rich in app.py)."""
    raise click.ClickException(f"{msg}\n💡 Hint: {hint}")


def require_clean_text(paths: dict[str, Path], types: list[str]) -> None:
    """Check that CleanText files exist for given types."""
    missing_types = [t for t in types if not (paths["clean_text"] / f"{t}.jsonl").exists()]
    if missing_types:
        missing(
            f"Missing CleanText for: {', '.join(missing_types)}",
            hint="contextrouter ingest preprocess --type " + " --type ".join(missing_types),
        )


def require_taxonomy(paths: dict[str, Path]) -> None:
    """Check that taxonomy.json exists."""
    if not paths["taxonomy"].exists():
        missing("Missing taxonomy.json", hint="contextrouter ingest taxonomy")


def require_graph(paths: dict[str, Path]) -> None:
    """Check that knowledge_graph.pickle exists."""
    if not paths["graph"].exists():
        missing("Missing knowledge_graph.pickle", hint="contextrouter ingest graph")


def require_shadow(paths: dict[str, Path], types: list[str]) -> None:
    """Check that shadow records exist for given types."""
    missing_types = [t for t in types if not (paths["shadow"] / f"{t}.jsonl").exists()]
    if missing_types:
        missing(
            f"Missing shadow records for: {', '.join(missing_types)}",
            hint="contextrouter ingest shadow --type " + " --type ".join(missing_types),
        )


def require_exports(paths: dict[str, Path], types: list[str]) -> None:
    """Check that JSONL exports exist for given types."""
    missing_types: list[str] = []
    for t in types:
        d = paths["jsonl"] / t
        if not d.exists() or not any(d.glob("*.jsonl")):
            missing_types.append(t)
    if missing_types:
        missing(
            f"Missing JSONL exports for: {', '.join(missing_types)}",
            hint="contextrouter ingest export --type " + " --type ".join(missing_types),
        )


__all__ = [
    "ALL_TYPES",
    "logger",
    "coerce_types",
    "load_core_cfg",
    "missing",
    "require_clean_text",
    "require_exports",
    "require_graph",
    "require_shadow",
    "require_taxonomy",
    "stage_banner",
    "suppress_noisy_loggers",
]
