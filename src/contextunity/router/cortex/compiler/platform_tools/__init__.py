"""Platform tools package — declarative registration table.

Each tool file contains only business logic + a frozen Pydantic config class.
Executor wrapping (config injection, error handling) is handled by
``_base.make_platform_executor`` — defined ONCE, not per-file.

Usage:
    from contextunity.router.cortex.compiler.platform_tools import (
        register_all_platform_tools,
    )
    register_all_platform_tools(registry)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..platform_registry import PlatformToolRegistry


def register_all_platform_tools(registry: PlatformToolRegistry) -> None:
    """Register all platform tools from the declarative table.

    Generic tools use ``make_platform_executor`` (single error-handling path).
    Tools with non-standard signatures provide a custom ``_platform_execute``.
    """
    # ── RAG pipeline tools (7 generic + 1 custom) ───────────────────
    from . import (
        extract,
        formatter,
        generate,
        ground,
        intent,
        no_results,
        reflect,
        retrieve,
        suggest,
        synthesizer,
        web_search,
    )
    from .helpers.base import make_platform_executor
    from .helpers.registration import ToolRegistrationSpec, register_tool_specs

    _GENERIC_RAG = [
        # (binding, function, config_class, scopes)
        (
            "router_extract_query",
            extract.extract_user_query,
            extract.ExtractQueryConfig,
            ["router:execute"],
        ),
        (
            "router_detect_intent",
            intent.detect_intent,
            intent.DetectIntentConfig,
            ["router:execute"],
        ),
        (
            "router_retrieve",
            retrieve.retrieve_documents,
            retrieve.RetrieveConfig,
            ["router:execute", "brain:read"],
        ),
        (
            "router_ground",
            ground.generate_with_native_grounding,
            ground.GroundConfig,
            ["router:execute"],
        ),
        (
            "router_generate",
            generate.generate_response,
            generate.GenerateConfig,
            ["router:execute"],
        ),
        ("router_reflect", reflect.reflect_interaction, reflect.ReflectConfig, ["router:execute"]),
        (
            "router_suggest",
            suggest.generate_search_suggestions,
            suggest.SuggestConfig,
            ["router:execute"],
        ),
        (
            "router_format_output",
            formatter.format_output,
            formatter.FormatterConfig,
            ["router:execute"],
        ),
        (
            "router_synthesize_results",
            synthesizer.synthesize_results,
            synthesizer.SynthesizerConfig,
            ["router:execute"],
        ),
        (
            "router_web_search",
            web_search.perform_web_search,
            web_search.WebSearchConfig,
            ["router:execute"],
        ),
    ]

    register_tool_specs(
        registry,
        [
            ToolRegistrationSpec(
                binding=binding,
                executor=make_platform_executor(func, binding),
                config_schema=config_cls,
                required_scopes=scopes,
            )
            for binding, func, config_cls, scopes in _GENERIC_RAG
        ],
    )

    from .helpers.adapters import no_results_adapter

    register_tool_specs(
        registry,
        [
            ToolRegistrationSpec(
                binding="router_no_results",
                executor=make_platform_executor(no_results_adapter, "router_no_results"),
                config_schema=no_results.NoResultsConfig,
                required_scopes=["router:execute"],
            )
        ],
    )

    # ── SQL pipeline tools ──────────────────────────────────────────
    # SQL tools use node factories — adapters create the factory node and call it.
    from . import sql_visualizer
    from .helpers import sql_executor, sql_planner, sql_verifier
    from .helpers.adapters import (
        sql_execute_adapter,
        sql_plan_adapter,
        sql_verify_adapter,
        sql_visualize_adapter,
    )

    _SQL_TOOLS = [
        ("router_sql_plan", sql_plan_adapter, sql_planner.SqlPlannerConfig),
        ("router_sql_execute", sql_execute_adapter, sql_executor.SqlExecutorConfig),
        ("router_sql_verify", sql_verify_adapter, sql_verifier.SqlVerifierConfig),
        ("router_sql_visualizer", sql_visualize_adapter, sql_visualizer.SqlVisualizerConfig),
    ]

    register_tool_specs(
        registry,
        [
            ToolRegistrationSpec(
                binding=binding,
                executor=make_platform_executor(adapter, binding, inject="sql"),
                config_schema=config_cls,
                required_scopes=["router:execute"],
            )
            for binding, adapter, config_cls in _SQL_TOOLS
        ],
    )

    # ── Content / news tools ────────────────────────────────────────
    # router_classify, router_generate_content, router_review_content,
    # router_filter_content, router_plan_content, router_match_semantic
    # Required by: news_pipeline.yaml, enricher.yaml, gardener.yaml
    from .content import register_router_content_tools

    register_router_content_tools(registry)

    # ── RLM process tool ────────────────────────────────────────────
    # router_rlm_process — Required by: rlm_bulk_matcher.yaml
    from .rlm import register_router_rlm_tools

    register_router_rlm_tools(registry)

    # ── Language tool ───────────────────────────────────────────────
    # language_tool — spell/grammar checking for bilingual enrichment
    from .language import register_language_tools

    register_language_tools(registry)

    # ── Brain service tools ─────────────────────────────────────────
    # brain_search, brain_memory_read/write, brain_blackboard_*, brain_kg_query, brain_upsert
    from .brain import register_brain_tools

    register_brain_tools(registry)

    # ── Shield service tools ────────────────────────────────────────
    # shield_scan
    from .shield import register_shield_tools

    register_shield_tools(registry)

    # ── Worker service tools ────────────────────────────────────────
    # worker_start_workflow, worker_get_status, worker_execute_code, worker_register_schedules
    from .worker import register_worker_tools

    register_worker_tools(registry)

    # ── Ingest / file download tools ────────────────────────────────
    # router_file_download — generic HTTP file download for LangGraph agents
    from .ingest import register_ingest_tools

    register_ingest_tools(registry)


__all__ = ["register_all_platform_tools"]
