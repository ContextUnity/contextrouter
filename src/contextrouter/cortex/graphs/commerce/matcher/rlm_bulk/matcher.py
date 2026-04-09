"""Core bulk matcher class for Deep product matching."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any

from contextcore import get_context_unit_logger

from contextrouter.core import Config, get_core_config
from contextrouter.modules.models import model_registry
from contextrouter.modules.models.types import ModelRequest, TextPart

from .fallback import fallback_chunked_match
from .parser import deduplicate_matches, parse_rlm_response
from .prompts import build_matching_prompt
from .stages import run_exact_match_stage, run_normalized_match_stage
from .types import BulkMatchResult

logger = get_context_unit_logger(__name__)


class RLMBulkMatcher:
    """
    Deep product matcher using Recursive Language Models.

    Processes 50k+ supplier products against 10k site products
    in a single recursive pass with multi-factor comparison.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        environment: str = "docker",
        verbose: bool = False,
        log_dir: str | None = None,
        tenant_id: str = "",
        rlm_model: str = "",
        reasoning_effort: str = "none",
    ):
        self._config = config or get_core_config()
        self._environment = environment
        self._verbose = verbose
        self._log_dir = log_dir
        self._tenant_id = tenant_id
        self._rlm_model = rlm_model or "mercury-2"
        self._reasoning_effort = reasoning_effort

    def _resolve_rlm_env(self) -> tuple[str, str]:
        """Resolve RLM model ID and API key via Shield → env fallback."""
        model_id = f"rlm/{self._rlm_model}"
        provider = self._rlm_model.split("/")[0] if "/" in self._rlm_model else self._rlm_model

        # 1. Try Shield (project-scoped key)
        if self._tenant_id:
            try:
                from contextrouter.service.shield_client import shield_get_secret

                shield_path = f"{self._tenant_id}/api_keys/matcher_model"
                secret = shield_get_secret(shield_path, tenant_id=self._tenant_id)
                if secret:
                    logger.info("RLM model: %s (Shield key: %s)", model_id, shield_path)
                    return model_id, secret
            except Exception as e:
                logger.warning("Shield unavailable for RLM key: %s", e)

        # 2. Fallback to Router's provider env key
        provider_key = self._get_provider_env_key(provider)
        if provider_key:
            logger.info("RLM model: %s (Router env fallback)", model_id)
            return model_id, provider_key

        logger.warning("No RLM API key configured. Matcher will likely fail.")
        return model_id, ""

    def _get_provider_env_key(self, provider: str) -> str:
        """Get API key from Router's config for a given provider."""
        provider_config_map = {
            "inception": lambda: self._config.inception.api_key,
            "mercury-2": lambda: self._config.inception.api_key,
            "openai": lambda: self._config.openai.api_key,
            "anthropic": lambda: self._config.anthropic.api_key,
            "groq": lambda: self._config.groq.api_key,
        }
        getter = provider_config_map.get(provider)
        if getter:
            try:
                return getter() or ""
            except Exception:
                return ""
        return ""

    async def match_all(
        self,
        supplier_products: list[dict[str, Any]],
        site_products: list[dict[str, Any]],
        *,
        confidence_threshold: float = 0.7,
        max_output_tokens: int = 50000,
        taxonomies: dict[str, Any] | None = None,
        manual_matches: list[dict[str, Any]] | None = None,
        wrong_pairs: list[dict[str, Any]] | None = None,
    ) -> BulkMatchResult:
        """Match all supplier products to site products using RLM."""
        logger.info(
            "RLM Bulk Matcher: Starting match of %s supplier → %s site products",
            len(supplier_products),
            len(site_products),
        )

        all_matches = []
        unmatched_suppliers = supplier_products

        # Stage 1: Exact Match (EAN/SKU)
        exact_matches, unmatched_suppliers = run_exact_match_stage(
            unmatched_suppliers, site_products
        )
        all_matches.extend(exact_matches)

        # Stage 2: Normalized Match (Gardener fields)
        normalized_matches, unmatched_suppliers = run_normalized_match_stage(
            unmatched_suppliers, site_products
        )
        all_matches.extend(normalized_matches)

        if not unmatched_suppliers:
            logger.info("All products matched in Stages 1 & 2. Skipping RLM.")
            return BulkMatchResult(
                total_supplier=len(supplier_products),
                total_site=len(site_products),
                matches=all_matches,
                unmatched=[],
                stats={
                    "match_rate": len(all_matches) / max(len(supplier_products), 1),
                    "exact": len(exact_matches),
                    "normalized": len(normalized_matches),
                    "rlm": 0,
                },
            )

        rlm_model, api_key = self._resolve_rlm_env()

        create_kwargs = {
            "config": self._config,
            "environment": self._environment,
            "verbose": self._verbose,
            "log_dir": self._log_dir,
            "tenant_id": self._tenant_id,
            "shield_key_name": "matcher_model",
            "reasoning_effort": self._reasoning_effort
            if self._reasoning_effort != "none"
            else None,
        }
        if api_key:
            create_kwargs["api_key"] = api_key
            logger.info("RLM: passing api_key=%s... to create_llm", api_key[:8])

        try:
            model = model_registry.create_llm(rlm_model, **create_kwargs)
        except ImportError:
            logger.warning("RLM not installed. Falling back to chunked matching.")
            return await fallback_chunked_match(
                supplier_products, site_products, confidence_threshold
            )

        out_path = str(Path(tempfile.gettempdir()) / f"rlm_matches_{uuid.uuid4().hex[:8]}.json")
        repl_data = {
            "supplier_products": unmatched_suppliers,
            "site_products": site_products,
            "taxonomies": taxonomies or {},
            "manual_matches": manual_matches or [],
            "wrong_pairs": wrong_pairs or [],
            "out_path": out_path,
            "matches": [],
        }

        prompt = build_matching_prompt(
            unmatched_suppliers, site_products, taxonomies, manual_matches, wrong_pairs, out_path
        )

        logger.info("Executing RLM deep matching...")
        response = await model.generate(
            ModelRequest(
                system=(
                    "You are a product matching expert. Your task is to match supplier "
                    "products to site catalog products with high precision. Write Python "
                    "code in the REPL to analyze and match products efficiently."
                ),
                parts=[TextPart(text=prompt)],
                temperature=0.3,
                max_output_tokens=max_output_tokens,
            ),
            custom_tools=repl_data,
        )

        debug_path = str(Path(tempfile.gettempdir()) / "rlm_last_response.txt")
        with open(debug_path, "w") as f:
            f.write(response.text)

        raw_matches = parse_rlm_response(response.text, out_path=out_path)
        final_matched, unmatched_list = deduplicate_matches(raw_matches)

        rlm_matches = final_matched + unmatched_list
        all_matches.extend(rlm_matches)

        all_matches_deduped, all_unmatched_deduped = deduplicate_matches(all_matches)

        match_types = {}
        for m in all_matches_deduped:
            if m.site_id:
                match_types[m.match_type] = match_types.get(m.match_type, 0) + 1

        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        total_tokens = (usage.total_tokens if usage else 0) or (input_tokens + output_tokens)

        # Merge stats
        stats = {
            "exact": len(exact_matches),
            "normalized": len(normalized_matches),
            "rlm": len(final_matched),
            "match_rate": len([m for m in all_matches_deduped if m.site_id])
            / max(len(supplier_products), 1),
            "match_types": match_types,
            "high_confidence": len(
                [m for m in all_matches_deduped if m.confidence >= 0.9 and m.site_id]
            ),
            "medium_confidence": len(
                [m for m in all_matches_deduped if 0.7 <= m.confidence < 0.9 and m.site_id]
            ),
            "low_confidence": len(
                [m for m in all_matches_deduped if m.confidence < 0.7 and m.site_id]
            ),
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        result = BulkMatchResult(
            total_supplier=len(supplier_products),
            total_site=len(site_products),
            matches=all_matches_deduped,
            unmatched=all_unmatched_deduped,
            stats=stats,
        )

        match_rate_pct = format(stats["match_rate"], ".1%")
        logger.info(
            "RLM Bulk Matcher complete: %s matched, %s unmatched",
            match_rate_pct,
            len(result.unmatched),
        )

        return result
