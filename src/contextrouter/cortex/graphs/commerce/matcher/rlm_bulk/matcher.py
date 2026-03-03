"""Core bulk matcher class for Deep product matching."""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from contextrouter.core import Config, get_core_config
from contextrouter.modules.models import model_registry
from contextrouter.modules.models.types import ModelRequest, TextPart

from .fallback import fallback_chunked_match
from .parser import deduplicate_matches, parse_rlm_response
from .prompts import build_matching_prompt
from .types import BulkMatchResult

logger = logging.getLogger(__name__)


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
        rlm_api_key: str = "",
        rlm_model: str = "",
    ):
        self._config = config or get_core_config()
        self._environment = environment
        self._verbose = verbose
        self._log_dir = log_dir
        self._rlm_api_key = rlm_api_key
        self._rlm_model = rlm_model or "gemini-2.5-flash"

    def _resolve_rlm_env(self) -> tuple[str, str]:
        """Resolve RLM model ID and API key."""
        model_id = f"rlm/{self._rlm_model}"

        if self._rlm_api_key:
            logger.info("RLM model: %s (project-scoped key)", model_id)
            return model_id, self._rlm_api_key

        if self._config.openai.api_key:
            logger.info("RLM: no project key, using router global OPENAI_API_KEY → rlm/gpt-5-mini")
            return "rlm/gpt-5-mini", self._config.openai.api_key

        logger.warning("No RLM API key configured. Matcher will likely fail.")
        return model_id, ""

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

        rlm_model, api_key = self._resolve_rlm_env()

        create_kwargs = {
            "config": self._config,
            "environment": self._environment,
            "verbose": self._verbose,
            "log_dir": self._log_dir,
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
            "supplier_products": supplier_products,
            "site_products": site_products,
            "taxonomies": taxonomies or {},
            "manual_matches": manual_matches or [],
            "wrong_pairs": wrong_pairs or [],
            "out_path": out_path,
            "matches": [],
        }

        prompt = build_matching_prompt(
            supplier_products, site_products, taxonomies, manual_matches, wrong_pairs, out_path
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

        matches = final_matched + unmatched_list

        matched_supplier_ids = {m.supplier_id for m in matches if m.site_id}
        unmatched = [
            p
            for p in supplier_products
            if str(p.get("id", p.get("sku", ""))) not in matched_supplier_ids
        ]

        match_types = {}
        for m in matches:
            match_types[m.match_type] = match_types.get(m.match_type, 0) + 1

        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        total_tokens = (usage.total_tokens if usage else 0) or (input_tokens + output_tokens)

        result = BulkMatchResult(
            total_supplier=len(supplier_products),
            total_site=len(site_products),
            matches=matches,
            unmatched=unmatched,
            stats={
                "match_types": match_types,
                "match_rate": len([m for m in matches if m.site_id]) / len(supplier_products)
                if supplier_products
                else 0,
                "high_confidence": len([m for m in matches if m.confidence >= 0.9]),
                "medium_confidence": len([m for m in matches if 0.7 <= m.confidence < 0.9]),
                "low_confidence": len([m for m in matches if m.confidence < 0.7 and m.site_id]),
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )

        match_rate_pct = format(result.stats["match_rate"], ".1%")
        logger.info(
            "RLM Bulk Matcher complete: %s matched, %s unmatched",
            match_rate_pct,
            len(result.unmatched),
        )

        return result
