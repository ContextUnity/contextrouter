"""Recursive Language Model (RLM) provider.
RLMs are a task-agnostic inference paradigm where LMs can programmatically
examine, decompose, and recursively call themselves over input context.
This enables processing of near-infinite length contexts by:
1. Storing context as Python variable in REPL environment
2. Allowing LM to interact with and recurse over context programmatically
3. Breaking down complex tasks into smaller sub-LLM calls
Reference: https://arxiv.org/abs/2512.24601
Library: https://github.com/alexzhang13/rlm
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import override

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError
from contextunity.core.parsing import json_dumps

from contextunity.router.core import RouterConfig

from ..base import BaseLLM as BaseModel
from ..registry import model_registry
from ..types import (
    FinalTextEvent,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ProviderInfo,
    TextDeltaEvent,
    TextPart,
    UsageStats,
)
from .rlm_boundary import (
    RLMEngine,
    ensure_rlm_installed,
    load_rlm_engine,
    load_rlm_logger,
    rlm_response_text,
    rlm_usage_tokens,
)

logger = get_contextunit_logger(__name__)


@model_registry.register_llm("rlm", "*")
class RLMLLM(BaseModel):
    """Recursive Language Model provider.

    Wraps any base LLM with recursive REPL capabilities for handling
    large context sizes (50k+ items) that would cause context degradation.

    Key benefits:
    - GPT-5-mini with RLM outperforms GPT-5 on long-context tasks
    - Context stored as Python variable, not in prompt
    - Model can grep, filter, iterate, and recursively analyze

    Usage:
        model = model_registry.create_llm(
            "rlm/gpt-5-mini",
            config=config,
            environment="docker",  # or "local", "modal", "prime"
        )
    """

    def __init__(
        self,
        config: RouterConfig,
        *,
        model_name: str | None = None,
        environment: str = "docker",
        environment_kwargs: dict[str, object] | None = None,
        verbose: bool = False,
        log_dir: str | None = None,
        custom_tools: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize RLM wrapper.

        Args:
            config: contextunity.router configuration
            model_name: Base model to use (e.g., "gpt-5-mini", "claude-sonnet")
            environment: REPL environment type:
                - "local": Uses Python exec in the Router process (explicit opt-in only)
                - "docker": Isolated Docker container (default)
                - "modal": Modal.com sandboxes (cloud)
                - "prime": Prime Intellect sandboxes (cloud)
            environment_kwargs: Additional environment configuration
            verbose: Enable rich console output
            log_dir: Directory for trajectory logs (for visualization)
        """
        try:
            ensure_rlm_installed()
        except ImportError as e:
            raise ConfigurationError(
                "RLMLLM requires the 'rlm' package. Install with: pip install rlm or uv add rlm"
            ) from e

        resolved_name = (model_name or "gpt-5-mini").strip()
        super().__init__(provider="rlm", model_name=resolved_name)
        self._cfg: RouterConfig = config
        self._environment: str = environment
        self._verbose: bool = verbose

        # Determine backend from model name
        backend, extra_kwargs = self._infer_backend(self._model_name)
        backend_kwargs: dict[str, object] = {"model_name": self._model_name, **extra_kwargs}

        # Build RLM instance
        rlm_kwargs: dict[str, object] = {
            "backend": backend,
            "backend_kwargs": backend_kwargs,
            "environment": environment,
            "verbose": verbose,
            "max_timeout": 300.0,  # 5 min per brand — prevents indefinite runs
            "max_iterations": 15,  # cap REPL iterations
        }

        # Forward api_key to backend if provided (bypasses module-level env cache)
        if "api_key" in kwargs:
            backend_kwargs["api_key"] = kwargs.pop("api_key")

        if environment_kwargs:
            rlm_kwargs["environment_kwargs"] = environment_kwargs

        if custom_tools:
            rlm_kwargs["custom_tools"] = custom_tools

        # Add logger if log_dir specified
        if log_dir:
            rlm_kwargs["logger"] = load_rlm_logger(log_dir)

        self._rlm: RLMEngine = load_rlm_engine(**rlm_kwargs)
        self._custom_tools: dict[str, object] | None = custom_tools

        self._capabilities: ModelCapabilities = ModelCapabilities(
            supports_text=True,
            supports_image=False,  # RLM is text-focused
            supports_audio=False,
        )

    @staticmethod
    def _infer_backend(model_name: str) -> tuple[str, dict[str, object]]:
        """Infer the RLM backend and extra kwargs from the model name."""
        model_lower = model_name.lower()

        if any(x in model_lower for x in ["gpt", "o1", "o3", "o4"]):
            return "openai", {}
        elif any(x in model_lower for x in ["claude", "sonnet", "haiku", "opus"]):
            return "anthropic", {}
        elif "gemini" in model_lower:
            # Route through Google's OpenAI-compatible endpoint.
            # The native google-genai SDK has Vertex AI routing issues
            # (silently sends to aiplatform.googleapis.com even with api_key).
            return "openai", {
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            }
        elif "llama" in model_lower or "mistral" in model_lower:
            return "openrouter", {}
        else:
            return "openai", {}  # default fallback

    @property
    @override
    def capabilities(self) -> ModelCapabilities:
        """Declare modality support for the RLM backend."""
        return self._capabilities

    @override
    async def _generate(
        self,
        request: ModelRequest,
        *,
        custom_tools: dict[str, object] | None = None,
    ) -> ModelResponse:
        """Call the ContextUnity RLM inference backend and return a complete response."""

        # Merge per-request tools with instance tools
        if custom_tools:
            self._rlm.custom_tools = {**(self._custom_tools or {}), **custom_tools}

        # Build prompt from request parts
        prompt = self._build_prompt(request)

        # RLM completion (synchronous, wrap in executor for async)
        import asyncio

        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(None, lambda: self._rlm.completion(prompt))

        response_text = rlm_response_text(result)
        usage = self._extract_usage(result)

        return ModelResponse(
            text=response_text,
            usage=usage,
            raw_provider=ProviderInfo(
                provider="rlm",
                model_name=self._model_name,
                model_key=f"rlm/{self._model_name}",
            ),
        )

    @override
    def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream token deltas from the ContextUnity RLM inference backend."""

        async def _event_stream() -> AsyncIterator[ModelStreamEvent]:
            # RLM doesn't support streaming natively, emit full response
            """Yield ``TextDelta`` / ``UsageEvent`` from the ContextUnity RLM inference backend stream."""
            response = await self._generate(request)

            # Emit as single chunk
            yield TextDeltaEvent(delta=response.text)
            yield FinalTextEvent(text=response.text)

        return _event_stream()

    def _build_prompt(self, request: ModelRequest) -> str:
        """Build RLM prompt from ``ModelRequest``."""
        parts: list[str] = []

        # Add system prompt if present
        if request.system:
            parts.append(f"SYSTEM: {request.system}\n")

        # Add content from parts
        for part in request.parts:
            if isinstance(part, TextPart):
                parts.append(part.text)

        return "\n".join(parts)

    def _extract_usage(self, result: object) -> UsageStats | None:
        """Extract usage stats from the RLM result."""
        try:
            tokens = rlm_usage_tokens(result)
            if tokens is None:
                return None
            input_tokens, output_tokens = tokens
            return UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        except Exception:
            pass
        return None


class RLMContextHandler:
    """Helper for preparing large contexts for RLM processing.

    Enables efficient handling of 50k+ items by storing them
    as Python variables that the RLM can programmatically access.
    """

    @staticmethod
    def prepare_product_matching_prompt(
        supplier_products: list[dict[str, object]],
        site_products: list[dict[str, object]],
        *,
        matching_instructions: str | None = None,
    ) -> str:
        """Prepare RLM prompt for bulk product matching.

        Args:
            supplier_products: List of supplier product dicts (50k+)
            site_products: List of site product dicts (10k)
            matching_instructions: Custom matching criteria

        Returns:
            RLM-optimized prompt that stores products as variables
        """
        default_instructions = """
TASK: Match supplier products to site products.

Variables available:
- `supplier_products`: List of {len(supplier_products)} supplier items
- `site_products`: List of {len(site_products)} site catalog items

For each supplier product, find the best matching site product based on:
1. SKU/Article code (exact or fuzzy match)
2. Name similarity (after normalization)
3. Brand match
4. Category alignment
5. Technical parameters overlap

APPROACH:
1. First, index site_products by SKU, brand, and key attributes
2. For each supplier product, use code to filter candidates
3. Score candidates and select best match
4. Use sub-LLM calls for semantic comparison when needed

OUTPUT FORMAT:
Return FINAL_VAR(matches) where matches is a JSON list:
[
  {
    "supplier_id": "...",
    "site_id": "...",
    "confidence": 0.95,
    "match_type": "sku_exact|name_fuzzy|semantic"
  }
]
"""

        instructions = matching_instructions or default_instructions

        # The RLM will receive this prompt and have access to the variables
        return f"""
{instructions}

The data is already loaded:
- supplier_products = {json_dumps(supplier_products[:5])}... # ({len(supplier_products)} total items, access full list via variable)
- site_products = {json_dumps(site_products[:5])}... # ({len(site_products)} total items, access full list via variable)

Write code to perform the matching. Use recursive LLM calls for complex comparisons.
"""

    @staticmethod
    def prepare_taxonomy_classification_prompt(
        products: list[dict[str, object]],
        taxonomy_tree: dict[str, object],
        *,
        classification_rules: str | None = None,
    ) -> str:
        """Prepare RLM prompt for bulk taxonomy classification."""
        default_rules = """
TASK: Classify products into taxonomy categories.

Variables:
- `products`: List of products to classify
- `taxonomy`: Full taxonomy tree (1000+ categories)

For each product:
1. Extract key attributes (name, brand, description)
2. Navigate taxonomy tree from root
3. Use code to filter relevant branches
4. Launch sub-LLM call to select final category

OUTPUT: FINAL_VAR(classifications) as JSON list
"""

        return f"""
{classification_rules or default_rules}

Products sample: {json_dumps(products[:3])}... ({len(products)} total)
Taxonomy root: {json_dumps(list(taxonomy_tree.keys())[:10])}... (navigate via variable)
"""


__all__ = ["RLMLLM", "RLMContextHandler"]
