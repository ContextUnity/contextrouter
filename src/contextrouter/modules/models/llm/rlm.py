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

import logging
from typing import AsyncIterator

from contextrouter.core import Config
from contextrouter.core.tokens import ContextToken

from ..base import BaseModel
from ..registry import model_registry
from ..types import (
    FinalTextEvent,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ProviderInfo,
    TextDeltaEvent,
    UsageStats,
)

logger = logging.getLogger(__name__)


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
        config: Config,
        *,
        model_name: str | None = None,
        environment: str = "local",
        environment_kwargs: dict | None = None,
        verbose: bool = False,
        log_dir: str | None = None,
        custom_tools: dict | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize RLM wrapper.

        Args:
            config: ContextRouter configuration
            model_name: Base model to use (e.g., "gpt-5-mini", "claude-sonnet")
            environment: REPL environment type:
                - "local": Uses Python exec (default, same process)
                - "docker": Isolated Docker container
                - "modal": Modal.com sandboxes (cloud)
                - "prime": Prime Intellect sandboxes (cloud)
            environment_kwargs: Additional environment configuration
            verbose: Enable rich console output
            log_dir: Directory for trajectory logs (for visualization)
        """
        try:
            from rlm import RLM
        except ImportError as e:
            raise ImportError(
                "RLMLLM requires the 'rlm' package. Install with: pip install rlm or uv add rlm"
            ) from e

        self._cfg = config
        self._model_name = (model_name or "gpt-5-mini").strip()
        self._environment = environment
        self._verbose = verbose

        # Determine backend from model name
        backend, extra_kwargs = self._infer_backend(self._model_name)

        # Build RLM instance
        rlm_kwargs: dict = {
            "backend": backend,
            "backend_kwargs": {"model_name": self._model_name, **extra_kwargs},
            "environment": environment,
            "verbose": verbose,
            "max_timeout": 300.0,  # 5 min per brand — prevents indefinite runs
            "max_iterations": 15,  # cap REPL iterations
        }

        # Forward api_key to backend if provided (bypasses module-level env cache)
        if "api_key" in kwargs:
            rlm_kwargs["backend_kwargs"]["api_key"] = kwargs.pop("api_key")

        if environment_kwargs:
            rlm_kwargs["environment_kwargs"] = environment_kwargs

        if custom_tools:
            rlm_kwargs["custom_tools"] = custom_tools

        # Add logger if log_dir specified
        if log_dir:
            from rlm.logger import RLMLogger

            rlm_kwargs["logger"] = RLMLogger(log_dir=log_dir)

        self._rlm = RLM(**rlm_kwargs)
        self._custom_tools = custom_tools

        self._capabilities = ModelCapabilities(
            supports_text=True,
            supports_image=False,  # RLM is text-focused
            supports_audio=False,
        )

    @staticmethod
    def _infer_backend(model_name: str) -> tuple[str, dict]:
        """Infer RLM backend and extra kwargs from model name.

        Returns:
            Tuple of (backend_name, extra_backend_kwargs).
        """
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
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(
        self,
        request: ModelRequest,
        *,
        token: ContextToken | None = None,
        custom_tools: dict | None = None,
    ) -> ModelResponse:
        """Generate response using RLM recursive completion.

        The RLM will create a REPL environment and may spawn recursive
        LLM calls to process the request, especially for large contexts.

        Args:
            request: Model request with prompt and parameters.
            token: Optional context token.
            custom_tools: Per-request REPL variables/functions.
                Non-callable values become REPL variables accessible immediately.
        """
        _ = token

        # Merge per-request tools with instance tools
        if custom_tools:
            self._rlm.custom_tools = {**(self._custom_tools or {}), **custom_tools}

        # Build prompt from request parts
        prompt = self._build_prompt(request)

        # RLM completion (synchronous, wrap in executor for async)
        import asyncio

        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(None, lambda: self._rlm.completion(prompt))

        # Extract response text
        response_text = getattr(result, "response", str(result))

        # Build usage stats from RLM metrics if available
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

    async def stream(
        self,
        request: ModelRequest,
        *,
        token: ContextToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream is not natively supported by RLM - falls back to generate."""
        _ = token

        # RLM doesn't support streaming natively, emit full response
        response = await self.generate(request, token=token)

        # Emit as single chunk
        yield TextDeltaEvent(delta=response.text)
        yield FinalTextEvent(text=response.text)

    def _build_prompt(self, request: ModelRequest) -> str:
        """Build RLM prompt from ModelRequest."""
        parts = []

        # Add system prompt if present
        if request.system:
            parts.append(f"SYSTEM: {request.system}\n")

        # Add content from parts
        for part in request.parts:
            if hasattr(part, "text"):
                parts.append(part.text)

        return "\n".join(parts)

    def _extract_usage(self, result: object) -> UsageStats | None:
        """Extract usage stats from RLM result (RLMChatCompletion)."""
        try:
            # RLMChatCompletion has usage_summary: UsageSummary
            usage_summary = getattr(result, "usage_summary", None)
            if usage_summary:
                input_tokens = getattr(usage_summary, "total_input_tokens", 0) or 0
                output_tokens = getattr(usage_summary, "total_output_tokens", 0) or 0
                return UsageStats(
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    total_tokens=int(input_tokens + output_tokens),
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
        supplier_products: list[dict],
        site_products: list[dict],
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
        import json

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
- supplier_products = {json.dumps(supplier_products[:5])}... # ({len(supplier_products)} total items, access full list via variable)
- site_products = {json.dumps(site_products[:5])}... # ({len(site_products)} total items, access full list via variable)

Write code to perform the matching. Use recursive LLM calls for complex comparisons.
"""

    @staticmethod
    def prepare_taxonomy_classification_prompt(
        products: list[dict],
        taxonomy_tree: dict,
        *,
        classification_rules: str | None = None,
    ) -> str:
        """Prepare RLM prompt for bulk taxonomy classification.

        Taxonomy tree can be 1000+ categories - RLM navigates it
        programmatically instead of having it all in context.
        """
        import json

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

Products sample: {json.dumps(products[:3])}... ({len(products)} total)
Taxonomy root: {json.dumps(list(taxonomy_tree.keys())[:10])}... (navigate via variable)
"""


__all__ = ["RLMLLM", "RLMContextHandler"]
