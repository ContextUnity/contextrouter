"""Types and built-in mappings for the registry."""

from __future__ import annotations

from typing import Literal

# Model selection strategies:
# - fallback: sequentially try candidates in order until one succeeds.
# - parallel: run candidates concurrently and return the first success (generate-only).
# - cost-priority: sequential fallback where you order candidates cheapest → most expensive.
ModelSelectionStrategy = Literal["fallback", "parallel", "cost-priority"]

# Built-in model mappings
BUILTIN_LLMS: dict[str, str] = {
    "vertex/*": "contextrouter.modules.models.llm.vertex.VertexLLM",
    "openai/*": "contextrouter.modules.models.llm.openai.OpenAILLM",
    "openrouter/*": "contextrouter.modules.models.llm.openrouter.OpenRouterLLM",
    "local/*": "contextrouter.modules.models.llm.local_openai.LocalOllamaLLM",
    "local-vllm/*": "contextrouter.modules.models.llm.local_openai.LocalVllmLLM",
    # Anthropic is provider-wildcarded like OpenAI/OpenRouter: any model name becomes `model_name`.
    "anthropic/*": "contextrouter.modules.models.llm.anthropic.AnthropicLLM",
    "groq/*": "contextrouter.modules.models.llm.groq.GroqLLM",
    # Inception Labs: Mercury-2 diffusion LLM (OpenAI-compatible)
    "inception/*": "contextrouter.modules.models.llm.inception.InceptionLLM",
    "runpod/*": "contextrouter.modules.models.llm.runpod.RunPodLLM",
    "hf-hub/*": "contextrouter.modules.models.llm.hf_hub.HuggingFaceHubLLM",
    # HuggingFace transformers: allow `hf/<model_id>`.
    "hf/*": "contextrouter.modules.models.llm.huggingface.HuggingFaceLLM",
    # Perplexity Sonar: built-in search LLM
    "perplexity/*": "contextrouter.modules.models.llm.perplexity.PerplexityLLM",
    # Recursive Language Models: wraps any LLM with REPL-based recursive context processing
    # for handling massive contexts (50k+ items). Uses `rlm/<base_model>` format.
    # Example: "rlm/gpt-5-mini" for GPT-5-mini with recursive capabilities.
    "rlm/*": "contextrouter.modules.models.llm.rlm.RLMLLM",
}

BUILTIN_EMBEDDINGS: dict[str, str] = {
    "vertex/text-embedding": ("contextrouter.modules.models.embeddings.vertex.VertexEmbeddings"),
    "hf/sentence-transformers": (
        "contextrouter.modules.models.embeddings.huggingface.HuggingFaceEmbeddings"
    ),
}
