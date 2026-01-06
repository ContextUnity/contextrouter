# Models (LLMs + Embeddings)

This package defines the **model registry contract** used by the cortex and other modules.

## Model keys

Models are selected by a **registry key** of the form:

- `"<provider>/<name>"`

Examples:

- `vertex/gemini-2.5-flash`
- `litellm/openai/gpt-4o-mini`
- `litellm/anthropic/claude-3-5-haiku-latest`

The key is passed verbatim to `model_registry.create_llm(key)` / `model_registry.get_llm()`.
Keys without `/` are **invalid** (there is no implicit fallback like `vertex/...`).

## Where defaults are defined

Defaults live in core config (`contextrouter.core.config.Config`):

- `models.default_llm` (default: `vertex/gemini-2.5-flash`)
- `models.default_embeddings` (default: `vertex/text-embedding`)
- optional component overrides:
  - `models.intent_llm`
  - `models.suggestions_llm`
  - `models.generation_llm`
  - `models.no_results_llm`

Request controls are provider-agnostic:

- `llm.temperature`
- `llm.max_output_tokens`
- `llm.timeout_sec`
- `llm.max_retries`

## How API keys are provided (LiteLLM / OpenAI-compatible)

Contextrouter **does not** store provider secrets in TOML. Keys are expected via environment variables
used by the underlying SDK (LiteLLM/OpenAI-compatible servers).

Common env vars (depending on the chosen `litellm/*` model key):

- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`

If you use a self-hosted OpenAI-compatible endpoint (proxy), set:

- `CONTEXTROUTER_LITELLM_API_BASE` to your `/v1` base URL

LiteLLM provider settings (optional):

- `CONTEXTROUTER_LITELLM_API_BASE`
- `CONTEXTROUTER_LITELLM_TIMEOUT_SEC`
- `CONTEXTROUTER_LITELLM_FALLBACK_MODELS` (comma-separated list of model keys)


