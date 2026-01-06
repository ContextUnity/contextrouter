# Models (LLMs + Embeddings)

This package defines the **model registry contract** used by the cortex and other modules.

## Model keys

Models are selected by a **registry key** of the form:

- `"<provider>/<name>"`

Examples:

- `vertex/gemini-2.5-flash`

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

## How API keys are provided

ContextRouter **does not** store provider secrets in TOML. Keys are expected via environment variables used by the underlying SDK.

Common env vars:

- **Vertex AI**: configured via `vertex` config section
- **OpenAI**: `OPENAI_API_KEY`
