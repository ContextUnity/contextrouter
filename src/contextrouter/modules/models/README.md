# Models (Multimodal LLMs + Embeddings)

This package defines the **multimodal model registry contract** used by the cortex and other modules. The interface supports text, image, and audio inputs with **strict capability-based fallback**.

## Multimodal Interface

Models use a **unified multimodal contract** that accepts text, image, and audio parts:

```python
from contextrouter.modules.models.types import ModelRequest, TextPart, ImagePart

# Text-only request
request = ModelRequest(
    parts=[TextPart(text="Hello, world!")],
    system="You are a helpful assistant",
    temperature=0.7,
)

# Multimodal request (text + image)
request = ModelRequest(
    parts=[
        TextPart(text="What's in this image?"),
        ImagePart(mime="image/jpeg", data_b64="...", uri="https://example.com/image.jpg")
    ]
)
```

## Model Registry & Fallback

### Model Keys

Models are selected by a **registry key** of the form: `"<provider>/<name>"`

Examples:
- `vertex/gemini-2.5-flash` (multimodal, Google)
- `openai/gpt-5.1` (multimodal, OpenAI)
- `anthropic/claude-opus-4.5` (text-only, Anthropic)
- `openrouter/openai/gpt-5.1` (OpenRouter, OpenAI-compatible)
- `local/llama3.1` (Ollama, OpenAI-compatible)
- `local-vllm/meta-llama/Llama-3.1-8B-Instruct` (vLLM, OpenAI-compatible)

### Fallback System

Models support **strict capability-based fallback**:

```python
from contextrouter.modules.models.registry import model_registry

# Get model with fallback
model = model_registry.get_llm_with_fallback(
    key="vertex/gemini-2.5-flash",
    fallback_keys=["openai/gpt-5.1", "anthropic/claude-sonnet-4.5"],
    strategy="fallback",  # sequential
)
```

**Fallback Rules:**
- Only models supporting **all required modalities** are considered
- No automatic conversion (e.g., image → text description)

### Fallback Strategies

- **`fallback` (sequential)**: try candidates in order until one succeeds.
- **`parallel`**: run all candidates concurrently and return the first success (**generate only**).
- **`cost-priority`**: same mechanics as `fallback`; you must order your `fallback` list cheapest → most expensive.

**Streaming rule:** streaming always behaves like **sequential fallback** — we never switch providers mid-stream.

## Providers

### Built-in Providers

| Provider | Key Pattern | Modalities | Notes |
|----------|-------------|------------|-------|
| **Vertex AI** | `vertex/*` | Text + Image + Audio + Video | Default, requires GCP credentials. Gemini 1.5/2.5 models are multimodal. |
| **OpenAI** | `openai/*` | Text + Image + Audio (ASR) | Requires `contextrouter[models-openai]` + `OPENAI_API_KEY`. ASR via Whisper. |
| **Anthropic** | `anthropic/*` | Text + Image | Requires `contextrouter[models-anthropic]` + `ANTHROPIC_API_KEY`. Claude supports images natively. |
| **OpenRouter** | `openrouter/*` | Text + Image | Requires `contextrouter[models-openai]` + `OPENROUTER_API_KEY`. Model-dependent capabilities. |
| **Groq** | `groq/*` | Text + Image + Audio (ASR) | Requires `contextrouter[models-openai]` + `GROQ_API_KEY`. Ultra-fast Whisper ASR. |
| **RunPod** | `runpod/*` | Text + Image | Requires `contextrouter[models-openai]`. OpenAI-compatible chat; custom workers can do more. |
| **HF Hub (Remote)** | `hf-hub/*` | Text + Image + Audio (task-dependent) | Requires `contextrouter[models-hf-hub]`. Depends on task: ASR, VQA, image-to-text. |
| **Local (vLLM/Ollama)** | `local/*`, `local-vllm/*` | Text + Image | Requires `contextrouter[models-openai]`. Vision models support images. |
| **HuggingFace Transformers** | `hf/*` | Task-dependent (Text/Image/Audio) | Local inference. Task controls modality: text-gen, ASR, image-classification. |
| **LiteLLM** | `litellm/*` | - | **Stub only** (not implemented). |

### HuggingFace Transformers ⚠️

**WARNING:** HuggingFace local inference requires heavy dependencies:

```bash
uv add contextrouter[hf-transformers]
```

**Limitations:**
- Requires `torch` + `transformers` (large installation)
- Heavy models can be very slow / memory-hungry depending on your hardware

**Use cases:**
- CPU-based development/testing
- Small / medium models for local dev (e.g., `hf/distilgpt2`, `hf/TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- Offline environments

**Do NOT use for:**
- Production inference
- Large models
- GPU workloads (use vLLM/TGI instead)

### LiteLLM (stub)

`litellm/*` exists as a **stub provider** (raises `NotImplementedError`).

Reasons:
- We prefer explicit providers (Vertex/OpenAI/Anthropic/OpenRouter/local) for clearer debugging and control.
- LiteLLM would add another abstraction layer that can complicate streaming/multimodal/error mapping.
- Observability/cost tracking is handled via Langfuse + our own normalized `UsageStats`.

### Running Local Models (vLLM / Ollama)

**vLLM** (OpenAI-compatible server):

You can run vLLM via Docker (recommended) or directly (`uv add vllm` + GPU drivers).

Example direct run:
```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --port 8000
```

- Set `LOCAL_VLLM_BASE_URL=http://localhost:8000/v1`
- Use `local-vllm/meta-llama/Llama-3.1-8B-Instruct` (any model ID supported by vLLM)

**Ollama** (OpenAI-compatible):

```bash
ollama serve
ollama pull llama3.1
```

- Set `LOCAL_OLLAMA_BASE_URL=http://localhost:11434/v1`
- Use `local/llama3.1`

+## Performance Considerations

### Local vs Remote Models

-   **Remote API models** (Vertex, OpenAI, Anthropic): High reliability, low setup effort, fast inference, multimodal support, and high accuracy for complex tasks (like JSON formatting).
-   **Aggregator API models** (OpenRouter): Access to hundreds of models via a single API; reliability depends on the specific provider.
-   **Local Model Servers** (vLLM, Ollama): High privacy, no per-token costs, but requires managing your own hardware (GPU/RAM). Good for text generation and random creative tasks.
-   **Local Libraries** (HuggingFace Transformers): Best for specialized small models (STT, classification, embeddings) running directly in your application process without an external server.

## Best Practices & Recommendations
+
+### Structured Output (JSON)
+
+For tasks that require high-quality structured output (e.g., `intent`, `suggestions`, ingestion stages):
+
+*   **Recommended**: `vertex/gemini-2.5-flash-lite` or better.
+*   **Warning**: Local models (vLLM/Ollama) often have difficulty maintaining strict JSON formatting for complex schemas. Using local models for these tasks may lead to parsing errors.
+
 ## Configuration

```toml
[models]
default = "vertex/gemini-2.5-flash"

[models.rag.intent]
model = "vertex/gemini-2.5-flash-lite"
fallback = ["anthropic/claude-haiku-4.5"]
strategy = "fallback"

[models.rag.generation]
model = "vertex/gemini-2.5-flash"
fallback = ["openai/gpt-5.1", "anthropic/claude-sonnet-4.5"]
strategy = "fallback"

[models.rag.no_results]
model = "vertex/gemini-2.5-flash-lite"
fallback = ["anthropic/claude-haiku-4.5"]
strategy = "fallback"
```

### Environment Variables

API keys are **never stored in config**, only via environment:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- `GROQ_API_KEY`
- `RUNPOD_API_KEY`
- `RUNPOD_BASE_URL`
- `HF_TOKEN`
- `OPENROUTER_BASE_URL` (optional)
- `LOCAL_OLLAMA_BASE_URL`, `LOCAL_VLLM_BASE_URL`

## Development & Testing

### Adding New Providers

1. Implement `BaseModel` subclass with proper `capabilities`
2. Register with `@model_registry.register_llm("provider", "name")`
3. Add optional dependencies to `pyproject.toml`
4. Update this README

### Testing Multimodal Features

Use the test utilities in `tests/unit/` for:
- Capability filtering validation
- Fallback strategy testing
- Stream safety verification
