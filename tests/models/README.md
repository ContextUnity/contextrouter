# Model Tests

This directory contains tests for the multimodal model registry and providers.

## Test Structure

### Unit Tests (Current)

- `test_providers.py` - Provider initialization and mock generation tests
- `test_extra_providers.py` - Tests for Groq, RunPod, HF Hub providers
- `test_base_model.py` - Base model contract tests (needs updating)
- `test_model_registry.py` - Registry and fallback tests (needs updating)

### Running Unit Tests

```bash
# All model tests
pytest tests/models/ -v

# Specific provider tests
pytest tests/models/test_providers.py -v
pytest tests/models/test_extra_providers.py -v
```

## Integration Tests (Recommended)

For proper multimodal testing, use the example script:

```bash
# List available providers
python examples/test_multimodal.py --list

# Test specific provider
python examples/test_multimodal.py --provider openai
python examples/test_multimodal.py --provider vertex --modality image

# Test audio with local file
python examples/test_multimodal.py --provider groq --audio-file /path/to/audio.wav

# Test video (Vertex only)
python examples/test_multimodal.py --provider vertex --video-uri gs://bucket/video.mp4
```

## Modality Support Matrix

| Provider | Text | Image | Audio | Video |
|----------|------|-------|-------|-------|
| Vertex AI | ✅ | ✅ | ✅ | ✅ |
| OpenAI | ✅ | ✅ | ✅ (ASR) | ❌ |
| Anthropic | ✅ | ✅ | ❌ | ❌ |
| Groq | ✅ | ✅ | ✅ (ASR) | ❌ |
| OpenRouter | ✅ | ✅ | ❌ | ❌ |
| RunPod | ✅ | ✅ | ❌ | ❌ |
| HF Hub | ✅ | ✅ (task) | ✅ (task) | ❌ |
| Local | ✅ | ✅ | ❌ | ❌ |
| HF Transformers | ✅ | ✅ (task) | ✅ (task) | ❌ |

## What to Test

### Unit Tests (Mock-based)

1. **Provider initialization** - Config handling, capability declaration
2. **Message building** - Multimodal content formatting
3. **Error handling** - Missing parts, invalid config
4. **Fallback behavior** - Capability filtering, strategy selection

### Integration Tests (Real API)

1. **Text generation** - Basic prompt → response
2. **Image input** - Vision model with base64 and URL images
3. **Audio input** - ASR/transcription
4. **Streaming** - Token-by-token generation
5. **Error recovery** - Rate limits, timeouts

### Pipeline Tests (RAG)

1. **Intent detection** - Model returns valid JSON
2. **Generation** - Full RAG pipeline with citations
3. **Fallback** - Primary model fails → backup succeeds

## CI/CD Recommendations

```yaml
# Example GitHub Actions workflow
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Unit tests (no API keys)
        run: pytest tests/models/ -v

  integration-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'  # Nightly only
    steps:
      - uses: actions/checkout@v4
      - name: Integration tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: python examples/test_multimodal.py --provider openai
```

## Notes

- Video support only works with **Vertex AI (Gemini)** and requires GCS URIs
- All tests use `pytest.mark.anyio` for async tests
- Mock providers should inherit from `MockProvider` for consistency
