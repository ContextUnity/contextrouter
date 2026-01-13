#!/usr/bin/env python3
"""Test multimodal capabilities across all providers.

This script helps verify that each provider correctly handles
text, image, audio, and video inputs according to their declared capabilities.

Usage:
    # Set required environment variables first
    export OPENAI_API_KEY="..."
    export ANTHROPIC_API_KEY="..."
    # etc.

    # Run all tests
    python examples/test_multimodal.py

    # Run specific provider tests
    python examples/test_multimodal.py --provider openai
    python examples/test_multimodal.py --provider vertex --modality image

Requirements:
    pip install contextrouter[models-openai,models-anthropic,models-hf-hub]
    pip install httpx  # For ASR
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
from pathlib import Path

# Sample test data (small inline images/audio for quick tests)
# 1x1 red PNG (minimal valid PNG for testing)
TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

# Sample audio URL (public domain)
SAMPLE_AUDIO_URL = "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav"

# Sample image URL
SAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/200px-PNG_transparency_demonstration_1.png"


async def test_text_generation(provider: str, model_key: str) -> bool:
    """Test basic text generation."""
    from contextrouter.core.config import Config
    from contextrouter.modules.models.registry import model_registry
    from contextrouter.modules.models.types import ModelRequest, TextPart

    print(f"\n{'=' * 60}")
    print(f"[TEXT] Testing {provider}: {model_key}")
    print("=" * 60)

    try:
        config = Config.load()
        llm = model_registry.create_llm(model_key, config=config)

        request = ModelRequest(
            parts=[TextPart(text="Say 'Hello World' and nothing else.")],
            temperature=0.1,
            max_output_tokens=50,
        )

        print(f"Capabilities: {llm.capabilities}")
        print("Generating...")

        response = await llm.generate(request)
        print(f"Response: {response.text[:200]}...")
        print(f"Provider: {response.raw_provider}")
        print("✅ TEXT TEST PASSED")
        return True

    except Exception as e:
        print(f"❌ TEXT TEST FAILED: {e}")
        return False


async def test_image_input(provider: str, model_key: str) -> bool:
    """Test image input (vision) capability."""
    from contextrouter.core.config import Config
    from contextrouter.modules.models.registry import model_registry
    from contextrouter.modules.models.types import ImagePart, ModelRequest, TextPart

    print(f"\n{'=' * 60}")
    print(f"[IMAGE] Testing {provider}: {model_key}")
    print("=" * 60)

    try:
        config = Config.load()
        llm = model_registry.create_llm(model_key, config=config)

        if not llm.capabilities.supports_image:
            print(f"⏭️  Skipping: {provider} does not support images")
            return True

        # Test with base64 image
        request = ModelRequest(
            parts=[
                TextPart(text="What color is this image? Answer in one word."),
                ImagePart(mime="image/png", data_b64=TINY_PNG_B64),
            ],
            temperature=0.1,
            max_output_tokens=50,
        )

        print(f"Capabilities: {llm.capabilities}")
        print("Generating with base64 image...")

        response = await llm.generate(request)
        print(f"Response: {response.text[:200]}...")
        print("✅ IMAGE TEST PASSED")
        return True

    except Exception as e:
        print(f"❌ IMAGE TEST FAILED: {e}")
        return False


async def test_image_url(provider: str, model_key: str) -> bool:
    """Test image URL input."""
    from contextrouter.core.config import Config
    from contextrouter.modules.models.registry import model_registry
    from contextrouter.modules.models.types import ImagePart, ModelRequest, TextPart

    print(f"\n{'=' * 60}")
    print(f"[IMAGE URL] Testing {provider}: {model_key}")
    print("=" * 60)

    try:
        config = Config.load()
        llm = model_registry.create_llm(model_key, config=config)

        if not llm.capabilities.supports_image:
            print(f"⏭️  Skipping: {provider} does not support images")
            return True

        request = ModelRequest(
            parts=[
                TextPart(text="Describe what you see in this image in 2-3 words."),
                ImagePart(mime="image/png", uri=SAMPLE_IMAGE_URL),
            ],
            temperature=0.1,
            max_output_tokens=100,
        )

        print("Generating with image URL...")
        response = await llm.generate(request)
        print(f"Response: {response.text[:200]}...")
        print("✅ IMAGE URL TEST PASSED")
        return True

    except Exception as e:
        print(f"❌ IMAGE URL TEST FAILED: {e}")
        return False


async def test_audio_asr(provider: str, model_key: str) -> bool:
    """Test audio input (ASR/transcription) capability."""
    from contextrouter.core.config import Config
    from contextrouter.modules.models.registry import model_registry
    from contextrouter.modules.models.types import AudioPart, ModelRequest

    print(f"\n{'=' * 60}")
    print(f"[AUDIO ASR] Testing {provider}: {model_key}")
    print("=" * 60)

    try:
        config = Config.load()
        llm = model_registry.create_llm(model_key, config=config)

        if not llm.capabilities.supports_audio:
            print(f"⏭️  Skipping: {provider} does not support audio")
            return True

        # Download sample audio
        print(f"Downloading sample audio from {SAMPLE_AUDIO_URL}...")
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.get(SAMPLE_AUDIO_URL)
            audio_data = base64.b64encode(resp.content).decode()

        request = ModelRequest(
            parts=[AudioPart(mime="audio/wav", data_b64=audio_data)],
            timeout_sec=120.0,
        )

        print("Transcribing...")
        response = await llm.generate(request)
        print(f"Transcript: {response.text[:500]}...")
        print("✅ AUDIO ASR TEST PASSED")
        return True

    except Exception as e:
        print(f"❌ AUDIO ASR TEST FAILED: {e}")
        return False


async def test_audio_from_file(provider: str, model_key: str, audio_path: str) -> bool:
    """Test audio input from local file."""
    from contextrouter.core.config import Config
    from contextrouter.modules.models.registry import model_registry
    from contextrouter.modules.models.types import AudioPart, ModelRequest

    print(f"\n{'=' * 60}")
    print(f"[AUDIO FILE] Testing {provider}: {model_key}")
    print("=" * 60)

    try:
        config = Config.load()
        llm = model_registry.create_llm(model_key, config=config)

        if not llm.capabilities.supports_audio:
            print(f"⏭️  Skipping: {provider} does not support audio")
            return True

        path = Path(audio_path)
        if not path.exists():
            print(f"⏭️  Skipping: Audio file not found: {audio_path}")
            return True

        # Determine mime type from extension
        mime_map = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}
        mime = mime_map.get(path.suffix.lower(), "audio/wav")

        request = ModelRequest(
            parts=[AudioPart(mime=mime, uri=str(path.absolute()))],
            timeout_sec=120.0,
        )

        print(f"Transcribing {audio_path}...")
        response = await llm.generate(request)
        print(f"Transcript: {response.text[:500]}...")
        print("✅ AUDIO FILE TEST PASSED")
        return True

    except Exception as e:
        print(f"❌ AUDIO FILE TEST FAILED: {e}")
        return False


async def test_video_input(provider: str, model_key: str, video_uri: str) -> bool:
    """Test video input capability (Vertex only)."""
    from contextrouter.core.config import Config
    from contextrouter.modules.models.registry import model_registry
    from contextrouter.modules.models.types import ModelRequest, TextPart, VideoPart

    print(f"\n{'=' * 60}")
    print(f"[VIDEO] Testing {provider}: {model_key}")
    print("=" * 60)

    try:
        config = Config.load()
        llm = model_registry.create_llm(model_key, config=config)

        if not llm.capabilities.supports_video:
            print(f"⏭️  Skipping: {provider} does not support video")
            return True

        request = ModelRequest(
            parts=[
                TextPart(text="Describe what happens in this video in 2-3 sentences."),
                VideoPart(mime="video/mp4", uri=video_uri),
            ],
            timeout_sec=180.0,
            max_output_tokens=200,
        )

        print(f"Processing video: {video_uri}...")
        response = await llm.generate(request)
        print(f"Response: {response.text[:500]}...")
        print("✅ VIDEO TEST PASSED")
        return True

    except Exception as e:
        print(f"❌ VIDEO TEST FAILED: {e}")
        return False


async def test_streaming(provider: str, model_key: str) -> bool:
    """Test streaming capability."""
    from contextrouter.core.config import Config
    from contextrouter.modules.models.registry import model_registry
    from contextrouter.modules.models.types import ModelRequest, TextPart

    print(f"\n{'=' * 60}")
    print(f"[STREAM] Testing {provider}: {model_key}")
    print("=" * 60)

    try:
        config = Config.load()
        llm = model_registry.create_llm(model_key, config=config)

        request = ModelRequest(
            parts=[TextPart(text="Count from 1 to 5, one number per line.")],
            temperature=0.1,
            max_output_tokens=100,
        )

        print("Streaming...")
        chunks = []
        async for event in llm.stream(request):
            if event.event_type == "text_delta":
                chunks.append(event.delta)
                print(event.delta, end="", flush=True)
            elif event.event_type == "final_text":
                print(f"\n[Final: {len(event.text)} chars]")

        print(f"\nReceived {len(chunks)} chunks")
        print("✅ STREAM TEST PASSED")
        return True

    except Exception as e:
        print(f"❌ STREAM TEST FAILED: {e}")
        return False


# Provider configurations for testing
PROVIDER_CONFIGS = {
    "openai": {
        "key": "openai/gpt-4o-mini",
        "tests": ["text", "image", "audio", "stream"],
    },
    "anthropic": {
        "key": "anthropic/claude-3-5-haiku-20241022",
        "tests": ["text", "image", "stream"],
    },
    "vertex": {
        "key": "vertex/gemini-2.5-flash",
        "tests": ["text", "image", "audio", "video", "stream"],
    },
    "groq": {
        "key": "groq/llama-3.3-70b-versatile",
        "tests": ["text", "image", "audio", "stream"],
    },
    "openrouter": {
        "key": "openrouter/openai/gpt-4o-mini",
        "tests": ["text", "image", "stream"],
    },
    "hf-hub": {
        "key": "hf-hub/mistralai/Mistral-7B-Instruct-v0.2",
        "tests": ["text", "stream"],
    },
    "local": {
        "key": "local/llama3.1",
        "tests": ["text", "stream"],
    },
    "local-vllm": {
        "key": "local-vllm/meta-llama/Llama-3.1-8B-Instruct",
        "tests": ["text", "image", "stream"],
    },
}


async def run_provider_tests(
    provider: str,
    modality: str | None = None,
    audio_file: str | None = None,
    video_uri: str | None = None,
) -> dict[str, bool]:
    """Run tests for a specific provider."""
    if provider not in PROVIDER_CONFIGS:
        print(f"Unknown provider: {provider}")
        print(f"Available: {list(PROVIDER_CONFIGS.keys())}")
        return {}

    cfg = PROVIDER_CONFIGS[provider]
    model_key = cfg["key"]
    tests = cfg["tests"]

    results: dict[str, bool] = {}

    if modality:
        tests = [modality] if modality in tests else []

    for test_name in tests:
        if test_name == "text":
            results["text"] = await test_text_generation(provider, model_key)
        elif test_name == "image":
            results["image_b64"] = await test_image_input(provider, model_key)
            results["image_url"] = await test_image_url(provider, model_key)
        elif test_name == "audio":
            results["audio_asr"] = await test_audio_asr(provider, model_key)
            if audio_file:
                results["audio_file"] = await test_audio_from_file(provider, model_key, audio_file)
        elif test_name == "video":
            if video_uri:
                results["video"] = await test_video_input(provider, model_key, video_uri)
            else:
                print("⏭️  Skipping video test: no --video-uri provided")
        elif test_name == "stream":
            results["stream"] = await test_streaming(provider, model_key)

    return results


async def main() -> int:
    parser = argparse.ArgumentParser(description="Test multimodal capabilities")
    parser.add_argument(
        "--provider",
        choices=list(PROVIDER_CONFIGS.keys()),
        help="Test specific provider (default: all)",
    )
    parser.add_argument(
        "--modality",
        choices=["text", "image", "audio", "video", "stream"],
        help="Test specific modality",
    )
    parser.add_argument(
        "--audio-file",
        help="Path to local audio file for ASR testing",
    )
    parser.add_argument(
        "--video-uri",
        help="GCS URI or URL to video file for video testing (e.g., gs://bucket/video.mp4)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available providers and their capabilities",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable providers and their test capabilities:\n")
        for name, cfg in PROVIDER_CONFIGS.items():
            print(f"  {name}:")
            print(f"    Model: {cfg['key']}")
            print(f"    Tests: {', '.join(cfg['tests'])}")
            print()
        return 0

    all_results: dict[str, dict[str, bool]] = {}

    if args.provider:
        providers = [args.provider]
    else:
        providers = list(PROVIDER_CONFIGS.keys())

    for provider in providers:
        print(f"\n{'#' * 70}")
        print(f"# Testing provider: {provider}")
        print("#" * 70)

        results = await run_provider_tests(
            provider,
            modality=args.modality,
            audio_file=args.audio_file,
            video_uri=args.video_uri,
        )
        all_results[provider] = results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_passed = 0
    total_failed = 0

    for provider, results in all_results.items():
        if not results:
            continue
        passed = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)
        total_passed += passed
        total_failed += failed

        status = "✅" if failed == 0 else "❌"
        print(f"{status} {provider}: {passed}/{passed + failed} tests passed")
        for test_name, success in results.items():
            mark = "✅" if success else "❌"
            print(f"    {mark} {test_name}")

    print(f"\nTotal: {total_passed} passed, {total_failed} failed")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
