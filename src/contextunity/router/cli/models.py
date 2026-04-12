"""Models CLI commands for testing and validation."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console

from contextunity.router.cli.registry import register_command

app = typer.Typer(help="Test and validate LLM providers and models.")
console = Console()

PROVIDER_TESTS = {
    "openai": {
        "key": "openai/gpt-4o-mini",
        "tests": ["text", "image", "audio", "stream"],
    },
    "anthropic": {
        "key": "anthropic/claude-3-5-haiku-latest",
        "tests": ["text", "image", "stream"],
    },
    "vertex": {
        "key": "vertex/gemini-2.5-flash",
        "tests": ["text", "image", "audio", "video", "stream"],
    },
    "groq": {
        "key": "groq/llama-3.3-70b-versatile",
        "tests": ["text", "stream"],
    },
    "openrouter": {
        "key": "openrouter/openai/gpt-4o-mini",
        "tests": ["text", "image", "stream"],
    },
}

TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
SAMPLE_AUDIO_URL = "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav"
SAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/200px-PNG_transparency_demonstration_1.png"


@app.command()
def generate(
    model: Annotated[str, typer.Option(..., help="Model key, e.g. vertex/gemini-2.5-flash")],
    prompt: Annotated[
        str, typer.Option(..., help="Text prompt to send")
    ] = "Say hello in one short sentence.",
    stream: Annotated[bool, typer.Option(help="Enable streaming output")] = True,
) -> None:
    """Quick smoke test for an LLM model."""
    from contextunity.router.core.config import Config
    from contextunity.router.modules.models.registry import model_registry
    from contextunity.router.modules.models.types import (
        FinalTextEvent,
        ModelRequest,
        TextDeltaEvent,
        TextPart,
    )

    async def _run() -> None:
        cfg = Config.load()
        try:
            llm = model_registry.create_llm(model, config=cfg)
        except Exception as e:
            console.print(f"[bold red]Failed to create LLM for {model}:[/bold red] {e}")
            raise typer.Exit(1)

        req = ModelRequest(parts=[TextPart(text=prompt)])

        console.print(f"[bold cyan]Provider:[/bold cyan] {llm.__class__.__name__}")
        console.print(f"[bold cyan]Capabilities:[/bold cyan] {llm.capabilities}")

        if not stream:
            console.print("\n[bold green]=== Generating ===[/bold green]")
            resp = await llm.generate(req)
            console.print(resp.text)
        else:
            console.print("\n[bold green]=== Streaming ===[/bold green]")
            async for ev in llm.stream(req):
                if isinstance(ev, TextDeltaEvent):
                    print(ev.delta, end="", flush=True)
                elif isinstance(ev, FinalTextEvent):
                    print(f"\n\n[dim][Final response: {len(ev.text)} chars][/dim]")
                    break

    asyncio.run(_run())


@app.command(name="test-multimodal")
def test_multimodal(
    provider: Annotated[
        str | None, typer.Option(help="Test specific provider (default: all)")
    ] = None,
    modality: Annotated[
        str | None, typer.Option(help="text, image, audio, video, or stream")
    ] = None,
) -> None:
    """Run multimodal capability tests across LLM providers."""

    async def _run() -> None:
        console.print("[bold]Multimodal Testing Suite[/bold]")

        providers = [provider] if provider else list(PROVIDER_TESTS.keys())

        for prov in providers:
            if prov not in PROVIDER_TESTS:
                console.print(f"[bold red]Unknown provider:[/bold red] {prov}")
                continue

            cfg = PROVIDER_TESTS[prov]
            tests = [modality] if modality else cfg["tests"]
            model_key = cfg["key"]

            console.print(f"\n[bold cyan]## Testing {prov} ({model_key})[/bold cyan]")

            from contextunity.router.core.config import Config
            from contextunity.router.modules.models.registry import model_registry
            from contextunity.router.modules.models.types import ImagePart, ModelRequest, TextPart

            config = Config.load()
            try:
                llm = model_registry.create_llm(model_key, config=config)
            except Exception as e:
                console.print(f"  [red]Failed to init {model_key}:[/red] {e}")
                continue

            for t in tests:
                try:
                    if t == "text":
                        req = ModelRequest(parts=[TextPart(text="Say OK.")])
                        resp = await llm.generate(req)
                        console.print(f"  [green]✓ Text:[/green] {resp.text.strip()}")
                    elif t == "image":
                        if not llm.capabilities.supports_image:
                            console.print("  [yellow]⏭️ Image: Not supported[/yellow]")
                            continue
                        req = ModelRequest(
                            parts=[
                                TextPart(text="What color is this image? One word."),
                                ImagePart(mime="image/png", data_b64=TINY_PNG_B64),
                            ]
                        )
                        resp = await llm.generate(req)
                        console.print(f"  [green]✓ Image:[/green] {resp.text.strip()}")
                    elif t == "audio":
                        if not llm.capabilities.supports_audio:
                            console.print("  [yellow]⏭️ Audio: Not supported[/yellow]")
                            continue
                        console.print("  [green]✓ Audio (stub)[/green]")
                    elif t == "video":
                        if not llm.capabilities.supports_video:
                            console.print("  [yellow]⏭️ Video: Not supported[/yellow]")
                            continue
                        console.print("  [green]✓ Video (stub)[/green]")
                    elif t == "stream":
                        req = ModelRequest(parts=[TextPart(text="Count 1 2 3")])
                        console.print("  [green]✓ Stream:[/green] ", end="")
                        async for ev in llm.stream(req):
                            if getattr(ev, "delta", None):
                                print(ev.delta, end="", flush=True)
                        print()
                except Exception as e:
                    console.print(f"  [red]✗ {t.capitalize()} failed:[/red] {e}")

    asyncio.run(_run())


register_command("models", app)
