"""Tests for fire-and-forget background task retention."""

from __future__ import annotations

import asyncio

import pytest

from contextunity.router.modules.retrieval.rag import background_tasks

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_background_tasks() -> None:
    background_tasks._background_tasks.clear()
    yield
    background_tasks._background_tasks.clear()


@pytest.mark.asyncio
async def test_spawn_background_task_retains_reference_until_done() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow() -> str:
        started.set()
        await release.wait()
        return "ok"

    task = background_tasks.spawn_background_task(_slow())
    assert task in background_tasks._background_tasks

    await started.wait()
    assert task in background_tasks._background_tasks

    release.set()
    await task
    await asyncio.sleep(0)

    assert task not in background_tasks._background_tasks


@pytest.mark.asyncio
async def test_spawn_background_task_logs_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []

    def _capture(msg: str, *args: object, **kwargs: object) -> None:
        errors.append(msg % args if args else msg)

    monkeypatch.setattr(background_tasks.logger, "error", _capture)

    async def _fail() -> None:
        raise ValueError("dual-read boom")

    task = background_tasks.spawn_background_task(_fail())
    with pytest.raises(ValueError, match="dual-read boom"):
        await task

    await asyncio.sleep(0)
    assert any("Background task failed" in entry for entry in errors)
