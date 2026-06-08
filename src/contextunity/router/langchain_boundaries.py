"""LangChain vendor boundaries — shared by cortex, modules, and tools.

Lives at the router package root (not under ``service/``) so tool modules never
import the gRPC ``service`` package barrel and trigger dispatcher import cycles.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import overload

from contextunity.core.narrowing import await_object
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool


def _structured_tool_from_function_boundary(**kwargs: object) -> StructuredTool:
    """Call LangChain's schema-inferring factory behind an explicit vendor guard."""
    factory: object = getattr(StructuredTool, "from_function", None)
    if not callable(factory):
        msg = "StructuredTool.from_function is not callable"
        raise TypeError(msg)
    tool_obj: object = factory(**kwargs)
    if not isinstance(tool_obj, StructuredTool):
        msg = "StructuredTool.from_function returned a non-StructuredTool"
        raise TypeError(msg)
    return tool_obj


async def invoke_runnable_ainvoke(
    runnable: object,
    input: object,
    /,
    *,
    config: RunnableConfig | None = None,
) -> object:
    """Invoke ``ainvoke`` on a LangGraph/LangChain runnable at the vendor boundary."""
    runner: object = getattr(runnable, "ainvoke", None)
    if not callable(runner):
        msg = f"{type(runnable).__name__} has no ainvoke method"
        raise TypeError(msg)
    pending: object = runner(input, config=config) if config is not None else runner(input)
    return await await_object(pending)


def structured_tool_from_function(
    *,
    func: Callable[..., object],
    name: str,
    description: str,
    tags: list[str],
    handle_tool_error: bool,
) -> StructuredTool:
    """Build a LangChain ``StructuredTool`` with inferred ``args_schema``."""
    return _structured_tool_from_function_boundary(
        func=func,
        name=name,
        description=description,
        tags=tags,
        handle_tool_error=handle_tool_error,
    )


def _structured_tool_from_callable(
    func: Callable[..., object],
    *,
    name: str | None = None,
    description: str | None = None,
    return_direct: bool = False,
) -> StructuredTool:
    """Build a ``StructuredTool`` from a sync or async callable."""
    tool_name = name or func.__name__
    description_text = description or ""
    if inspect.iscoroutinefunction(func):
        return _structured_tool_from_function_boundary(
            coroutine=func,
            name=tool_name,
            description=description_text,
            return_direct=return_direct,
            infer_schema=True,
        )
    return _structured_tool_from_function_boundary(
        func=func,
        name=tool_name,
        description=description_text,
        return_direct=return_direct,
        infer_schema=True,
    )


@overload
def langchain_tool(
    func: Callable[..., object],
    /,
) -> StructuredTool: ...


@overload
def langchain_tool(
    name_or_callable: str,
    runnable: object,
    *,
    description: str | None = None,
    return_direct: bool = False,
) -> StructuredTool: ...


@overload
def langchain_tool(
    name_or_callable: str,
    *,
    description: str | None = None,
    return_direct: bool = False,
) -> Callable[[Callable[..., object]], StructuredTool]: ...


def langchain_tool(
    *args: object,
    **kwargs: object,
) -> StructuredTool | Callable[[Callable[..., object]], StructuredTool]:
    """Typed facade over LangChain's ``tool`` decorator factory."""
    description = kwargs.get("description")
    description_str = description if isinstance(description, str) else None
    return_direct = kwargs.get("return_direct", False)
    return_direct_bool = return_direct if isinstance(return_direct, bool) else False

    if len(args) == 1 and callable(args[0]) and not isinstance(args[0], str):
        func = args[0]
        if not callable(func):
            msg = "LangChain tool decorator expected a callable"
            raise TypeError(msg)
        typed_func: Callable[..., object] = func
        return _structured_tool_from_callable(
            typed_func,
            description=description_str,
            return_direct=return_direct_bool,
        )

    preset_name: str | None = None
    if len(args) == 1 and isinstance(args[0], str):
        preset_name = args[0]
    elif len(args) == 2 and isinstance(args[0], str):
        preset_name = args[0]
        runnable = args[1]
        if not callable(runnable):
            msg = "LangChain tool factory expected a callable runnable"
            raise TypeError(msg)
        typed_runnable: Callable[..., object] = runnable
        return _structured_tool_from_callable(
            typed_runnable,
            name=preset_name,
            description=description_str,
            return_direct=return_direct_bool,
        )

    def decorator(func: Callable[..., object]) -> StructuredTool:
        return _structured_tool_from_callable(
            func,
            name=preset_name,
            description=description_str,
            return_direct=return_direct_bool,
        )

    return decorator


def tool_run_method(tool: BaseTool) -> Callable[..., object]:
    """Return the vendor private sync runner for wrapped tool delegation."""
    method: object = getattr(tool, "_run", None)
    if not callable(method):
        msg = f"{type(tool).__name__} has no _run method"
        raise TypeError(msg)
    return method


def tool_arun_method(tool: BaseTool) -> Callable[..., object]:
    """Return the vendor private async runner for wrapped tool delegation."""
    method: object = getattr(tool, "_arun", None)
    if not callable(method):
        msg = f"{type(tool).__name__} has no _arun method"
        raise TypeError(msg)
    return method


def invoke_tool_run(
    tool: BaseTool,
    /,
    *args: object,
    **kwargs: object,
) -> object:
    """Run a LangChain tool synchronously at the vendor boundary."""
    runner: object = getattr(tool, "invoke", None)
    if not callable(runner):
        msg = f"{type(tool).__name__} has no invoke method"
        raise TypeError(msg)
    return runner(*args, **kwargs)


async def invoke_tool_arun(
    tool: BaseTool,
    /,
    *args: object,
    **kwargs: object,
) -> object:
    """Run a LangChain tool asynchronously at the vendor boundary."""
    runner: object = getattr(tool, "ainvoke", None)
    if not callable(runner):
        msg = f"{type(tool).__name__} has no ainvoke method"
        raise TypeError(msg)
    pending: object = runner(*args, **kwargs)
    return await await_object(pending)


tool = langchain_tool

__all__ = [
    "invoke_runnable_ainvoke",
    "invoke_tool_arun",
    "invoke_tool_run",
    "langchain_tool",
    "structured_tool_from_function",
    "tool",
    "tool_arun_method",
    "tool_run_method",
]
