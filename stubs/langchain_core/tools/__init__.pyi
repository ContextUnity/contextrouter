from collections.abc import Callable, Coroutine
from typing import Literal

from pydantic import BaseModel

class BaseTool:
    name: str
    description: str
    tags: list[str]
    handle_tool_error: bool

    def invoke(
        self,
        input: object,
        config: object | None = None,
        **kwargs: object,
    ) -> object: ...
    async def ainvoke(
        self,
        input: object,
        config: object | None = None,
        **kwargs: object,
    ) -> object: ...

class StructuredTool(BaseTool):
    def __init__(
        self,
        *,
        func: Callable[..., object] | None = None,
        coroutine: Callable[..., Coroutine[object, object, object]] | None = None,
        name: str = "",
        description: str = "",
        tags: list[str] | None = None,
        handle_tool_error: bool = False,
        args_schema: type[BaseModel] | dict[str, object] | None = None,
        return_direct: bool = False,
        **kwargs: object,
    ) -> None: ...
    @classmethod
    def from_function(
        cls,
        *,
        func: Callable[..., object] | None = None,
        coroutine: Callable[..., Coroutine[object, object, object]] | None = None,
        name: str | None = None,
        description: str | None = None,
        return_direct: bool = False,
        args_schema: type[BaseModel] | dict[str, object] | None = None,
        infer_schema: bool = True,
        response_format: Literal["content", "content_and_artifact"] = "content",
        parse_docstring: bool = False,
        error_on_invalid_docstring: bool = False,
        **kwargs: object,
    ) -> StructuredTool: ...

class ToolException(Exception): ...
