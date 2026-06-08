"""Token usage extraction helpers for LangChain callback responses."""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

from contextunity.core.types import is_object_dict, is_object_list


class LangchainUsageDict(TypedDict, total=False):
    """Normalized token usage extracted from LangChain LLM responses."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float


@runtime_checkable
class LangchainLLMResult(Protocol):
    """Minimal LangChain LLM result surface for token extraction."""

    @property
    def llm_output(self) -> object: ...

    @property
    def generations(self) -> object: ...


@runtime_checkable
class LangchainGeneration(Protocol):
    """Single generation chunk from a LangChain LLM result."""

    @property
    def text(self) -> object: ...

    @property
    def message(self) -> object: ...


@runtime_checkable
class LangchainMessage(Protocol):
    """Chat message attached to a generation chunk."""

    @property
    def usage_metadata(self) -> object: ...

    @property
    def response_metadata(self) -> object: ...


@runtime_checkable
class AttributeCarrier(Protocol):
    """Generic attribute carrier for token usage metadata objects."""

    @property
    def input_tokens(self) -> object: ...

    @property
    def output_tokens(self) -> object: ...

    @property
    def total_tokens(self) -> object: ...

    @property
    def prompt_tokens(self) -> object: ...

    @property
    def completion_tokens(self) -> object: ...


def _int_token(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _float_cost(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _mapping_from_object(obj: object) -> dict[str, object]:
    if is_object_dict(obj):
        return dict(obj)
    if isinstance(obj, AttributeCarrier):
        return {
            "input_tokens": obj.input_tokens,
            "output_tokens": obj.output_tokens,
            "total_tokens": obj.total_tokens,
            "prompt_tokens": obj.prompt_tokens,
            "completion_tokens": obj.completion_tokens,
        }
    return {}


def _usage_from_token_mapping(data: dict[str, object]) -> LangchainUsageDict:
    prompt = _int_token(data.get("prompt_tokens"))
    completion = _int_token(data.get("completion_tokens"))
    total = _int_token(data.get("total_tokens"))
    if prompt is not None or completion is not None:
        return {
            "prompt_tokens": int(prompt or 0),
            "completion_tokens": int(completion or 0),
            "total_tokens": int(total or 0) or int(prompt or 0) + int(completion or 0),
        }

    inp = _int_token(data.get("input_tokens"))
    out = _int_token(data.get("output_tokens"))
    tot = _int_token(data.get("total_tokens"))
    if inp is not None or out is not None:
        return {
            "prompt_tokens": int(inp or 0),
            "completion_tokens": int(out or 0),
            "total_tokens": int(tot or 0) or int(inp or 0) + int(out or 0),
        }
    return {}


def _first_generation(response: LangchainLLMResult) -> LangchainGeneration | None:
    generations = response.generations
    if not is_object_list(generations) or not generations:
        return None
    first_row = generations[0]
    if not is_object_list(first_row) or not first_row:
        return None
    first_gen = first_row[0]
    return first_gen if isinstance(first_gen, LangchainGeneration) else None


def extract_generation_text(response: object) -> str:
    """Extract primary generation text from a LangChain LLM response object."""
    if not isinstance(response, LangchainLLMResult):
        return ""
    first_gen = _first_generation(response)
    if first_gen is None:
        return ""
    text_val = first_gen.text
    if isinstance(text_val, str):
        return text_val
    return str(first_gen)


def extract_langchain_usage(response: object) -> LangchainUsageDict:
    """Extract token usage from classic and modern LangChain LLM response shapes."""
    if not isinstance(response, LangchainLLMResult):
        return {}

    llm_output = response.llm_output
    if is_object_dict(llm_output):
        token_usage_raw = llm_output.get("token_usage")
        if is_object_dict(token_usage_raw):
            usage = _usage_from_token_mapping(token_usage_raw)
            if usage:
                return usage

    first_gen = _first_generation(response)
    if first_gen is None:
        return {}

    message = first_gen.message
    if isinstance(message, LangchainMessage):
        usage = _usage_from_token_mapping(_mapping_from_object(message.usage_metadata))
        if usage:
            return usage

        response_meta = message.response_metadata
        if is_object_dict(response_meta):
            token_usage_raw = response_meta.get("token_usage")
            if is_object_dict(token_usage_raw):
                usage = _usage_from_token_mapping(token_usage_raw)
                if usage:
                    total_cost = _float_cost(response_meta.get("total_cost"))
                    if total_cost is not None:
                        usage["total_cost"] = total_cost
                    return usage
            total_cost = _float_cost(response_meta.get("total_cost"))
            if total_cost is not None:
                return {"total_cost": total_cost}

    return {}
