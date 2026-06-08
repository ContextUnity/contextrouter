"""Typed boundaries for optional HuggingFace ``transformers`` / ``torch`` imports."""

from __future__ import annotations

import importlib
from typing import Protocol

from contextunity.core.exceptions import ConfigurationError
from contextunity.core.narrowing import optional_str_field
from contextunity.core.types import is_json_dict, is_object_list


class _TransformersPipelineFactory(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


def load_transformers_pipeline() -> _TransformersPipelineFactory:
    """Return ``transformers.pipeline`` when installed."""
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            (
                "HuggingFace transformers not installed. "
                "HuggingFaceLLM requires `contextunity.router[hf-transformers]`."
            )
        ) from exc
    pipeline_obj: object = getattr(transformers, "pipeline", None)
    if not callable(pipeline_obj):
        raise ConfigurationError("transformers.pipeline is unavailable")
    return pipeline_obj


def torch_cuda_device_index() -> int:
    """Return ``0`` when CUDA is available, otherwise ``-1`` for CPU pipelines."""
    try:
        torch_mod = importlib.import_module("torch")
    except ImportError:
        return -1
    cuda_obj: object = getattr(torch_mod, "cuda", None)
    if cuda_obj is None:
        return -1
    is_available_fn: object = getattr(cuda_obj, "is_available", None)
    if callable(is_available_fn) and bool(is_available_fn()):
        return 0
    return -1


def classification_label_score(row: object) -> tuple[str | None, float | None]:
    """Extract label/score from a transformers classification row."""
    if not is_json_dict(row):
        return None, None
    label_val = row.get("label")
    score_val = row.get("score")
    label = label_val if isinstance(label_val, str) else None
    score = float(score_val) if isinstance(score_val, (int, float)) else None
    return label, score


def object_dict_get_str(data: object, key: str) -> str | None:
    """Read a string field from an object dict payload."""
    return optional_str_field(data, key)


def tokenizer_encode_length(tokenizer: object, text: str) -> int | None:
    """Return token count from a tokenizer ``encode`` call when supported."""
    encode_fn: object = getattr(tokenizer, "encode", None)
    if not callable(encode_fn):
        return None
    tokens: object = encode_fn(text)
    if is_object_list(tokens):
        return len(tokens)
    return None


__all__ = [
    "classification_label_score",
    "load_transformers_pipeline",
    "object_dict_get_str",
    "tokenizer_encode_length",
    "torch_cuda_device_index",
]
