"""Typed Protocol boundaries for optional embedding SDKs."""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

from contextunity.core.exceptions import ConfigurationError
from contextunity.core.types import is_object_list


@runtime_checkable
class SentenceTransformerModel(Protocol):
    def encode(self, sentences: list[str]) -> object: ...


@runtime_checkable
class VertexEmbeddingValues(Protocol):
    values: list[float]


@runtime_checkable
class VertexTextEmbeddingModel(Protocol):
    @classmethod
    def from_pretrained(cls, model_name: str) -> VertexTextEmbeddingModel: ...

    def get_embeddings(self, texts: list[str]) -> list[VertexEmbeddingValues]: ...


def first_encode_row(batch: object) -> object:
    """Return the first embedding row from a batch ``encode`` result."""
    if is_object_list(batch):
        if not batch:
            raise ConfigurationError("Embedding encode returned an empty batch")
        return batch[0]
    getter: object = getattr(batch, "__getitem__", None)
    if callable(getter):
        return getter(0)
    raise ConfigurationError("Embedding encode returned an unsupported batch shape")


def _float_from_wire(value: object) -> float:
    if isinstance(value, bool):
        raise ConfigurationError("Embedding encode returned a boolean vector element")
    if isinstance(value, (int, float)):
        return float(value)
    raise ConfigurationError("Embedding encode returned a non-numeric vector element")


def float_vector_from_encode(result: object) -> list[float]:
    """Normalize a single ``encode`` result row to ``list[float]``."""
    tolist: object = getattr(result, "tolist", None)
    if callable(tolist):
        values_obj: object = tolist()
        if is_object_list(values_obj):
            return [_float_from_wire(item) for item in values_obj]
    if is_object_list(result):
        return [_float_from_wire(item) for item in result]
    raise ConfigurationError("Embedding encode returned an unsupported vector shape")


def float_vectors_from_encode_batch(result: object) -> list[list[float]]:
    """Normalize a batch ``encode`` result to ``list[list[float]]``."""
    if not is_object_list(result):
        raise ConfigurationError("Embedding encode batch returned a non-list result")
    return [float_vector_from_encode(row) for row in result]


def load_sentence_transformer(model_name: str) -> SentenceTransformerModel:
    """Load ``SentenceTransformer`` when the optional dependency is installed."""
    try:
        module = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise ImportError(
            "SentenceTransformers requires `contextunity.router[hf-embeddings]`."
        ) from exc
    model_cls: object = getattr(module, "SentenceTransformer", None)
    if not callable(model_cls):
        raise ConfigurationError("sentence_transformers.SentenceTransformer is unavailable")
    model_obj: object = model_cls(model_name)
    if not isinstance(model_obj, SentenceTransformerModel):
        raise ConfigurationError("SentenceTransformer instance is incompatible")
    return model_obj


def init_vertexai(project_id: str, location: str) -> None:
    """Initialize the Vertex AI SDK for the given project and region."""
    vertexai = importlib.import_module("vertexai")
    init_fn: object = getattr(vertexai, "init", None)
    if not callable(init_fn):
        raise ConfigurationError("vertexai.init is unavailable")
    _ = init_fn(project=project_id, location=location)


def load_vertex_text_embedding_model(model_name: str) -> VertexTextEmbeddingModel:
    """Load Vertex ``TextEmbeddingModel`` when the optional SDK is installed."""
    try:
        language_models = importlib.import_module("vertexai.preview.language_models")
    except Exception as exc:
        raise ConfigurationError(
            "vertexai SDK is required for Vertex embeddings. Install extras: contextunity.router[vertex]"
        ) from exc
    model_cls: object = getattr(language_models, "TextEmbeddingModel", None)
    if model_cls is None:
        raise ConfigurationError("vertexai TextEmbeddingModel is unavailable")
    from_pretrained_fn: object = getattr(model_cls, "from_pretrained", None)
    if not callable(from_pretrained_fn):
        raise ConfigurationError("vertexai TextEmbeddingModel.from_pretrained is unavailable")
    model_obj: object = from_pretrained_fn(model_name)
    if not isinstance(model_obj, VertexTextEmbeddingModel):
        raise ConfigurationError("Vertex TextEmbeddingModel instance is incompatible")
    return model_obj


def vertex_embedding_values(model: VertexTextEmbeddingModel, texts: list[str]) -> list[list[float]]:
    """Return embedding value rows from a Vertex text embedding model."""
    rows = model.get_embeddings(texts)
    return [[float(x) for x in row.values] for row in rows]


__all__ = [
    "SentenceTransformerModel",
    "VertexTextEmbeddingModel",
    "float_vector_from_encode",
    "float_vectors_from_encode_batch",
    "init_vertexai",
    "load_sentence_transformer",
    "load_vertex_text_embedding_model",
    "vertex_embedding_values",
]
