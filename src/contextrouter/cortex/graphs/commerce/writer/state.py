"""
Commerce product writer state.
"""

from typing import TypedDict


class WriterState(TypedDict):
    tenant_id: str
    trace_id: str

    title: str
    product_type: str
    brand: str
    model: str
    extra: str

    model_key: str

    descriptions: dict[str, str]
