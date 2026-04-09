"""
Commerce product writer subgraph.
"""

from __future__ import annotations

from contextcore import get_context_unit_logger
from langgraph.graph import END, START, StateGraph

from .nodes import make_generate_descriptions
from .state import WriterState

logger = get_context_unit_logger(__name__)


def build_writer_graph():
    """Create Writer subgraph for product descriptions.

    Flow:
        generate → END
    """
    from contextrouter.cortex.graphs.secure_node import make_secure_node

    graph = StateGraph(WriterState)

    node_gen = make_generate_descriptions()
    secure_gen = make_secure_node("generate", node_gen, model_secret_ref="generation_llm")

    graph.add_node("generate", secure_gen)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", END)

    return graph.compile()
