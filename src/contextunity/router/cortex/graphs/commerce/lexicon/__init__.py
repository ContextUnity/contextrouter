"""
Lexicon subgraph package.

Lexicon: AI content generation for products.

Nodes:
    analyze   — Fetch products from Brain and prepare requests
    generate  — LLM-powered content generation (title, description, features)
    validate  — Quality gates (length, format)
    write     — Write validated content back to Brain
"""

from .graph import compile_lexicon_graph, create_lexicon_subgraph

__all__ = ["create_lexicon_subgraph", "compile_lexicon_graph"]
