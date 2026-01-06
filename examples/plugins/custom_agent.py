"""Example custom agent plugin for ContextRouter.

This demonstrates how to add custom agents without modifying the core codebase.
Place this file in a directory listed in your settings.toml [plugins].paths
"""

from __future__ import annotations

from contextrouter.core.registry import register_agent, register_transformer


@register_agent("custom_research_agent")
class CustomResearchAgent:
    """A custom research agent that combines multiple tools."""

    def __init__(self):
        """Initialize the custom agent."""
        self.name = "Custom Research Agent"
        self.description = "Advanced research agent with custom tools"

    def run(self, query: str, **kwargs) -> str:
        """Execute the research workflow."""
        # Your custom agent logic here
        return f"Research completed for: {query}. This is a custom implementation."


@register_transformer("custom_cleaner")
def custom_text_cleaner(text: str) -> str:
    """Custom text cleaning transformer."""
    # Your custom cleaning logic
    return text.strip().upper()


print("Custom agent plugin loaded!")
