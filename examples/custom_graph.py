"""Example custom graph plugin for ContextRouter with Langfuse tracing.

This demonstrates how to add custom graphs with full observability support.
Place this file in a directory listed in your settings.toml [plugins].paths
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from contextrouter.core.registry import register_graph
from contextrouter.cortex.state import AgentState, InputState, OutputState
from contextrouter.modules.observability import get_langfuse_callbacks, retrieval_span


@register_graph("simple_echo_graph")
def build_simple_echo_graph():
    """Build a simple echo graph that just returns the input."""

    def echo_node(state: AgentState) -> AgentState:
        """Simple echo node that adds a response."""
        last_message = state["messages"][-1]
        response_content = f"Echo: {last_message.content}"

        from langchain_core.messages import AIMessage

        response = AIMessage(content=response_content)
        state["messages"].append(response)
        return state

    workflow = StateGraph(AgentState, input=InputState, output=OutputState)
    workflow.add_node("echo", echo_node)
    workflow.add_edge(START, "echo")
    workflow.add_edge("echo", END)

    return workflow


@register_graph("custom_rag_graph")
def build_custom_rag_graph():
    """Build a custom RAG graph with simplified logic."""

    def custom_retrieve(state: AgentState) -> AgentState:
        """Custom retrieval logic with tracing."""
        with retrieval_span(
            name="custom_retrieve", input_data={"query": state["messages"][-1].content[:100]}
        ) as span_ctx:
            # Your custom retrieval implementation
            state["retrieved_docs"] = [
                {"content": "Custom retrieved content", "title": "Custom Doc"}
            ]
            span_ctx["metadata"] = {"docs_found": len(state["retrieved_docs"])}
            span_ctx["output"] = {"docs": state["retrieved_docs"]}
            return state

    def custom_generate(state: AgentState) -> AgentState:
        """Custom generation logic with tracing."""
        with retrieval_span(
            name="custom_generate", input_data={"docs_count": len(state.get("retrieved_docs", []))}
        ) as span_ctx:
            from langchain_core.messages import AIMessage

            # Simple generation based on retrieved docs
            docs = state.get("retrieved_docs", [])
            if docs:
                response = f"Based on custom retrieval: {docs[0]['content']}"
            else:
                response = "No documents found for generation"

            ai_message = AIMessage(content=response)
            state["messages"].append(ai_message)

            span_ctx["metadata"] = {"response_length": len(response), "docs_used": len(docs)}
            span_ctx["output"] = {"response": response}
            return state

    workflow = StateGraph(AgentState, input=InputState, output=OutputState)
    workflow.add_node("custom_retrieve", custom_retrieve)
    workflow.add_node("custom_generate", custom_generate)

    workflow.add_edge(START, "custom_retrieve")
    workflow.add_edge("custom_retrieve", "custom_generate")
    workflow.add_edge("custom_generate", END)

    return workflow


def run_custom_graph_with_tracing():
    """Example of running a custom graph with full Langfuse tracing.

    This function demonstrates how to add observability to custom graphs
    that are not part of the standard ContextRouter runners.
    """
    # Build your custom graph
    graph_builder = build_custom_rag_graph()
    graph = graph_builder.compile()

    # Add Langfuse callbacks for full tracing
    callbacks = get_langfuse_callbacks(
        session_id="custom_graph_example",
        user_id="example_user",
        platform="custom_plugin",
        tags=["custom", "example", "tracing"],
    )

    # Prepare input state
    input_state = {
        "messages": [HumanMessage(content="What is ContextRouter?")],
        "session_id": "custom_graph_example",
        "platform": "custom_plugin",
    }

    # Execute with full tracing
    result = graph.invoke(input_state, config={"callbacks": callbacks})

    print(f"Custom graph executed with tracing. Messages: {len(result['messages'])}")
    return result


@register_graph("traced_custom_graph")
def build_traced_custom_graph():
    """Build a custom graph with built-in tracing spans."""

    def traced_retrieve_node(state: AgentState) -> AgentState:
        """Retrieval node with detailed tracing."""
        with retrieval_span(
            name="traced_retrieve", input_data={"query": state["messages"][-1].content[:100]}
        ) as span_ctx:
            # Your custom retrieval logic here
            state["retrieved_docs"] = [
                {"content": "ContextRouter is a modular RAG framework", "title": "About"},
                {"content": "Built with LangGraph for orchestration", "title": "Architecture"},
                {"content": "Supports custom graphs and plugins", "title": "Extensibility"},
            ]

            span_ctx["metadata"] = {
                "retrieved_docs": len(state["retrieved_docs"]),
                "search_time_ms": 150,
                "sources": ["documentation", "features"],
            }
            span_ctx["output"] = {"docs": state["retrieved_docs"]}
            return state

    def traced_generate_node(state: AgentState) -> AgentState:
        """Generation node with tracing."""
        with retrieval_span(
            name="traced_generate",
            input_data={
                "docs_count": len(state.get("retrieved_docs", [])),
                "query": state["messages"][-1].content[:50],
            },
        ) as span_ctx:
            # Your custom generation logic here
            docs = state.get("retrieved_docs", [])
            response_text = (
                f"Based on {len(docs)} documents: ContextRouter is a powerful RAG framework!"
            )

            from langchain_core.messages import AIMessage

            response = AIMessage(content=response_text)
            state["messages"].append(response)

            span_ctx["metadata"] = {
                "response_length": len(response_text),
                "docs_used": len(docs),
                "generation_time_ms": 200,
            }
            span_ctx["output"] = {"response_text": response_text}
            return state

    workflow = StateGraph(AgentState, input=InputState, output=OutputState)
    workflow.add_node("traced_retrieve", traced_retrieve_node)
    workflow.add_node("traced_generate", traced_generate_node)

    workflow.add_edge(START, "traced_retrieve")
    workflow.add_edge("traced_retrieve", "traced_generate")
    workflow.add_edge("traced_generate", END)

    return workflow


# Example usage (uncomment to run)
# if __name__ == "__main__":
#     result = run_custom_graph_with_tracing()
#     print("Result:", result)

# For comprehensive tracing examples, see:
# examples/04_custom_graph_tracing.py

# To use with Langfuse tracing:
# 1. Configure Langfuse in settings.toml
# 2. Call run_custom_graph_with_tracing() for graph-level tracing
# 3. Use @register_graph("traced_custom_graph") for plugin-based tracing

print("Custom graph plugin with tracing loaded!")
