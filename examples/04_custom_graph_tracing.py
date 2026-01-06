"""Example: Custom Graph with Full Langfuse Tracing

This example demonstrates how to add comprehensive observability to custom graphs
using ContextRouter's Langfuse integration.

The example shows:
- Graph-level tracing with session/user tracking
- Operation-level spans with detailed metadata
- Streaming execution with tracing
- Custom span attributes and performance metrics

Run this example to see full tracing in action:
    python examples/04_custom_graph_tracing.py

Make sure Langfuse is configured in your settings.toml:
    [langfuse]
    secret_key = "your_secret_key"
    public_key = "your_public_key"
    host = "https://cloud.langfuse.com"
"""

import asyncio

from langgraph.graph import END, START, StateGraph

from contextrouter.cortex.state import AgentState, InputState, OutputState
from contextrouter.modules.observability import get_langfuse_callbacks, retrieval_span


def build_example_custom_graph():
    """Build a custom graph with comprehensive tracing."""

    def analyze_query(state: AgentState) -> AgentState:
        """Analyze user query with detailed tracing."""
        with retrieval_span(
            name="query_analysis", input_data={"query": state["messages"][-1].content}
        ) as span:
            query = state["messages"][-1].content

            # Simulate analysis
            analysis = {
                "intent": "question" if "?" in query else "statement",
                "topic": "ContextRouter" if "ContextRouter" in query else "general",
                "complexity": "high" if len(query) > 50 else "low",
            }

            span.metadata = analysis
            state["query_analysis"] = analysis

            return state

    def retrieve_context(state: AgentState) -> AgentState:
        """Retrieve relevant context with tracing."""
        with retrieval_span(
            name="context_retrieval",
            input_data={
                "analysis": state.get("query_analysis"),
                "query_length": len(state["messages"][-1].content),
            },
        ) as span:
            # Simulate retrieval from different sources
            analysis = state.get("query_analysis", {})

            docs = []
            if analysis.get("topic") == "ContextRouter":
                docs = [
                    {
                        "content": "ContextRouter is a modular LangGraph-powered agent framework for RAG applications.",
                        "source": "documentation",
                        "relevance": 0.95,
                    },
                    {
                        "content": "Supports custom graphs, plugins, and enterprise integrations.",
                        "source": "features",
                        "relevance": 0.87,
                    },
                ]
            else:
                docs = [
                    {
                        "content": "General knowledge document about AI frameworks.",
                        "source": "general",
                        "relevance": 0.65,
                    }
                ]

            span.metadata = {
                "docs_found": len(docs),
                "avg_relevance": (sum(d["relevance"] for d in docs) / len(docs) if docs else 0),
                "sources": list(set(d["source"] for d in docs)),
            }

            state["retrieved_docs"] = docs
            return state

    def generate_response(state: AgentState) -> AgentState:
        """Generate response with comprehensive tracing."""
        with retrieval_span(
            name="response_generation",
            input_data={
                "docs_count": len(state.get("retrieved_docs", [])),
                "analysis": state.get("query_analysis"),
            },
        ) as span:
            docs = state.get("retrieved_docs", [])
            analysis = state.get("query_analysis", {})

            if docs:
                # Generate based on retrieved docs
                context_summary = " ".join([doc["content"][:100] + "..." for doc in docs[:2]])
                response_text = f"Based on my knowledge: {context_summary}"
            else:
                response_text = "I don't have specific information about that topic."

            # Add analysis insights
            if analysis.get("complexity") == "high":
                response_text += " This seems like a complex topic that might benefit from more specific details."

            from langchain_core.messages import AIMessage

            response = AIMessage(content=response_text)
            state["messages"].append(response)

            span.metadata = {
                "response_length": len(response_text),
                "docs_used": len(docs),
                "response_type": "contextual" if docs else "fallback",
            }

            return state

    # Build the graph
    workflow = StateGraph(AgentState, input=InputState, output=OutputState)
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_response)

    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow


async def run_traced_custom_graph():
    """Run the custom graph with full Langfuse tracing."""
    print("🚀 Running custom graph with full Langfuse tracing...")

    # Build and compile the graph
    graph_builder = build_example_custom_graph()
    graph = graph_builder.compile()

    # Add comprehensive Langfuse tracing
    callbacks = get_langfuse_callbacks(
        session_id="custom_tracing_example",
        user_id="demo_user",
        platform="custom_graph_example",
        tags=["custom", "tracing", "example", "comprehensive"],
    )

    # Prepare input
    input_state = {
        "messages": [{"role": "user", "content": "What is ContextRouter and how does it work?"}],
        "session_id": "custom_tracing_example",
        "platform": "custom_graph_example",
        "citations_output": "raw",
    }

    # Execute with full tracing
    print("📊 Executing graph with Langfuse tracing...")
    result = await graph.ainvoke(input_state, config={"callbacks": callbacks})

    print("✅ Graph execution completed!")
    print(f"📝 Final messages count: {len(result['messages'])}")
    print(f"📄 Retrieved docs: {len(result.get('retrieved_docs', []))}")

    # Show the response
    final_message = result["messages"][-1]
    if hasattr(final_message, "content"):
        print(f"💬 Response: {final_message.content[:200]}...")

    return result


async def run_streaming_example():
    """Example of streaming with tracing."""
    print("\n🌊 Running streaming example with tracing...")

    graph_builder = build_example_custom_graph()
    graph = graph_builder.compile()

    callbacks = get_langfuse_callbacks(
        session_id="streaming_tracing_example",
        user_id="demo_user",
        platform="streaming_example",
        tags=["streaming", "tracing", "example"],
    )

    input_state = {
        "messages": [{"role": "user", "content": "Explain RAG architecture"}],
        "session_id": "streaming_tracing_example",
        "platform": "streaming_example",
    }

    print("📡 Streaming events with tracing...")
    async for event in graph.astream_events(
        input_state, config={"callbacks": callbacks}, version="v2"
    ):
        if event["event"] == "on_chain_end" and event.get("name") == "generate":
            print("🎯 Response generation completed!")
            break

    print("✅ Streaming example completed!")


if __name__ == "__main__":
    print("🔍 ContextRouter Custom Graph Tracing Example")
    print("=" * 50)

    # Run the main example
    asyncio.run(run_traced_custom_graph())

    # Run streaming example
    asyncio.run(run_streaming_example())

    print("\n📈 Check your Langfuse dashboard to see the complete trace!")
    print("🔗 You'll see spans for: query_analysis, context_retrieval, response_generation")
    print("📊 Each span includes metadata about the operation and performance metrics")
