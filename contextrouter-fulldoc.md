# ContextRouter: Reasoning Engine

## Overview
ContextRouter is the **Orchestrator**. It hosts the Agent Graph (LangGraph) and executes reasoning logic.

## Architecture
- **Framework**: FastAPI + LangGraph.
- **Components**:
    - `cortex/graphs`: Behavioral definition of agents (Chat, Gardener, Matcher).
    - `cortex/steps`: Atomic reasoning blocks (Retrieval, Tool Use).
    - `modules/retrieval`: RAG Pipeline (using Brain as backend).

## Agents
1.  **Brain Agent**: General purpose Chat/QA.
2.  **Gardener**: Taxonomy classification & enrichment.
3.  **Matcher**: Product linking agent.

## Tools
Router exposes tools from Brain (Search) and Commerce (ProductDB) to LLMs.
