# ContextRouter

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.md)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![GitHub](https://img.shields.io/badge/GitHub-ContextRouter-black.svg)](https://github.com/ContextRouter/contextrouter)
[![Docs](https://img.shields.io/badge/docs-contextrouter.org-green.svg)](https://contextrouter.org)

ContextRouter is a modular **"shared brain" framework for AI agents** (LangGraph-based). RAG is a core capability, but the framework is designed for **multi-step orchestration** (routing, extraction, generation, tool execution) and can be embedded into any platform—Web, Telegram, or custom APIs.

## Key Features

- **🧩 Plug-and-Play Architecture**: Easily swap LLMs, vector stores, and data connectors.
- **🧠 LangGraph Orchestration**: Sophisticated state management and conditional routing.
- **🛡️ Bisquit Protocol**: Built-in data provenance and security tracing for every piece of information.
- **📡 Streaming First**: Optimized for real-time SSE (Server-Sent Events) and event-driven UIs.
- **🌍 Pluggable Retrieval Sources**: Stable Vertex AI Search integration, optional web search connector, and stubs for upcoming backends.

## Production Deployment

For production deployments requiring advanced security, multi-tenancy, and enterprise integrations, visit [contextrouter.dev](https://contextrouter.dev) to learn about ContextRouter for production use.

Production features include:
- Advanced security for regulated industries
- Multi-tenant architecture with isolation
- Enterprise system integrations (SAP, Oracle, Salesforce)
- Production monitoring and alerting
- Deployment automation and scaling
- Professional support and SLAs

## Status / Support Matrix

**Golden path (supported end-to-end):**
- **Retrieval**: Vertex AI Search (`provider: vertex`)
- **LLM**: Gemini on Vertex (`vertex/gemini-*`)

| Capability | Component | Status | Notes |
|---|---|---:|---|
| Provider | Vertex (`modules/providers/storage/vertex.py`) | ✅ Stable (read) / ⚠️ Stub (write) | Retrieval works; ingestion sink write is not implemented |
| Provider | Postgres (`modules/providers/storage/postgres.py`) | ⚠️ Stub | Planned (see `PLAN/22-postgres-pgvector-knowledge-store.md`) |
| Provider | GCS (`modules/providers/storage/gcs.py`) | ⚠️ Stub | Write stub |
| Connector | Web Search (Google CSE) (`modules/connectors/web.py`) | ✅ Optional | Requires Google CSE credentials + allowlist domains |
| Connector | Web Scraper (`modules/connectors/web.py:web_scraper`) | ⚠️ Stub | Not implemented |
| Connector | RSS / API (`modules/connectors/rss.py`, `modules/connectors/api.py`) | ⚠️ Stub | Not implemented |
| Models | Vertex LLM/Embeddings (`modules/models/*/vertex.py`) | ✅ Supported | Driven by core config |
| Models | OpenAI/HF models | ⚠️ Stub | Placeholders for future community contributions |

## Quick Start

```python
from contextrouter.cortex import stream_agent

# Initialize the shared brain
async for event in stream_agent(
    messages=[{"role": "user", "content": "How does RAG work?"}],
    session_id="session_123",
    platform="web",
    style_prompt="Be concise and technical."
):
    print(event)
```

For more examples, see the [`examples/`](./examples/) directory.

## CLI

ContextRouter ships with a Click-based CLI:

```bash
contextrouter --help
contextrouter help
```

Common groups:
- `contextrouter rag ...`: run the golden-path RAG flow (Vertex AI Search + Gemini on Vertex)
- `contextrouter ingest ...`: ingestion pipeline utilities (separate ingestion config)

See [CLI documentation](https://contextrouter.org/cli) for the full reference.

## Configuration (Runtime / Core)

Runtime configuration is loaded via `contextrouter.core.config.get_core_config()` using:
**defaults < environment variables < `settings.toml`**.

### Minimal `settings.toml` (Golden path: Vertex AI Search + Gemini on Vertex)

Create a `settings.toml` in your working directory (or pass it explicitly via `--config`):

```toml
[vertex]
project_id = "my-gcp-project"
location = "us-central1"

[rag]
# Option 1: Use blue/green deployment pattern (recommended)
db_name = "blue"  # or "green"
data_store_id_blue = "projects/.../locations/.../collections/.../dataStores/..."
data_store_id_green = "projects/.../locations/.../collections/.../dataStores/..."

# Option 2: Direct datastore ID (skip blue/green)
# db_name = "projects/.../locations/.../collections/.../dataStores/..."

[models]
default_llm = "vertex/gemini-2.5-flash"
default_embeddings = "vertex/text-embedding"
```

**Note on `data_store_id_blue` / `data_store_id_green`**: These are only needed if you use the blue/green deployment pattern (`db_name = "blue"` or `"green"`). If `db_name` is already a full Vertex datastore ID, these fields are ignored.

### Plugin Configuration

ContextRouter supports dynamic loading of custom components:

```toml
[plugins]
paths = [
    "~/my-contextrouter-plugins",
    "./custom-components"
]

[router]
# Choose which graph to run
graph = "rag_retrieval"  # or custom registered graph
# graph = "my_custom_graph"
```

### Credentials for Golden Path (Vertex AI)

For the golden path (Vertex AI Search + Gemini), you need GCP authentication:

**Option 1: Application Default Credentials (ADC) — Recommended**
```bash
# Set up ADC (works with gcloud CLI, Cloud Run, GCE, etc.)
gcloud auth application-default login
```

**Option 2: Service Account Key**
```toml
[vertex]
project_id = "my-gcp-project"
location = "us-central1"
```

**Authentication Options:**

**Option 1: Application Default Credentials (ADC) - Recommended**
```bash
# Set up ADC (works with gcloud CLI, Cloud Run, GCE, etc.)
gcloud auth application-default login
```

**Option 2: Service Account Key via Environment Variable**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

The framework will automatically use ADC if available, falling back to `GOOGLE_APPLICATION_CREDENTIALS` if provided.

### Running with a config file

```bash
contextrouter --config ./settings.toml rag query "What is ContextRouter?"
```

### Passing values via environment variables

You can override TOML/defaults with env vars:

```bash
export CONTEXTROUTER_VERTEX_PROJECT_ID="my-gcp-project"
export CONTEXTROUTER_VERTEX_LOCATION="us-central1"
export RAG_DB_NAME="blue"
export DATA_STORE_ID_BLUE="projects/.../dataStores/..."
export DATA_STORE_ID_GREEN="projects/.../dataStores/..."
```

You can also point to a runtime config file:

```bash
export CONTEXTROUTER_CORE_CONFIG_PATH="/abs/path/to/settings.toml"
```

### Using `.env`

If you put environment variables into `.env` in the working directory, the CLI will load it automatically:

```bash
# .env
CONTEXTROUTER_VERTEX_PROJECT_ID=my-gcp-project
CONTEXTROUTER_VERTEX_LOCATION=us-central1
RAG_DB_NAME=blue
DATA_STORE_ID_BLUE=projects/.../dataStores/...
DATA_STORE_ID_GREEN=projects/.../dataStores/...

# For credentials, use ADC or:
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### Programmatic usage (Python)

```python
from contextrouter.core.config import load_config, set_core_config
from contextrouter.cortex import stream_agent

set_core_config(load_config(toml_path="settings.toml"))

# Now you can run the brain using the configured providers/models.
```

## Creating Plugins

ContextRouter supports **dynamic plugin loading** - you can add custom agents, tools, connectors, and graphs without modifying the core code.

### Custom Agent Example

Create `my_agent.py` in your plugin directory:

```python
from contextrouter.core.registry import register_agent

@register_agent("my_special_agent")
class MySpecialAgent:
    def __init__(self):
        self.name = "My Special Agent"

    def run(self, query: str, **kwargs) -> str:
        # Your custom logic here
        return f"Special response to: {query}"
```

### Custom Graph Example

```python
from langgraph.graph import END, START, StateGraph
from contextrouter.core.registry import register_graph
from contextrouter.cortex.state import AgentState, InputState, OutputState

@register_graph("my_custom_graph")
def build_my_custom_graph():
    def my_node(state: AgentState) -> AgentState:
        # Custom processing
        return state

    workflow = StateGraph(AgentState, input=InputState, output=OutputState)
    workflow.add_node("my_node", my_node)
    workflow.add_edge(START, "my_node")
    workflow.add_edge("my_node", END)

    return workflow
```

Then set in your config:
```toml
[router]
graph = "my_custom_graph"
```

## Architecture Overview

ContextRouter separates the AI "intelligence" from the technical implementation:

- **Cortex**: The orchestration layer. Manages the flow of the conversation, intent detection, and routing.
- **Modules**: Pluggable capabilities.
    - **Providers**: Interfaces for storage and databases (Vertex AI Search, Postgres).
    - **Connectors**: Fetchers for raw data (Web CSE, RSS, Local Files).
    - **Models**: Abstractions for LLMs (Gemini, GPT) and Embeddings.
- **Core**: The framework kernel handling configuration, security, and the `Bisquit` transport protocol.

## Directory Structure

```text
src/contextrouter/
├── core/                # Kernel: Registry, Config, Bisquit
├── cortex/              # Orchestration: Graph, State, Nodes, Steps
├── modules/             # Pluggable Capabilities
│   ├── providers/       # Storage implementations (Postgres, Vertex)
│   ├── connectors/      # Raw data fetchers (Web, API)
│   ├── retrieval/       # Search logic (Orchestration, Reranking)
│   └── models/          # LLM & Embedding abstractions
└── protocols/           # Transport mapping (AG-UI, Telegram)
```

## Documentation

Comprehensive documentation is available at [contextrouter.org](https://contextrouter.org):

- [Architecture Guide](https://contextrouter.org/architecture)
- [Configuration Reference](https://contextrouter.org/configuration)
- [Module Development](https://contextrouter.org/modules)
- [Roadmap](https://contextrouter.org/roadmap)

## Development Setup

ContextRouter uses modern Python tooling for development. This section covers setting up your development environment.

### Prerequisites

- **mise**: Version manager for Python, Node.js, and other tools
  ```bash
  # Install mise (see https://mise.jdx.dev/getting-started.html)
  curl https://mise.jdx.dev/install.sh | bash
  ```

- **Python 3.13+**: Required for development
- **uv**: Fast Python package manager (installed via mise)

### Quick Setup

1. **Clone and enter the project:**
   ```bash
   git clone https://github.com/ContextRouter/contextrouter.git
   cd contextrouter
   ```

2. **Install tools and dependencies:**
   ```bash
   # mise automatically installs Python 3.13 and uv
   mise install

   # Install all dependencies
   mise run sync
   ```

3. **Verify setup:**
   ```bash
   # Run all checks (lint + security + tests)
   mise run check
   ```

### Development Workflow

ContextRouter provides convenient commands via mise:

```bash
# Install/update dependencies
mise run sync

# Code quality checks
mise run lint          # Lint with Ruff
mise run format        # Format code
mise run security      # Security scan with Bandit

# Testing
mise run test          # Run all tests
mise run test-unit     # Run unit tests only

# Ingestion pipeline (development)
mise run ingest-run    # Full pipeline: preprocess → structure → index → deploy

# Cleanup
mise run clean         # Remove cache files
```

### Python Version Management

The project uses **Python 3.13** for development:

- **mise** manages Python versions per project
- **`.python-version`** specifies the required version
- **`.mise.toml`** configures tools and tasks

All commands automatically use the correct Python version - no manual activation needed!

### Code Quality Tools

- **Ruff**: Fast Python linter and formatter
- **Bandit**: Security vulnerability scanner for Python
- **pytest**: Testing framework with coverage reporting
- **pre-commit**: Git hooks for automated quality checks

### Pre-commit Hooks

Enable pre-commit hooks for automatic code quality checks:

```bash
# Install git hooks
mise run pre-commit install

# Or run manually on all files
mise run pre-commit run --all-files
```

### CI/CD

GitHub Actions runs automated checks on every push/PR:
- Python 3.13 environment
- Full test suite with coverage
- Security scanning
- Code quality checks

## Contributing

We welcome contributions! Please see our [Roadmap](https://contextrouter.org/roadmap) for current priorities and stubs awaiting implementation.

## Configuration Model (Core vs Ingestion)

- **Runtime (core config)**: Used by the agent at runtime (LLM selection, Vertex/Search settings, security). Loaded via `contextrouter.core.config.get_core_config()` from layered sources (defaults < env < TOML).
- **Ingestion config**: Separate, ingestion-only TOML used by ingestion CLI/stages (`modules/ingestion/rag/config.py`). This keeps ingestion knobs isolated from runtime.

## License

This project is licensed under the terms specified in [LICENSE.md](LICENSE.md).
