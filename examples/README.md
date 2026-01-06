# ContextRouter Examples

This directory contains examples demonstrating various ContextRouter features and integrations.

## Available Examples

### Core Features
- **[01_basic_agent.py](01_basic_agent.py)** - Basic agent usage with streaming
- **[02_custom_connector.py](02_custom_connector.py)** - Custom data connector implementation
- **[03_standalone_retrieval.py](03_standalone_retrieval.py)** - Standalone RAG retrieval without full agent
- **[04_custom_graph_tracing.py](04_custom_graph_tracing.py)** - Custom graphs with full Langfuse tracing
- **[05_custom_errors.py](05_custom_errors.py)** - Custom error handling and recovery

### Plugins
- **[plugins/](plugins/)** - Custom plugin examples

### Configuration
- **[settings.toml.example](settings.toml.example)** - Example configuration file

## Running Examples

Each example can be run independently:

```bash
# Basic agent example
python examples/01_basic_agent.py

# Custom graph with tracing
python examples/04_custom_graph_tracing.py
```

Make sure to:
1. Copy `settings.toml.example` to your working directory as `settings.toml`
2. Configure your API keys and settings
3. Install ContextRouter: `pip install -e .`

## Example Categories

### 🔧 **Core Usage**
Basic ContextRouter functionality, streaming, error handling.

### 🔌 **Extensions**
Custom connectors, plugins, and integrations.

### 📊 **Observability**
Langfuse tracing, monitoring, and analytics examples.

### 🚀 **Advanced**
Custom graphs, enterprise features, production deployments.

## Contributing Examples

When adding new examples:
1. Follow the numbering scheme (next available number)
2. Include comprehensive docstrings
3. Add proper error handling
4. Update this README
5. Test with both configured and unconfigured Langfuse
