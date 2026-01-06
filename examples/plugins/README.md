# ContextRouter Plugin Examples

This directory contains examples of how to extend ContextRouter with custom plugins without modifying the core codebase.

## How Plugin Scanning Works

ContextRouter automatically scans directories listed in your `settings.toml` configuration:

```toml
[plugins]
paths = [
    "~/my-contextrouter-plugins",
    "./examples/plugins"
]
```

When ContextRouter starts, it imports all `.py` files from these directories, allowing you to register custom components.

## Plugin Types

### 1. Custom Agents
Use `@register_agent()` to add new agent types:

```python
from contextrouter.core.registry import register_agent

@register_agent("my_agent")
class MyAgent:
    def run(self, query: str, **kwargs) -> str:
        # Your agent logic
        return f"Processed: {query}"
```

### 2. Custom Connectors
Use `@register_connector()` to add new data sources:

```python
from contextrouter.core.registry import register_connector

@register_connector("my_api")
class MyAPIConnector:
    def fetch(self, query: str, **kwargs) -> list[dict]:
        # Your data fetching logic
        return [{"content": "data", "title": "result"}]
```

### 3. Custom Transformers
Use `@register_transformer()` for data processing:

```python
from contextrouter.core.registry import register_transformer

@register_transformer("my_cleaner")
def my_text_cleaner(text: str) -> str:
    # Your processing logic
    return text.upper()
```

### 4. Custom Providers
Use `@register_provider()` for storage backends:

```python
from contextrouter.core.registry import register_provider

@register_provider("my_storage")
class MyStorageProvider:
    def store(self, data: Any) -> None:
        # Your storage logic
        pass
```

## Usage

1. Create your plugin files in a directory
2. Add the directory path to `settings.toml` under `[plugins].paths`
3. Restart ContextRouter - plugins will be loaded automatically

## Example Plugin Files

- `custom_agent.py` - Shows how to add a custom agent
- `custom_connector.py` - Shows how to add a custom data connector

## Best Practices

- Use descriptive names for your components
- Handle errors gracefully in your plugins
- Test your plugins independently before deploying
- Document your plugin interfaces clearly

## Debugging

If your plugin doesn't load, check the logs for import errors. Common issues:
- Missing dependencies
- Syntax errors in plugin files
- Incorrect decorator usage
- Path not added to `settings.toml`
