"""Flow orchestration: Connector -> Transformer(s) -> Provider.
This module is intentionally small and deterministic:
- no hidden imports
- no global singletons
- explicit registry lookups
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from contextunity.core.tokens import ContextToken, TokenBuilder
from contextunity.core.types import WireValue, is_json_dict
from pydantic import BaseModel, ConfigDict, Field

from contextunity.router.core.config import RouterConfig, get_core_config
from contextunity.router.core.exceptions import ContextrouterError
from contextunity.router.core.interfaces import BaseConnector, BaseProvider, BaseTransformer
from contextunity.router.core.registry import ComponentFactory, Registry


class FlowConfig(BaseModel):
    """Configuration for a specific data processing flow.

    This is used by the flow manager to execute custom processing pipelines.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    # Flow identification
    name: str = ""
    description: str = ""

    # Source configuration
    source: str = ""  # e.g., "web", "file", "api"
    source_params: dict[str, WireValue] = Field(default_factory=dict)

    # Processing pipeline
    logic: list[str] = Field(default_factory=list)  # Transformer names
    logic_params: dict[str, WireValue] = Field(default_factory=dict)

    # Sink configuration
    sink: str = ""  # e.g., "vertex", "postgres"
    sink_params: dict[str, WireValue] = Field(default_factory=dict)

    # Execution controls
    overwrite: bool = True
    workers: int = 1


@dataclass(frozen=True)
class FlowResult:
    """Outcome of a flow execution — item count and per-item sink results."""

    processed: int
    results: list[object]


class FlowManager:
    """Orchestrate a Connector → Transformer chain → Provider (sink) pipeline per ``FlowConfig``."""

    def __init__(
        self,
        *,
        connector_registry: Registry[type[BaseConnector]],
        transformer_registry: Registry[type[BaseTransformer]],
        provider_registry: Registry[type[BaseProvider]],
        config: RouterConfig | None = None,
        token_builder: TokenBuilder | None = None,
    ) -> None:
        """Store the three component registries, resolve config, and prepare a default ``TokenBuilder``."""
        self._connectors: Registry[type[BaseConnector]] = connector_registry
        self._transformers: Registry[type[BaseTransformer]] = transformer_registry
        self._providers: Registry[type[BaseProvider]] = provider_registry
        self._config: RouterConfig = config or get_core_config()
        self._token_builder: TokenBuilder = token_builder or TokenBuilder()

    async def run(self, flow: FlowConfig, *, token: ContextToken) -> FlowResult:
        """Execute the connect → transform → sink pipeline for *flow* and return aggregated results."""
        connector = ComponentFactory.create_connector(flow.source, **(flow.source_params or {}))
        transformers: list[BaseTransformer] = []
        for key in flow.logic:
            raw_params = flow.logic_params.get(key)
            params: dict[str, WireValue] = {}
            if is_json_dict(raw_params):
                for param_key, param_value in raw_params.items():
                    params[param_key] = param_value
            transformers.append(ComponentFactory.create_transformer(key, **params))

        sink_key = flow.sink.strip()
        # Reserved core sinks (not external providers):
        # - "context"/"agent_context": return the ContextUnit to the caller to attach to state
        # - "response": return envelope.content (synced with legacy envelope.data)
        is_context_sink = sink_key in {"context", "agent_context"}
        is_response_sink = sink_key == "response"

        provider: BaseProvider | None = None
        if not (is_context_sink or is_response_sink):
            provider = ComponentFactory.create_provider(sink_key, **(flow.sink_params or {}))

        processed = 0
        results: list[object] = []

        async for unit in connector.connect():
            # Audit hook: ensure token_id is attached if caller provided token.
            if getattr(token, "token_id", None):
                unit.payload["token_id"] = token.token_id

            cur = unit
            for t in transformers:
                cur = await t.transform(cur)

            if is_context_sink:
                # Core handles it: caller decides how to persist to state.
                results.append(cur)
            elif is_response_sink:
                # Core handles it: return content from payload.
                results.append(cur.payload.get("content") if cur.payload else None)
            else:
                if provider is None:
                    raise ContextrouterError("Provider must be initialized for sink operations")
                # Enforce external sink boundary through gRPC interceptors.
                results.append(await provider.sink(cur))
            processed += 1

        return FlowResult(processed=processed, results=results)


__all__ = ["FlowManager", "FlowConfig", "FlowResult"]
