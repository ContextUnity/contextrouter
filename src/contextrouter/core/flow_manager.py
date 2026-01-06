"""Flow orchestration: Connector -> Transformer(s) -> Provider.

This module is intentionally small and deterministic:
- no hidden imports
- no global singletons
- explicit registry lookups
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextrouter.core.config import Config, FlowConfig, get_core_config
from contextrouter.core.interfaces import BaseConnector, BaseProvider, BaseTransformer
from contextrouter.core.registry import ComponentFactory, Registry
from contextrouter.core.tokens import AccessManager, BiscuitToken, TokenBuilder


@dataclass(frozen=True)
class FlowResult:
    """Debug-friendly flow output."""

    processed: int
    results: list[Any]


class FlowManager:
    def __init__(
        self,
        *,
        connector_registry: Registry[type[BaseConnector]],
        transformer_registry: Registry[type[BaseTransformer]],
        provider_registry: Registry[type[BaseProvider]],
        config: Config | None = None,
        token_builder: TokenBuilder | None = None,
        access_manager: AccessManager | None = None,
    ) -> None:
        self._connectors = connector_registry
        self._transformers = transformer_registry
        self._providers = provider_registry
        self._config = config or get_core_config()
        self._token_builder = token_builder or TokenBuilder(
            enabled=self._config.security.enabled,
            private_key_path=self._config.security.private_key_path,
        )
        self._access = access_manager or AccessManager(
            config=self._config, token_builder=self._token_builder
        )

    async def run(self, flow: FlowConfig, *, token: BiscuitToken) -> FlowResult:
        connector = ComponentFactory.create_connector(flow.source, **(flow.source_params or {}))
        transformers = [
            ComponentFactory.create_transformer(
                key,
                **(
                    flow.logic_params.get(key)
                    if isinstance(flow.logic_params, dict) and flow.logic_params.get(key)
                    else {}
                ),
            )
            for key in flow.logic
        ]

        sink_key = flow.sink.strip()
        # Reserved core sinks (not external providers):
        # - "context"/"agent_context": return the BisquitEnvelope to the caller to attach to state
        # - "response": return envelope.content (synced with legacy envelope.data)
        is_context_sink = sink_key in {"context", "agent_context"}
        is_response_sink = sink_key == "response"

        provider: BaseProvider | None = None
        if not (is_context_sink or is_response_sink):
            provider = ComponentFactory.create_provider(sink_key, **(flow.sink_params or {}))

        processed = 0
        results: list[Any] = []

        async for envelope in connector.connect():
            # Audit hook: ensure token_id is attached if caller provided token.
            if getattr(token, "token_id", None):
                envelope.sign(token.token_id)

            cur = envelope
            for t in transformers:
                cur = await t.transform(cur)

            if is_context_sink:
                # Core handles it: caller decides how to persist to state.
                results.append(cur)
            elif is_response_sink:
                # Core handles it: return content (legacy callers may still read .data).
                results.append(cur.content)
            else:
                if provider is None:
                    raise ValueError("Provider must be initialized for sink operations")
                # Enforce write permission + envelope token_id consistency at external sink boundary.
                self._access.verify_envelope_write(cur, token)
                results.append(await provider.sink(cur, token=token))
            processed += 1

        return FlowResult(processed=processed, results=results)


__all__ = ["FlowManager", "FlowConfig", "FlowResult"]
