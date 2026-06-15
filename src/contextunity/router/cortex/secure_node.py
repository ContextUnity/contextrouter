"""Capability stripping wrapper for LangGraph nodes.

This module provides execution sandboxing for declarative node execution.
It intercepts node invocation to:
1. Dynamically resolve and attenuate the caller's ContextToken down to the
   exact subset of permissions (scopes) required by the node's specification.
2. Verify prompt integrity using HMAC signatures to prevent injection.
3. Ensure PII anonymization is applied where requested by project policy.
4. Propagate the stripped context token using a safe contextvar lifecycle.
"""

from __future__ import annotations

import functools
import inspect
from contextvars import Token
from typing import NamedTuple

from contextunity.core import ContextToken, get_contextunit_logger
from contextunity.core.exceptions import SecurityError
from contextunity.core.tokens import TokenBuilder
from contextunity.core.types import JsonDict, is_json_dict
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig

from contextunity.router.core.context import (
    get_current_access_token,
    reset_current_access_token,
    set_current_access_token,
)
from contextunity.router.cortex.compiler.state_routing import STATE_TOP_LEVEL_KEYS
from contextunity.router.cortex.compiler.types import CompilerNodeSpec
from contextunity.router.cortex.config_resolution import (
    get_graph_runtime_config,
    get_node_attr,
    get_node_config,
    metadata_project_config,
)
from contextunity.router.cortex.events import BrainEvent
from contextunity.router.cortex.tenant_scope import resolve_node_effective_tenants
from contextunity.router.cortex.types import (
    ExecutionMetadata,
    GraphState,
    NodeFunc,
    StateUpdate,
)

logger = get_contextunit_logger(__name__)

_VALID_TOOL_BINDING_MODES = frozenset({"read", "write", "execute", "admin"})


class _ResolvedSecurity(NamedTuple):
    """Result of scope + PII resolution for a secure node."""

    required_scopes: list[str]
    """Token permissions this node needs."""
    pii_masking: bool
    """Whether PII anonymization is active for this node."""
    effective_tenants: tuple[str, ...]
    """Narrowed tenant scope for this node (attenuated into the context token)."""


def make_secure_node(
    node_name: str,
    node_func: NodeFunc,
    node_spec: CompilerNodeSpec | None = None,
    *,
    requires_llm: bool = True,
    prompt_signature: str | None = None,
    _schema_tools: list[str] | None = None,
    execute_tools: list[str] | None = None,
    service_scopes: list[str] | None = None,
    pass_through_token: bool = False,
) -> NodeFunc:
    """Wrap a LangGraph node to enforce strict capability stripping and security invariants.

    Capability stripping restricts the node's active security context (via ContextToken attenuation)
    to only the permissions explicitly required by the node, such as reading specific secrets,
    invoking particular tools, or performing PII anonymization/deanonymization.

    Args:
        node_name: The name of the node in the compiled graph topology.
        node_func: The actual callable execution logic of the node.
        node_spec: Declarative specification from the project manifest containing node
            settings, such as PII masking overrides or specific secret references.
        requires_llm: Whether this node invokes a large language model. If True, automatically
            resolves the required Shield secret read permissions for the model API key.
        prompt_signature: Optional HMAC signature of the prompt content to verify integrity.
        _schema_tools: Tool names whose schemas are visible to the node.
        execute_tools: Tool names that the node has permission to execute.
        service_scopes: Explicit platform-level scopes required for external service calls.
        pass_through_token: If True, bypasses token attenuation, allowing the node to preserve
            and propagate the original caller token without capability stripping.

    Returns:
        A wrapped async function that executes the node with attenuated permissions, integrity
        checks, PII masking rules, and custom telemetry events.

    Raises:
        SecurityError: If the token is missing, invalid, or does not contain required scopes.
        TamperDetectedError: If the node's prompt text does not match the registered HMAC signature.
    """
    # Read manifest-driven settings from node_spec (built-in graphs pass None)
    spec = node_spec or {}
    pii_masking: bool = bool(spec.get("pii_masking", True))
    model_secret_ref: str | None = spec.get("model_secret_ref")

    def execute_with_token(
        _state: GraphState,
        required_scopes: list[str],
        effective_tenants: tuple[str, ...],
    ) -> tuple[Token[ContextToken | None], ContextToken]:
        """Attenuate the current context token to the node's required scopes and tenant set.

        This sets the attenuated token as the active access token in the current contextvar
        context, enabling capability stripping for the duration of the node execution.

        Args:
            _state: The current LangGraph state dictionary.
            required_scopes: A list of permission scopes required by the node.
            effective_tenants: Narrowed tenant scope for this node (subset of project/graph).

        Returns:
            A tuple of (previous_tokenvar_token, new_attenuated_token).

        Raises:
            SecurityError: If the thread context lacks a valid active access token.
        """
        token = get_current_access_token()
        if not token:
            raise SecurityError(
                message=(
                    f"Node '{node_name}' cannot execute without an active ContextToken. "
                    "This indicates broken security context propagation — "
                    "the gRPC entry point must set the token before graph execution."
                ),
                node_name=node_name,
            )

        if pass_through_token:
            logger.debug("Node '%s' capability pass-through: preserving original token", node_name)
            return set_current_access_token(token), token

        logger.debug(
            "Node '%s' capability stripping: permissions=%s tenants=%s",
            node_name,
            required_scopes,
            list(effective_tenants),
        )

        attenuated_token = TokenBuilder().attenuate(
            token,
            permissions=required_scopes,
            allowed_tenants=effective_tenants,
            agent_id=f"node:{node_name}",
        )

        return set_current_access_token(attenuated_token), attenuated_token

    def _resolve_scopes_and_pii(state: GraphState) -> _ResolvedSecurity:
        """Resolve the final permission scopes and PII masking status for the node.

        Combines static parameters with dynamic configurations defined in the project
        manifest metadata. Standardizes tool execution permissions into `tool:{name}:execute`
        and `tool:{name}` formats, and adds Shield secret read scopes for LLM access.

        Args:
            state: The current graph state containing metadata and project manifest.

        Returns:
            A named tuple containing the list of required scopes and a boolean indicating
            if PII masking/anonymization is enabled.
        """
        required_scopes: list[str] = []

        dynamic_secret_ref = None
        dynamic_tool_scopes: list[str] = []
        dynamic_pii_masking = None
        actual_requires_llm = requires_llm
        project_config = metadata_project_config(state)

        # Resolve node-level overrides from the structured manifest
        node_cfg = get_node_config(project_config, node_name)
        if node_cfg:
            dynamic_secret_ref = node_cfg.get("model_secret_ref")
            dynamic_pii_masking = node_cfg.get("pii_masking")
            if "type" in node_cfg:
                actual_requires_llm = node_cfg.get("type") in ("llm", "agent")

        # Permissions stay canonical: tool:name:mode — never federated_tool: in permissions.
        # The federated distinction is a provenance/observability concern only.
        graph_runtime = get_graph_runtime_config(project_config)
        node_tool_bindings_raw = graph_runtime.get("node_tool_bindings", {})
        bindings_raw = (
            node_tool_bindings_raw.get(node_name, {})
            if is_json_dict(node_tool_bindings_raw)
            else {}
        )
        node_tool_bindings: JsonDict = bindings_raw if is_json_dict(bindings_raw) else {}
        if is_json_dict(node_tool_bindings):
            for tool_name, mode in node_tool_bindings.items():
                if not isinstance(mode, str) or mode not in _VALID_TOOL_BINDING_MODES:
                    raise SecurityError(
                        f"Invalid node_tool_bindings mode for tool '{tool_name}': {mode!r}"
                    )
                dynamic_tool_scopes.append(f"tool:{tool_name}:{mode}")

        actual_secret_ref = dynamic_secret_ref or model_secret_ref

        # Manifest overrides hardcoded default if provided
        if dynamic_pii_masking is not None:
            actual_pii_masking = bool(dynamic_pii_masking)
        else:
            actual_pii_masking = pii_masking

        # Add execution tools explicitly from kwargs if present (used in built-in graphs)
        if execute_tools:
            for tool in execute_tools:
                scope = f"tool:{tool}:execute"
                if scope not in dynamic_tool_scopes:
                    dynamic_tool_scopes.append(scope)

        if service_scopes:
            for scope in service_scopes:
                if scope not in required_scopes:
                    required_scopes.append(scope)

        required_scopes.extend(dynamic_tool_scopes)
        # Grant bare tool scope — SecureTool._enforce_permission expects tool:{name}
        for scope in dynamic_tool_scopes:
            parts = scope.split(":")
            if len(parts) >= 3:
                bare_scope = f"tool:{parts[1]}"
                if bare_scope not in required_scopes:
                    required_scopes.append(bare_scope)

        token = get_current_access_token()
        if token is None:
            raise SecurityError(f"Node '{node_name}' requires a tenant-scoped token")

        effective_tenants = resolve_node_effective_tenants(
            state,
            node_name,
            token=token,
            node_spec=spec,
        )

        if actual_requires_llm:
            if actual_secret_ref:
                # Per-node: Shield path = {tenant}/api_keys/{node_name}/model_secret_ref
                shield_suffix = f"{node_name}/model_secret_ref"
                required_scopes.append("shield:secrets:read")
                for tenant_id in effective_tenants:
                    required_scopes.append(
                        f"shield:secrets:read:{tenant_id}/api_keys/{shield_suffix}"
                    )
            else:
                # No per-node ref — but node needs LLM. Uses default/fallback model from policy.
                policy_raw = project_config.get("policy", {}) or {}
                policy = policy_raw if is_json_dict(policy_raw) else {}
                models_raw = policy.get("models", {}) or {}
                models_policy = models_raw if is_json_dict(models_raw) else {}
                llm_raw = models_policy.get("llm", {}) or {}
                llm_policy = llm_raw if is_json_dict(llm_raw) else {}
                default_model = llm_policy.get("default")
                fallback_raw = llm_policy.get("fallback")
                for tenant_id in effective_tenants:
                    if isinstance(default_model, str) and default_model:
                        required_scopes.append("shield:secrets:read")
                        required_scopes.append(
                            f"shield:secrets:read:{tenant_id}/api_keys/{default_model}"
                        )
                    if isinstance(fallback_raw, list):
                        for model in fallback_raw:
                            if isinstance(model, str):
                                required_scopes.append(
                                    f"shield:secrets:read:{tenant_id}/api_keys/{model}"
                                )

        if actual_pii_masking:
            required_scopes.extend(
                [
                    "privacy:anonymize",
                    "privacy:deanonymize",
                    "privacy:check_pii",
                ]
            )

        return _ResolvedSecurity(required_scopes, actual_pii_masking, effective_tenants)

    def _check_prompt_integrity(state: GraphState) -> None:
        """Verify that the node's prompt content matches its registered signature.

        Computes an HMAC of the prompt text using the tenant/project secret and compares
        it against the expected signature. Prevents prompt injection or tampering at the
        infrastructure level.

        Args:
            state: The current graph state containing the prompt text and signature.

        Raises:
            TamperDetectedError: If the computed signature does not match the expected signature.
        """
        project_config = metadata_project_config(state)
        metadata: ExecutionMetadata = state.get("metadata", {})

        # Resolve the actual prompt text first (same key convention as the graph
        # builder). A node that ships no LLM prompt has nothing to verify.
        graph_runtime = get_graph_runtime_config(project_config)
        prompt_text = graph_runtime.get(f"{node_name}_prompt")
        if not isinstance(prompt_text, str) or not prompt_text:
            return

        # Dynamic resolution: prefer manifest config over build-time parameter.
        actual_signature = get_node_attr(
            project_config, node_name, "prompt_signature", prompt_signature
        )
        # Fail closed (WS-9): a node that ships an LLM prompt MUST carry a
        # signature. Treat a missing/stripped signature on a prompted node as
        # tampering — never silently skip — so an attacker cannot bypass
        # integrity verification simply by removing the signature from config.
        if not actual_signature:
            from contextunity.core.exceptions import TamperDetectedError

            raise TamperDetectedError(
                message=(
                    f"Prompt integrity check failed on node '{node_name}': the node "
                    "defines an LLM prompt but carries no integrity signature. A "
                    "stripped or missing signature is treated as tampering (fail-closed)."
                ),
                node_name=node_name,
            )

        project_id_raw = project_config.get("project_id")
        project_id = (
            project_id_raw
            if isinstance(project_id_raw, str)
            else str(metadata.get("tenant_id", ""))
        )

        from contextunity.core.discovery import get_project_key

        key_data = get_project_key(project_id) or {}
        project_secret = key_data.get("project_secret")

        if not isinstance(project_secret, str) or not project_secret:
            from contextunity.core.exceptions import TamperDetectedError

            raise TamperDetectedError(
                message=(
                    f"Prompt integrity cannot be verified on node '{node_name}': "
                    "project-scoped HMAC key material is unavailable."
                ),
                node_name=node_name,
            )

        from contextunity.core.sdk.prompt_integrity import verify_prompt
        from contextunity.core.signing import HmacBackend

        backend = HmacBackend(project_id, project_secret)
        if not verify_prompt(str(prompt_text), str(actual_signature), backend):
            from contextunity.core.exceptions import TamperDetectedError

            raise TamperDetectedError(
                message=(
                    f"Prompt integrity violation on node '{node_name}': "
                    "signature does not match stored prompt content. "
                    "Possible infrastructure-level prompt injection."
                ),
                node_name=node_name,
            )

    def _prepare_execution(
        state: GraphState,
    ) -> tuple[Token[ContextToken | None], GraphState, bool]:
        """Perform pre-execution validation, provenance tracking, and context setup.

        Resolves node permissions, attenuates the context token, appends node execution and
        secret-read actions to the token's provenance log, verifies prompt integrity,
        dispatches a telemetry start event, and injects the attenuated token into the state.

        Args:
            state: The current graph state.

        Returns:
            A tuple of (previous_context_token_ref, modified_state_with_attenuated_token, pii_masking_enabled).
        """
        required_scopes, actual_pii_masking, effective_tenants = _resolve_scopes_and_pii(state)
        token_ref, attenuated = execute_with_token(state, required_scopes, effective_tenants)

        from contextunity.router.core.context import append_provenance

        # Record node execution step (flat string, not full token tuple)
        append_provenance(f"node:{node_name}")

        # Record infrastructural actions: Shield secret fetches and PII masking.
        # Service provenance is determined by the PROJECT manifest, not the platform.
        metadata = state.get("metadata", {})
        project_cfg_raw = metadata.get("project_config", {})
        project_cfg = dict(project_cfg_raw) if is_json_dict(project_cfg_raw) else {}
        services_raw = project_cfg.get("services", {})
        services = services_raw if is_json_dict(services_raw) else {}
        shield_raw = services.get("shield")
        shield_on = bool(is_json_dict(shield_raw) and shield_raw.get("enabled"))
        for req in required_scopes:
            if req.startswith("shield:secrets:read:") and req.count(":") >= 3:
                if shield_on:
                    append_provenance(req)
            elif req == "privacy:anonymize":
                append_provenance("privacy:pii_applied")

        # Prompt integrity: verify signature before execution
        _check_prompt_integrity(state)

        try:
            dispatch_custom_event(
                "brain_event", {"event": BrainEvent(type="node_start", node=node_name)}
            )
        except RuntimeError:
            pass

        effective_state = state.copy()
        effective_state["__token__"] = attenuated

        return token_ref, effective_state, actual_pii_masking

    def _finalize_execution(token_ref: Token[ContextToken | None]) -> None:
        """Perform post-execution cleanup, telemetry dispatch, and context restoration.

        Dispatches a telemetry completion event for the node and restores the previous
        unattenuated security context token in the contextvar.

        Args:
            token_ref: The contextvar token reference returned during context setup.
        """
        try:
            dispatch_custom_event(
                "brain_event", {"event": BrainEvent(type="node_end", node=node_name)}
            )
        except RuntimeError:
            pass
        reset_current_access_token(token_ref)

    @functools.wraps(node_func)
    async def secure_wrapper(
        state: GraphState,
        config: RunnableConfig,
    ) -> StateUpdate:
        """Execute the wrapped node within an isolated, capability-stripped context.

        Args:
            state: The incoming graph state.
            config: Execution configuration dict for LangChain/LangGraph.

        Returns:
            The update dictionary or object returned by the wrapped node function.
        """
        token_ref, effective_state, actual_pii = _prepare_execution(state)

        try:
            # ── Execute inner node ────────────────────────────────────
            raw_result = node_func(effective_state, config)
            result: StateUpdate = (
                await raw_result if inspect.isawaitable(raw_result) else raw_result
            )

            # ── PII bypass enforcement ────────────────────────────────
            if actual_pii:
                leaked_keys = (
                    set(result.keys())
                    - STATE_TOP_LEVEL_KEYS
                    - {
                        "dynamic",
                        "_token_usage",
                        "_start_ts",
                        "_raw_output",
                        "components",
                        "structured_output",
                        "__token__",
                    }
                )
                if leaked_keys:
                    logger.warning(
                        (
                            "PII_BYPASS Node '%s': keys %s written to root state, "
                            "bypassing PII deanonymization. Use state_routing."
                            "write_state_output() to write via dynamic bucket."
                        ),
                        node_name,
                        leaked_keys,
                    )

            return result
        finally:
            _finalize_execution(token_ref)

    return secure_wrapper


__all__ = ["make_secure_node"]
