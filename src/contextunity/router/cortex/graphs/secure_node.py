"""Generic secure node wrapper for graph execution capability stripping."""

import functools
import inspect
from typing import Any, Callable, TypeVar

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import SecurityError
from contextunity.core.tokens import TokenBuilder

from contextunity.router.cortex.runtime_context import (
    get_current_access_token,
    reset_current_access_token,
    set_current_access_token,
)

logger = get_contextunit_logger(__name__)

StateT = TypeVar("StateT")
ReturnT = TypeVar("ReturnT")


def make_secure_node(
    node_name: str,
    node_func: Callable[[StateT], ReturnT],
    *,
    requires_llm: bool = True,
    pii_masking: bool = False,
    model_secret_ref: str | None = None,
    prompt_signature: str | None = None,
    schema_tools: list[str] | None = None,
    execute_tools: list[str] | None = None,
) -> Callable[[StateT], ReturnT]:
    """Wrap a LangGraph node to strictly execute with narrowed permissions (Capability Stripping).

    This calculates the exact required scopes statically, forcing the token identity to the node,
    and dropping any permissions that were granted broadly to the parent graph runner.

    Args:
        node_name: Name of the node in the graph (e.g. 'planner').
        node_func: The actual node executable function.
        requires_llm: Whether this node invokes LLMs. If False, skips granting shield:secrets:read.
        pii_masking: If true, node gets 'zero:anonymize' and 'zero:deanonymize' permissions.
        model_secret_ref: Secret reference. Will grant 'shield:secrets:read:<tenant>/api_keys/<ref>'.
        prompt_signature: Serialized HMAC signature for prompt tamper-proofing.
        schema_tools: List of tool names bound for schema reading only. No execution permissions granted.
        execute_tools: List of tool names bound for execution. Will grant 'tool:<name>'.
    """

    def execute_with_token(state: StateT, required_scopes: list[str]) -> tuple[Any, Any]:
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

        logger.debug("Node '%s' capability stripping: permissions=%s", node_name, required_scopes)

        attenuated_token = TokenBuilder().attenuate(
            token, permissions=required_scopes, agent_id=f"node:{node_name}"
        )

        return set_current_access_token(attenuated_token), attenuated_token

    def _resolve_scopes(state: StateT) -> list[str]:
        required_scopes = []

        dynamic_secret_ref = None
        dynamic_tool_scopes: list[str] = []
        dynamic_pii_masking = None
        actual_requires_llm = requires_llm

        if isinstance(state, dict):
            metadata = state.get("metadata", {})
            project_config = metadata.get("project_config", {})

            # Resolve node-level overrides from the structured manifest
            from contextunity.router.cortex.graphs.config_resolution import get_node_config

            node_cfg = get_node_config(project_config, node_name)
            if node_cfg:
                dynamic_secret_ref = node_cfg.get("model_secret_ref")
                dynamic_pii_masking = node_cfg.get("pii_masking")
                if "type" in node_cfg:
                    actual_requires_llm = node_cfg.get("type") == "llm"

            # Permissions stay canonical: tool:name:mode — never federated_tool: in permissions.
            # The federated distinction is a provenance/observability concern only.
            node_tool_bindings = project_config.get("node_tool_bindings", {}).get(node_name, {})
            for tool_name, mode in node_tool_bindings.items():
                dynamic_tool_scopes.append(f"tool:{tool_name}:{mode}")

        actual_secret_ref = dynamic_secret_ref or model_secret_ref

        # Manifest overrides hardcoded default if provided
        if dynamic_pii_masking is not None:
            actual_pii_masking = dynamic_pii_masking
        else:
            actual_pii_masking = pii_masking

        # Add execution tools explicitly from kwargs if present (used in built-in graphs)
        if execute_tools:
            for t in execute_tools:
                scope = f"tool:{t}:execute"
                if scope not in dynamic_tool_scopes:
                    dynamic_tool_scopes.append(scope)

        required_scopes.extend(dynamic_tool_scopes)
        # Grant bare tool scope — SecureTool._enforce_permission expects tool:{name}
        for scope in dynamic_tool_scopes:
            parts = scope.split(":")
            if len(parts) >= 3:
                bare_scope = f"tool:{parts[1]}"
                if bare_scope not in required_scopes:
                    required_scopes.append(bare_scope)

        token = get_current_access_token()
        tenant_id = token.allowed_tenants[0] if token and token.allowed_tenants else "default"

        if actual_requires_llm:
            if actual_secret_ref:
                # Per-node: Shield path = {tenant}/api_keys/{node_name}/model_secret_ref
                shield_suffix = f"{node_name}/model_secret_ref"
                required_scopes.append("shield:secrets:read")
                required_scopes.append(f"shield:secrets:read:{tenant_id}/api_keys/{shield_suffix}")
            else:
                # No per-node ref — but node needs LLM. Uses default/fallback model from policy.
                # Grant access to policy-level model paths.
                policy = project_config.get("policy", {}) or {}
                ai_policy = policy.get("ai_model_policy", {}) or {}
                default_model = ai_policy.get("default_ai_model")
                if default_model:
                    required_scopes.append("shield:secrets:read")
                    required_scopes.append(
                        f"shield:secrets:read:{tenant_id}/api_keys/{default_model}"
                    )
                for model in ai_policy.get("fallback_ai_models") or []:
                    required_scopes.append(f"shield:secrets:read:{tenant_id}/api_keys/{model}")

        if actual_pii_masking:
            required_scopes.extend(
                [
                    "zero:anonymize",
                    "zero:deanonymize",
                    "zero:check_pii",
                ]
            )

        return required_scopes

    def _check_prompt_integrity(state: StateT) -> None:
        """Verify prompt has not been modified since manifest registration."""
        if not isinstance(state, dict):
            return

        metadata = state.get("metadata", {})
        project_config = metadata.get("project_config", {})
        from contextunity.router.cortex.graphs.config_resolution import get_node_attr

        # Dynamic resolution: prefer manifest config over build-time parameter
        actual_signature = get_node_attr(
            project_config, node_name, "prompt_signature", prompt_signature
        )
        if not actual_signature:
            return

        # Resolve the actual prompt text (same key convention as graph builder)
        prompt_text = project_config.get(f"{node_name}_prompt")
        if not prompt_text:
            return

        project_id = project_config.get("project_id") or metadata.get("tenant_id", "")

        from contextunity.core.config import get_core_config

        project_secret = get_core_config().security.project_secret

        if not project_secret:
            logger.debug("No project_secret for integrity check on node '%s' — skipping", node_name)
            return

        from contextunity.core.sdk.prompt_integrity import verify_prompt
        from contextunity.core.signing import HmacBackend

        backend = HmacBackend(project_id, project_secret)
        if not verify_prompt(prompt_text, actual_signature, backend):
            from contextunity.core.exceptions import TamperDetectedError

            raise TamperDetectedError(
                message=f"Prompt integrity violation on node '{node_name}': "
                f"signature does not match stored prompt content. "
                f"Possible infrastructure-level prompt injection.",
                node_name=node_name,
            )

    if inspect.iscoroutinefunction(node_func):

        @functools.wraps(node_func)
        async def a_secure_node_wrapper(state: StateT, *args: Any, **kwargs: Any) -> ReturnT:
            required_scopes = _resolve_scopes(state)
            token_ref, attenuated = execute_with_token(state, required_scopes)

            from contextunity.router.cortex.runtime_context import append_provenance

            # Record node execution step (flat string, not full token tuple)
            append_provenance(f"node:{node_name}")

            # Only record infrastructural ACTIONS actually taken (fetching a specific LLM secret).
            # We do NOT record zero: scopes here, because the provenance will naturally
            # capture the internal zero_tool:anonymize_text:execute when the node actually runs it.
            for req in required_scopes:
                if req.startswith("shield:secrets:read:") and req.count(":") >= 3:
                    # Log the specific secret path accessed
                    append_provenance(req)

            # Prompt integrity: verify signature before execution
            _check_prompt_integrity(state)

            try:
                return await node_func(state, *args, **kwargs)
            finally:
                reset_current_access_token(token_ref)

        return a_secure_node_wrapper

    @functools.wraps(node_func)
    def secure_node_wrapper(state: StateT, *args: Any, **kwargs: Any) -> ReturnT:
        required_scopes = _resolve_scopes(state)
        token_ref, attenuated = execute_with_token(state, required_scopes)

        from contextunity.router.cortex.runtime_context import append_provenance

        # Record node execution step (flat string, not full token tuple)
        append_provenance(f"node:{node_name}")

        # Only record infrastructural ACTIONS actually taken (fetching a specific LLM secret).
        # We do NOT record zero: scopes here, because the provenance will naturally
        # capture the internal zero_tool:anonymize_text:execute when the node actually runs it.
        for req in required_scopes:
            if req.startswith("shield:secrets:read:") and req.count(":") >= 3:
                # Log the specific secret path accessed
                append_provenance(req)

        # Prompt integrity: verify signature before execution
        _check_prompt_integrity(state)

        try:
            return node_func(state, *args, **kwargs)
        finally:
            reset_current_access_token(token_ref)

    return secure_node_wrapper


__all__ = ["make_secure_node"]
