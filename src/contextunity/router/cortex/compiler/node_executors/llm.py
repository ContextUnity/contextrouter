"""LLM Node Executor for the Graph Compiler.
Creates a LangGraph node function that invokes model_registry.create_llm()
for real LLM calls. Reads/writes from state keys specified in manifest config.
Persona resolution (node > global > default):
    1. node_spec["persona"] — explicit per-node persona template name
    2. manifest_config["persona"] — project-wide default
    3. DEFAULT_PERSONA ("contextunity-harmless-agent")
"""

from __future__ import annotations

from typing import Literal

from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_dumps
from contextunity.core.types import is_object_dict, is_object_list
from langchain_core.callbacks import dispatch_custom_event

from contextunity.router.core.exceptions import RouterLLMError
from contextunity.router.modules.models.types import ModelError

from ...compiler.node_config import NodeConfig
from ...compiler.state_routing import read_state_input
from ...compiler.types import CompilerNodeSpec, ProjectManifest
from ...events import BrainEvent, BrainEventPayload, BrainEventType
from ...types import GraphState, NodeFunc, StateUpdate

logger = get_contextunit_logger(__name__)


def _string_object_dict(value: object) -> dict[str, object]:
    """Return a string-keyed object dict when possible."""
    if not is_object_dict(value):
        return {}
    return {str(key): item for key, item in value.items()}


def _system_for_model_request(system_prompt: str | None) -> str | None:
    """Sanitize a system prompt for ``ModelRequest.system``.

    Strips whitespace and converts empty / whitespace-only strings
    to ``None`` — providers reject blank system prompts.

    Args:
        system_prompt: Raw prompt text or ``None``.

    Returns:
        Stripped non-empty string, or ``None``.
    """
    if system_prompt is None:
        return None
    stripped = system_prompt.strip()
    return stripped or None


def _emit_brain_event(event_type: BrainEventType, node_name: str, data: BrainEventPayload) -> None:
    """Dispatch a ``BrainEvent`` via LangChain custom event callback.

    Silently swallows ``RuntimeError`` when no callback manager is
    active (e.g., during unit tests or standalone execution).

    Args:
        event_type: Semantic event tag (``llm_error``, ``llm_start``, etc.).
        node_name: Logical node identifier for trace grouping.
        data: Event payload dict matching ``BrainEventPayload``.
    """
    try:
        dispatch_custom_event(
            "brain_event",
            {"event": BrainEvent(type=event_type, node=node_name, data=data)},
        )
    except RuntimeError:
        pass


def _get_model_registry():
    """Lazy import to avoid circular dependencies at module level."""
    from contextunity.router.modules.models.registry.main import model_registry

    return model_registry


def make_llm_node(node_spec: CompilerNodeSpec, manifest_config: ProjectManifest) -> NodeFunc:
    """Create a LangGraph node for LLM invocation with persona and privacy.

    Mode dispatch:
    - ``"sql_visualizer"`` — delegates to the SQL visualization sub-graph.
    - ``"parallel"`` — reserved (raises ``NotImplementedError``).
    - ``None`` / absent — standard single-call LLM with persona resolution.

    Persona resolution cascade:
    ``node_spec["persona"]`` → ``node_config.persona`` →
    ``manifest_config["persona"]`` → ``DEFAULT_PERSONA``.

    The returned closure builds a ``ModelRequest``, invokes the LLM
    through ``generate_with_node_privacy`` (which handles PII masking
    and callback tracing), and routes output to the configured state key.

    Args:
        node_spec: Compiled node specification (model, mode, persona, config).
        manifest_config: Full project manifest for prompt and config lookup.

    Returns:
        Async ``NodeFunc`` closure for LangGraph registration.

    Raises:
        RouterLLMError: If the LLM cannot be created or generation fails.
        NotImplementedError: If ``mode='parallel'`` is requested.
    """
    node_name = node_spec.get("name", "unnamed_llm")
    model_name = node_spec.get("model", "")
    model_secret_ref = node_spec.get("model_secret_ref")
    _raw_cfg = node_spec.get("config")
    _cfg_dict = _string_object_dict(_raw_cfg)
    nc = NodeConfig.model_validate(_cfg_dict)

    # Mode dispatch: explicit execution mode set at node level.
    #   "sql_visualizer" — SQL-specific visualization (report + table + chart components)
    #   "parallel"       — generic parallel sub-prompt execution (stub, not yet implemented)
    #   None / absent    — standard single-call LLM
    mode = node_spec.get("mode")

    if mode == "sql_visualizer":
        from contextunity.router.cortex.compiler.node_executors.llm_invocation import (
            invoke_messages_with_node_privacy,
        )
        from contextunity.router.cortex.compiler.platform_tools.sql_visualizer import (
            make_visualizer_node,
        )

        inner_config = manifest_config.get("config", manifest_config)
        prompt = inner_config.get(f"{node_name}_prompt")
        sub_raw = inner_config.get(f"{node_name}_sub_prompts")
        visualizer_sub: dict[str, str] | None = None
        if is_object_dict(sub_raw):
            visualizer_sub = {
                str(key): str(value) for key, value in sub_raw.items() if isinstance(value, str)
            }
        return make_visualizer_node(
            node_name=node_name,
            visualizer_prompt=str(prompt) if prompt else None,
            visualizer_sub_prompts=visualizer_sub,
            default_model_key=model_name or None,
            shield_key_name=f"{node_name}/model_secret_ref" if model_secret_ref else None,
            invoke_model_fn=invoke_messages_with_node_privacy,
        )

    if mode == "parallel":
        raise NotImplementedError(
            (
                f"Node '{node_name}': mode='parallel' is reserved for future "
                "generic parallel sub-prompt execution. Use 'sql_visualizer' for "
                "SQL analytics visualization."
            )
        )

    state_input_key = nc.state_input_key or "messages"
    state_output_key = nc.state_output_key or "final_output"

    # Resolve per-node retry policy (3rd tier: node_config overrides global/per-type).
    node_retry_policy = None
    retry_raw = nc.retry
    if is_object_dict(retry_raw):
        from contextunity.core.manifest.router import RetryPolicy

        try:
            node_retry_policy = RetryPolicy.model_validate(retry_raw)
        except Exception as exc:  # graceful-degrade: LLM errors logged and returned to graph
            logger.warning(
                "Node '%s': invalid retry config %r, ignoring: %s", node_name, retry_raw, exc
            )

    # Pre-compute Shield key path if model_secret_ref is declared.
    # Matches SecureNode scope pattern: {node_name}/model_secret_ref
    _shield_key_name = f"{node_name}/model_secret_ref" if model_secret_ref else None

    # Resolve static prompt_version from manifest for this node.
    _node_prompt_version: str | None = nc.prompt_version

    from langchain_core.runnables.config import RunnableConfig

    async def llm_executor(state: GraphState, config: RunnableConfig) -> StateUpdate:
        """Execute a single LLM call with persona, privacy, and tracing.

        Reads input from ``state_input_key``, resolves persona and system
        prompt, builds a ``ModelRequest``, invokes the model through the
        privacy-aware generation path, and writes output to
        ``state_output_key`` and ``intermediate_results``.

        Args:
            state: Graph execution state (must include ``__token__``).
            config: LangGraph runnable config with callback handlers.

        Returns:
            State update with generated content and ``_last_node`` marker.

        Raises:
            RouterLLMError: On model creation failure or generation error.
        """
        logger.debug("Executing LLM Node: %s (model: %s)", node_name, model_name)

        # Resolve tenant_id from ContextToken (SPOT: Token is Single Point of Truth).
        # SecureNode has already attenuated the token with exact Shield scopes,
        # so create_llm can auto-infer shield_key_name from token.permissions.
        ctx_token = state.get("__token__")
        if ctx_token is None:
            raise RouterLLMError(
                message="Context token missing in state",
                node_name=node_name,
                model=model_name,
            )
        tenant_id = ""
        if ctx_token.allowed_tenants:
            tenant_id = ctx_token.allowed_tenants[0]

        logger.debug(
            "LLM Node '%s': tenant_id=%s",
            node_name,
            tenant_id,
        )

        try:
            registry = _get_model_registry()
            create_kwargs: dict[str, object] = {}
            if tenant_id:
                create_kwargs["tenant_id"] = tenant_id
            from contextunity.router.cortex.config_resolution import metadata_project_id

            project_id = metadata_project_id(state)
            if project_id:
                create_kwargs["project_id"] = project_id

            # shield_key_name is auto-inferred from attenuated token permissions
            # in create_llm (lines 80-98 of registry/main.py)
            llm = registry.create_llm(model_name, config=None, **create_kwargs)
        except Exception as exc:  # graceful-degrade: LLM errors logged and returned to graph
            logger.error_exc("Node '%s' failed to create LLM '%s'", node_name, model_name)
            raise RouterLLMError(
                message=f"Node '{node_name}' failed to create LLM '{model_name}': {exc}",
                node_name=node_name,
                model=model_name,
            ) from exc

        # ── Persona resolution: node > global > default ─────────────
        # Persona provides the base system prompt for every LLM call.
        # prompt_ref extends it (appended), not replaces.
        from contextunity.router.cortex.privacy.persona import (
            DEFAULT_PERSONA,
            PersonaEngine,
        )

        _persona_engine = PersonaEngine()
        persona_name = (
            node_spec.get("persona")
            or nc.persona
            or manifest_config.get("persona")
            or DEFAULT_PERSONA
        )
        persona_template = _persona_engine.get_template(persona_name)
        persona_system_prompt = persona_template.system_prompt if persona_template else ""

        # Node-level system_prompt (from config) overrides persona completely
        explicit_system_prompt = nc.system_prompt
        system_prompt: str | None
        if explicit_system_prompt:
            system_prompt = str(explicit_system_prompt)
        elif persona_system_prompt:
            system_prompt = persona_system_prompt
        else:
            system_prompt = None

        # prompt_ref extends the base (persona or explicit), not replaces
        if node_spec.get("prompt_ref"):
            inner_config = manifest_config.get("config", manifest_config)
            ref_prompt = inner_config.get(f"{node_name}_prompt")
            if ref_prompt is not None:
                ref_text = str(ref_prompt).strip()
                if ref_text:
                    system_prompt = f"{system_prompt}\n\n{ref_text}" if system_prompt else ref_text

        # Read input from state (resolves dynamic → top-level → default)
        input_data_obj: object = read_state_input(state, state_input_key, default=[])

        # Build ModelRequest from state messages
        from contextunity.router.modules.models.types import ModelPart, ModelRequest, TextPart

        parts: list[ModelPart] = []

        if is_object_list(input_data_obj):
            # Entry node: input_data is messages list
            for msg in input_data_obj:
                role = str(msg.get("role", "user")) if is_object_dict(msg) else "user"
                content = str(msg.get("content", "")) if is_object_dict(msg) else str(msg)
                if role == "system":
                    text = str(content).strip()
                    if text:
                        system_prompt = f"{system_prompt}\n\n{text}" if system_prompt else text
                else:
                    if content:
                        parts.append(TextPart(text=content))
        elif is_object_dict(input_data_obj):
            # Downstream node: input_data is the direct predecessor's output
            # (final_output by default) or intermediate_results if explicitly
            # configured. User question is always prepended for LLM context.
            input_data = {str(key): value for key, value in input_data_obj.items()}

            # Prepend user's original question for context
            user_messages = state.get("messages", [])
            for msg in user_messages:
                if is_object_dict(msg) and msg.get("role") == "user":
                    user_content = str(msg.get("content", ""))
                    if user_content:
                        parts.append(TextPart(text=f"User question: {user_content}"))
                        break

            if state_input_key == "intermediate_results":
                # Format accumulated pipeline results with labels
                sections: list[str] = []
                for key, value in input_data.items():
                    try:
                        val_str = json_dumps(value, ensure_ascii=False, default=str)
                    except (TypeError, ValueError):
                        val_str = str(value)
                    sections.append(f"--- {key} ---\n{val_str}")
                if sections:
                    parts.append(TextPart(text="\n\n".join(sections)))
            else:
                try:
                    parts.append(
                        TextPart(
                            text=json_dumps(input_data, ensure_ascii=False, default=str),
                        )
                    )
                except (TypeError, ValueError):
                    parts.append(TextPart(text=str(input_data)))
        elif isinstance(input_data_obj, str):
            parts.append(TextPart(text=input_data_obj))
        else:
            parts.append(TextPart(text=str(input_data_obj)))

        if not parts:
            parts.append(TextPart(text=""))

        # PII anonymization is handled by secure_node wrapper — state
        # arrives with messages already anonymized.

        prov: dict[str, object] = dict(nc.provider_config)
        if nc.tool_choice is not None:
            prov["tool_choice"] = nc.tool_choice
        if nc.max_tool_calls is not None:
            prov["max_tool_calls"] = nc.max_tool_calls

        out_fmt = nc.output_format or "text"
        resp_fmt: Literal["text", "json_object"] | None = (
            "json_object" if out_fmt == "json" else None
        )
        if resp_fmt is not None:
            prov["response_format"] = resp_fmt

        model_request = ModelRequest(
            parts=parts,
            system=_system_for_model_request(system_prompt),
            provider_config=prov,
            temperature=nc.temperature,
            max_output_tokens=nc.max_tokens,
            response_format=resp_fmt,
        )

        # ── Traced LLM call ──────────────────────────────────────────
        # All callback tracing (model name, prompt_version, tokens, cost)
        # is handled by model_telemetry() — the single entry point for
        # traced LLM invocations. Nodes never manage callbacks directly.
        # prompt_version is auto-resolved from state's project_config
        # when _node_prompt_version is None.
        from contextunity.router.cortex.compiler.node_executors.llm_invocation import (
            generate_with_node_privacy,
        )

        try:
            response = await generate_with_node_privacy(
                llm,
                model_request,
                config,
                prompt_version=_node_prompt_version,
                node_name=node_name,
                state=state,
                fallback_model_name=model_name,
                retry_policy=node_retry_policy,
            )
        except (ModelError, Exception) as e:
            error_kind = "model_error" if isinstance(e, ModelError) else "unexpected_error"
            _emit_brain_event(
                "llm_error",
                node_name,
                {
                    "model": model_name,
                    "error": error_kind,
                },
            )
            # Build cause chain — ContextUnitFormatter strips exc_info in plain text mode,
            # so we must inline the real underlying error into the message.
            cause_parts = [f"{type(e).__name__}: {e}"]
            cause = e.__cause__
            while cause:
                cause_parts.append(f"  caused by {type(cause).__name__}: {cause}")
                cause = cause.__cause__
            cause_chain = "\n".join(cause_parts)
            logger.error(
                "Node '%s' LLM %s (model=%s):\n%s",
                node_name,
                error_kind,
                model_name,
                cause_chain,
            )
            raise RouterLLMError(
                message=f"Node '{node_name}' LLM {error_kind} ({model_name}): {e}",
                node_name=node_name,
                model=model_name,
            ) from e

        # Extract text from ModelResponse
        content = response.text

        # output_format: "json" → parse LLM output as JSON dict.
        # Data passthrough: merge upstream input with parsed LLM output.
        # This way data flows forward through the pipeline — each node
        # enriches it. Condition routing reads verdict keys (e.g. "valid"),
        # downstream nodes read data keys (e.g. "rows").
        output_format = nc.output_format or "text"
        if output_format == "json":
            from contextunity.router.cortex.compiler.platform_tools.helpers.sql import extract_json

            parsed = extract_json(content)
            if parsed:
                # Merge: upstream data + LLM verdict (LLM keys win on collision).
                # Non-dict upstream (e.g. entry-node message lists) is preserved
                # under ``_upstream_input`` so JSON-only nodes do not drop context.
                if is_object_dict(input_data_obj):
                    merged: dict[str, object] = {**input_data_obj, **parsed}
                else:
                    merged = dict(parsed)
                    if input_data_obj is not None:
                        merged["_upstream_input"] = input_data_obj
                logger.info(
                    "LLM Node '%s': parsed JSON keys=%s -> state['%s']",
                    node_name,
                    list(parsed.keys()),
                    state_output_key,
                )
                return {
                    state_output_key: merged,
                    "intermediate_results": {node_name: parsed},
                    "_last_node": node_name,
                }
            # JSON parse failed — preserve upstream data, pass raw text alongside.
            logger.info(
                "LLM Node '%s': JSON parse failed, passing raw text (%d chars)",
                node_name,
                len(content),
            )
            if is_object_dict(input_data_obj):
                fallback = {**input_data_obj, "_raw_output": content}
                return {
                    state_output_key: fallback,
                    "intermediate_results": {node_name: content},
                    "_last_node": node_name,
                }

        return {
            state_output_key: content,
            "intermediate_results": {node_name: content},
            "_last_node": node_name,
        }

    return llm_executor


__all__ = ["make_llm_node"]
