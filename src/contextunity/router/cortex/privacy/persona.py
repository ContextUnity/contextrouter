"""contextunity.router.cortex.privacy Persona Engine — synthetic identity injection.
Personas provide a layer of abstraction between the user and the LLM.
They inject system prompts, behavioral guidelines, and synthetic identity
attributes into the conversation context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contextunity.core import get_contextunit_logger

logger = get_contextunit_logger(__name__)

__all__ = [
    "Persona",
    "PersonaEngine",
    "PersonaTemplate",
]


@dataclass(frozen=True)
class PersonaTemplate:
    """A reusable persona template.

    Attributes:
        name: Unique template identifier.
        display_name: Human-readable name.
        system_prompt: System prompt to inject.
        traits: Key-value traits (e.g. tone="professional").
        restrictions: Things this persona must NOT do.
        language: Preferred language code (e.g. "uk", "en").
    """

    name: str
    display_name: str = ""
    system_prompt: str = ""
    traits: dict[str, str] = field(default_factory=dict)
    restrictions: list[str] = field(default_factory=list)
    language: str = "en"


@dataclass
class Persona:
    """An active persona instance — a template bound to a session.

    Attributes:
        template: The underlying persona template.
        session_id: Session this persona is bound to.
        custom_traits: Session-specific trait overrides.
    """

    template: PersonaTemplate
    session_id: str = ""
    custom_traits: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Delegate to the underlying template's unique identifier."""
        return self.template.name

    @property
    def system_prompt(self) -> str:
        """Delegate to the underlying template's system prompt."""
        return self.template.system_prompt

    def get_trait(self, key: str, default: str = "") -> str:
        """Look up a trait value, preferring session overrides.

        Resolution order: ``custom_traits[key]`` → ``template.traits[key]``
        → ``default``.

        Args:
            key: The trait name to look up (e.g. ``"tone"``).
            default: Fallback value if the trait is not defined anywhere.

        Returns:
            The resolved trait value.
        """
        return self.custom_traits.get(key, self.template.traits.get(key, default))

    def inject_into_prompt(self, user_prompt: str) -> str:
        """Prepend the persona's system prompt to a user message.

        If the template has no system prompt, the user prompt is returned
        unchanged.

        Args:
            user_prompt: The raw user message to enrich.

        Returns:
            The user prompt preceded by the system prompt (separated by
            a double newline), or the original prompt if no system prompt
            is configured.
        """
        if not self.template.system_prompt:
            return user_prompt
        return f"{self.template.system_prompt}\n\n{user_prompt}"


# Built-in persona templates
BUILTIN_TEMPLATES: dict[str, PersonaTemplate] = {
    "contextunity-harmless-agent": PersonaTemplate(
        name="contextunity-harmless-agent",
        display_name="ContextUnity Harmless Agent",
        system_prompt=(
            "You are ContextUnity Harmless Agent. You are helpful, accurate, and safe. "
            "You never produce harmful, misleading, or biased content. "
            "You respect user privacy and handle sensitive data responsibly. "
            "When uncertain, you acknowledge limitations rather than speculate."
        ),
        traits={"tone": "neutral", "verbosity": "concise", "safety": "strict"},
        restrictions=[
            "Do not produce harmful content",
            "Do not reveal PII from training data",
            "Do not generate misleading information",
        ],
    ),
    "neutral": PersonaTemplate(
        name="neutral",
        display_name="Neutral Assistant",
        system_prompt="You are a helpful, accurate, and concise assistant.",
        traits={"tone": "neutral", "verbosity": "concise"},
    ),
    "professional": PersonaTemplate(
        name="professional",
        display_name="Professional Advisor",
        system_prompt=(
            "You are a professional advisor. Provide clear, actionable, and well-structured advice. "
            "Use formal language. Cite sources when available."
        ),
        traits={"tone": "formal", "verbosity": "detailed"},
        restrictions=["Do not use slang", "Do not speculate without evidence"],
    ),
    "creative": PersonaTemplate(
        name="creative",
        display_name="Creative Writer",
        system_prompt=(
            "You are a creative writing assistant. Help with brainstorming, storytelling, "
            "and creative expression. Be imaginative and encouraging."
        ),
        traits={"tone": "warm", "verbosity": "moderate"},
    ),
}

DEFAULT_PERSONA = "contextunity-harmless-agent"


class PersonaEngine:
    """Manage persona templates and create active persona instances.

    Usage:
        engine = PersonaEngine()
        engine.register_template(my_custom_persona)
        persona = engine.create_persona("professional", session_id="abc123")
        enriched_prompt = persona.inject_into_prompt("What is the best approach?")
    """

    def __init__(self) -> None:
        """Create a persona engine pre-loaded with built-in templates.

        The engine starts with the ``BUILTIN_TEMPLATES`` registry
        (harmless-agent, neutral, professional, creative) and an empty
        active persona map.
        """
        self._templates: dict[str, PersonaTemplate] = dict(BUILTIN_TEMPLATES)
        self._active_personas: dict[str, Persona] = {}  # session_id → Persona

    def register_template(self, template: PersonaTemplate) -> None:
        """Register a custom persona template, making it available for activation.

        Overwrites any existing template with the same ``name``.

        Args:
            template: The persona template to register.
        """
        self._templates[template.name] = template
        logger.info("Registered persona template: %s", template.name)

    def get_template(self, name: str) -> PersonaTemplate | None:
        """Retrieve a template by its unique name.

        Args:
            name: The template identifier (e.g. ``"professional"``).

        Returns:
            The matching template, or ``None`` if not registered.
        """
        return self._templates.get(name)

    def list_templates(self) -> list[str]:
        """List all registered template names in sorted order.

        Returns:
            Sorted list of template identifiers.
        """
        return sorted(self._templates.keys())

    def create_persona(
        self,
        template_name: str,
        session_id: str = "",
        custom_traits: dict[str, str] | None = None,
    ) -> Persona:
        """Create an active persona from a template.

        Args:
            template_name: Name of the template to instantiate.
            session_id: Session to bind this persona to.
            custom_traits: Optional trait overrides for this session.

        Returns:
            Active Persona instance.

        Raises:
            KeyError: If template_name is not registered.
        """
        template = self._templates.get(template_name)
        if template is None:
            available = ", ".join(sorted(self._templates.keys()))
            raise KeyError(f"Persona template '{template_name}' not found. Available: {available}")

        persona = Persona(
            template=template,
            session_id=session_id,
            custom_traits=custom_traits or {},
        )

        if session_id:
            self._active_personas[session_id] = persona
            logger.debug("Activated persona '%s' for session '%s'", template_name, session_id)

        return persona

    def get_active_persona(self, session_id: str) -> Persona | None:
        """Retrieve the currently active persona for a session.

        Args:
            session_id: The session identifier to look up.

        Returns:
            The active persona, or ``None`` if no persona is bound.
        """
        return self._active_personas.get(session_id)

    def deactivate_persona(self, session_id: str) -> None:
        """Remove the active persona binding for a session.

        No-op if the session has no active persona.

        Args:
            session_id: The session to unbind.
        """
        removed = self._active_personas.pop(session_id, None)
        _ = removed

    def switch_persona(
        self,
        session_id: str,
        new_template_name: str,
        custom_traits: dict[str, str] | None = None,
    ) -> Persona:
        """Switch the active persona for a session.

        Args:
            session_id: Session to switch.
            new_template_name: New template to activate.
            custom_traits: Optional trait overrides.

        Returns:
            New active Persona instance.
        """
        _ = self.deactivate_persona(session_id)
        return self.create_persona(new_template_name, session_id, custom_traits)
