"""Sub-agent orchestration -- spawn, manage, and aggregate results from child agent instances."""

from .orchestrator import SubAgentOrchestrator
from .prompt_generator import SubAgentPromptGenerator
from .spawner import SubAgentSpawner

__all__ = ["SubAgentSpawner", "SubAgentOrchestrator", "SubAgentPromptGenerator"]
