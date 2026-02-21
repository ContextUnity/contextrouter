"""Sub-Agent Prompt Generator - Generates system prompts using Brain Semantic Memory."""

from __future__ import annotations

import logging
from typing import Any, Dict

from contextcore import BrainClient, ContextUnit

logger = logging.getLogger(__name__)


class SubAgentPromptGenerator:
    """Generate system prompts for sub-agents using Brain Semantic Memory.

    This class queries Brain's Semantic Memory and Procedural Memory to build
    context-aware system prompts for sub-agents based on their task and agent type.
    """

    def __init__(self, brain_endpoint: str = "brain.contextunity.ts.net:50051"):
        """Initialize prompt generator.

        Args:
            brain_endpoint: Brain gRPC endpoint
        """
        from contextrouter.core.brain_token import get_brain_service_token

        self.brain = BrainClient(host=brain_endpoint, mode="grpc", token=get_brain_service_token())

    async def generate_manifest(
        self,
        task: Dict[str, Any],
        agent_type: str,
        tenant_id: str = "default",
        limit: int = 5,
    ) -> str:
        """Generate sub-agent manifest (system prompt) from Brain knowledge.

        Args:
            task: Task description and parameters
            agent_type: Type of agent (task_executor, parallel_worker, etc.)
            tenant_id: Tenant ID for multi-tenant isolation
            limit: Maximum number of knowledge chunks to retrieve

        Returns:
            System prompt/manifest string for the sub-agent
        """
        logger.info("Generating manifest for agent_type=%s, tenant_id=%s", agent_type, tenant_id)

        # Extract task description
        task_description = task.get("description", "")
        task_context = task.get("context", "")

        # Build query from task
        query = f"{task_description} {task_context}".strip()
        if not query:
            query = f"agent type {agent_type} task execution"

        # Query Semantic Memory for relevant knowledge
        semantic_knowledge = await self._query_semantic_memory(
            query=query,
            tenant_id=tenant_id,
            limit=limit,
        )

        # Query Procedural Memory for relevant tools/examples
        procedural_knowledge = await self._query_procedural_memory(
            agent_type=agent_type,
            task_description=task_description,
            tenant_id=tenant_id,
            limit=limit,
        )

        # Build manifest
        manifest = self._build_manifest(
            task=task,
            agent_type=agent_type,
            semantic_knowledge=semantic_knowledge,
            procedural_knowledge=procedural_knowledge,
        )

        logger.debug("Generated manifest (length=%s) for agent_type=%s", len(manifest), agent_type)
        return manifest

    async def _query_semantic_memory(
        self,
        query: str,
        tenant_id: str,
        limit: int = 5,
    ) -> list[Dict[str, Any]]:
        """Query Semantic Memory for relevant knowledge.

        Args:
            query: Search query
            tenant_id: Tenant ID
            limit: Maximum results

        Returns:
            List of knowledge chunks
        """
        try:
            unit = ContextUnit(
                payload={
                    "tenant_id": tenant_id,
                    "query_text": query,
                    "limit": limit,
                    "source_types": ["semantic"],  # Filter for semantic memory
                },
                provenance=["router:subagent:prompt_generator"],
            )

            results = []
            async for response in self.brain.query_memory(unit):
                results.append(
                    {
                        "content": response.payload.get("content", ""),
                        "metadata": response.payload.get("metadata", {}),
                        "score": response.payload.get("score", 0.0),
                    }
                )

            logger.debug("Retrieved %s semantic memory chunks", len(results))
            return results

        except Exception as e:
            logger.warning("Failed to query semantic memory: %s", e)
            return []

    async def _query_procedural_memory(
        self,
        agent_type: str,
        task_description: str,
        tenant_id: str,
        limit: int = 5,
    ) -> list[Dict[str, Any]]:
        """Query Procedural Memory for relevant tools/examples.

        Args:
            agent_type: Type of agent
            task_description: Task description
            tenant_id: Tenant ID
            limit: Maximum results

        Returns:
            List of tool examples or procedural knowledge
        """
        try:
            # Query for tools relevant to agent type and task
            query = f"{agent_type} {task_description} tools examples"

            unit = ContextUnit(
                payload={
                    "tenant_id": tenant_id,
                    "query_text": query,
                    "limit": limit,
                    "source_types": ["procedural"],  # Filter for procedural memory
                },
                provenance=["router:subagent:prompt_generator"],
            )

            results = []
            async for response in self.brain.query_memory(unit):
                results.append(
                    {
                        "content": response.payload.get("content", ""),
                        "metadata": response.payload.get("metadata", {}),
                        "score": response.payload.get("score", 0.0),
                    }
                )

            logger.debug("Retrieved %s procedural memory chunks", len(results))
            return results

        except Exception as e:
            logger.warning("Failed to query procedural memory: %s", e)
            return []

    def _build_manifest(
        self,
        task: Dict[str, Any],
        agent_type: str,
        semantic_knowledge: list[Dict[str, Any]],
        procedural_knowledge: list[Dict[str, Any]],
    ) -> str:
        """Build manifest (system prompt) from knowledge.

        Args:
            task: Task description
            agent_type: Type of agent
            semantic_knowledge: Semantic memory results
            procedural_knowledge: Procedural memory results

        Returns:
            Complete system prompt/manifest
        """
        parts = []

        # Base role definition
        role_map = {
            "task_executor": "You are a specialized task executor agent. Your role is to execute specific tasks efficiently and accurately.",
            "parallel_worker": "You are a parallel processing worker agent. Your role is to process tasks in parallel with other workers.",
            "specialized_agent": "You are a specialized domain agent. Your role is to handle domain-specific tasks with expertise.",
        }

        role = role_map.get(
            agent_type, f"You are a {agent_type} agent. Your role is to execute tasks as specified."
        )
        parts.append(role)

        # Task description
        task_desc = task.get("description", "")
        if task_desc:
            parts.append(f"\n## Task\n{task_desc}")

        # Task context
        task_context = task.get("context", {})
        if task_context:
            parts.append(f"\n## Context\n{self._format_context(task_context)}")

        # Semantic knowledge
        if semantic_knowledge:
            parts.append("\n## Relevant Knowledge")
            for i, chunk in enumerate(semantic_knowledge[:3], 1):  # Top 3 chunks
                content = chunk.get("content", "")
                if content:
                    parts.append(f"\n### Knowledge {i}\n{content}")

        # Procedural knowledge (tools/examples)
        if procedural_knowledge:
            parts.append("\n## Available Tools & Examples")
            for i, chunk in enumerate(procedural_knowledge[:3], 1):  # Top 3 chunks
                content = chunk.get("content", "")
                if content:
                    parts.append(f"\n### Example {i}\n{content}")

        # Instructions
        parts.append("\n## Instructions")
        parts.append("1. Execute the task as described")
        parts.append("2. Use relevant knowledge and examples as guidance")
        parts.append("3. Report progress and results clearly")
        parts.append("4. Handle errors gracefully and report them")

        return "\n".join(parts)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dictionary into readable text.

        Args:
            context: Context dictionary

        Returns:
            Formatted context string
        """
        if isinstance(context, dict):
            lines = []
            for key, value in context.items():
                if isinstance(value, (dict, list)):
                    value = str(value)
                lines.append(f"- {key}: {value}")
            return "\n".join(lines)
        return str(context)
