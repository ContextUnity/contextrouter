"""Self-Healing Tools for Dispatcher Agent."""

from __future__ import annotations

import logging

from contextunity.api.admin_client import AdminClient
from contextunity.healing.code_fixer import CodeFixer
from contextunity.healing.service_healer import ServiceHealer
from langchain_core.tools import tool

from contextrouter.modules.tools import register_tool as _register_tool

logger = logging.getLogger(__name__)

admin_client = AdminClient(
    endpoint="unity.contextunity.ts.net:50056",
    mode="grpc",
)
service_healer = ServiceHealer(admin_client)
code_fixer = CodeFixer()


@tool
async def check_agent_status(agent_id: str) -> dict:
    """Check status of a specific agent.

    Args:
        agent_id: Agent identifier

    Returns:
        Agent status including health, last activity, error count, etc.
    """
    try:
        result = await admin_client.get_agent_activity(agent_id=agent_id, hours=1)
        activity = result.payload

        # Get agent config
        config_result = await admin_client.get_agent_config(agent_id=agent_id)
        config = config_result.payload

        return {
            "agent_id": agent_id,
            "status": "healthy" if activity.get("error_count", 0) == 0 else "unhealthy",
            "last_activity": activity.get("last_activity"),
            "error_count": activity.get("error_count", 0),
            "request_count": activity.get("request_count", 0),
            "config": config,
        }
    except Exception as e:
        logger.error("Failed to check agent status: %s", e)
        return {
            "agent_id": agent_id,
            "status": "unknown",
            "error": str(e),
        }


@tool
async def analyze_error(error_id: str | None = None, service_endpoint: str | None = None) -> dict:
    """Analyze an error and get recommendations.

    Args:
        error_id: Optional specific error ID to analyze
        service_endpoint: Optional service endpoint to analyze errors for

    Returns:
        Error analysis with recommendations for available tools.
    """
    try:
        # Detect errors
        errors_result = await admin_client.detect_errors()
        errors = errors_result.payload.get("errors", [])

        # Filter if specific error or service
        if error_id:
            errors = [e for e in errors if e.get("error_id") == error_id]
        elif service_endpoint:
            errors = [e for e in errors if e.get("service_endpoint") == service_endpoint]

        if not errors:
            return {
                "errors": [],
                "recommendations": [],
                "message": "No errors found",
            }

        # Analyze each error
        recommendations = []
        for error in errors:
            error_type = error.get("error_type")
            service = error.get("service_endpoint")
            severity = error.get("severity")

            # Generate recommendations based on error type
            recs = []

            if error_type == "service_unhealthy":
                recs.append(
                    {
                        "action": "restart_service",
                        "tool": "restart_service",
                        "params": {"service_endpoint": service},
                        "priority": "critical" if severity == "critical" else "high",
                        "description": f"Restart service {service}",
                    }
                )
                recs.append(
                    {
                        "action": "check_service_logs",
                        "tool": "get_service_logs",
                        "params": {"service_endpoint": service},
                        "priority": "medium",
                        "description": f"Check logs for {service}",
                    }
                )

            elif error_type == "high_latency":
                recs.append(
                    {
                        "action": "scale_service",
                        "tool": "scale_service",
                        "params": {"service_endpoint": service, "replicas": "+1"},
                        "priority": "high",
                        "description": f"Scale up {service}",
                    }
                )
                recs.append(
                    {
                        "action": "check_service_metrics",
                        "tool": "get_service_metrics",
                        "params": {"service_endpoint": service},
                        "priority": "medium",
                        "description": f"Check metrics for {service}",
                    }
                )

            elif error_type == "code_error":
                recs.append(
                    {
                        "action": "fix_code",
                        "tool": "fix_code_error",
                        "params": {
                            "error": error.get("error_details"),
                            "file_path": error.get("file_path"),
                        },
                        "priority": "high",
                        "description": f"Fix code error in {error.get('file_path')}",
                    }
                )

            recommendations.extend(recs)

        return {
            "errors": errors,
            "recommendations": recommendations,
            "total_errors": len(errors),
        }
    except Exception as e:
        logger.error("Failed to analyze error: %s", e)
        return {
            "errors": [],
            "recommendations": [],
            "error": str(e),
        }


@tool
async def restart_service(service_endpoint: str) -> dict:
    """Restart a service.

    Args:
        service_endpoint: Service endpoint to restart

    Returns:
        Restart operation result.
    """
    try:
        result = await service_healer.restart_service(service_endpoint)
        return result
    except Exception as e:
        logger.error("Failed to restart service: %s", e)
        return {
            "success": False,
            "error": str(e),
        }


@tool
async def scale_service(service_endpoint: str, replicas: str) -> dict:
    """Scale a service up or down.

    Args:
        service_endpoint: Service endpoint to scale
        replicas: Number of replicas (e.g., "+1", "-1", "3")

    Returns:
        Scale operation result.
    """
    try:
        result = await service_healer.scale_service(service_endpoint, replicas)
        return result
    except Exception as e:
        logger.error("Failed to scale service: %s", e)
        return {
            "success": False,
            "error": str(e),
        }


@tool
async def get_service_logs(service_endpoint: str, lines: int = 100) -> dict:
    """Get recent logs from a service.

    Args:
        service_endpoint: Service endpoint
        lines: Number of lines to retrieve (default: 100)

    Returns:
        Service logs.
    """
    try:
        result = await admin_client.get_service_logs(service_endpoint, lines=lines)
        return result.payload
    except Exception as e:
        logger.error("Failed to get service logs: %s", e)
        return {
            "logs": [],
            "error": str(e),
        }


@tool
async def fix_code_error(error: str, file_path: str) -> dict:
    """Fix a code error automatically.

    Args:
        error: Error message or details
        file_path: Path to the file with error

    Returns:
        Fix result including updated code and explanation.
    """
    try:
        result = await code_fixer.fix_error(error, file_path)
        return result
    except Exception as e:
        logger.error("Failed to fix code error: %s", e)
        return {
            "success": False,
            "error": str(e),
        }


@tool
async def evaluate_healing_result(healing_action: str, service_endpoint: str) -> dict:
    """Evaluate result of a healing action.

    Args:
        healing_action: Action that was taken (e.g., "restart_service")
        service_endpoint: Service endpoint that was healed

    Returns:
        Evaluation result including success status, health check, recommendations.
    """
    try:
        # Check service health after healing
        health_result = await admin_client.get_service_health()
        services = health_result.payload.get("services", {})
        service_health = services.get(service_endpoint)

        if not service_health:
            return {
                "success": False,
                "message": "Service not found in health check",
            }

        status = service_health.get("status")
        latency_ms = service_health.get("latency_ms", 0)

        # Evaluate success
        success = status == "healthy" and latency_ms < 1000

        # Generate recommendations
        recommendations = []
        if not success:
            if status == "unhealthy":
                recommendations.append(
                    {
                        "action": "restart_service",
                        "tool": "restart_service",
                        "description": "Service is still unhealthy, try restart again",
                    }
                )
            elif latency_ms > 1000:
                recommendations.append(
                    {
                        "action": "scale_service",
                        "tool": "scale_service",
                        "description": "Service latency is still high, consider scaling",
                    }
                )

        return {
            "success": success,
            "healing_action": healing_action,
            "service_endpoint": service_endpoint,
            "current_status": status,
            "latency_ms": latency_ms,
            "recommendations": recommendations,
        }
    except Exception as e:
        logger.error("Failed to evaluate healing result: %s", e)
        return {
            "success": False,
            "error": str(e),
        }


@tool
async def orchestrate_agents(agent_ids: list[str], action: str) -> dict:
    """Orchestrate multiple agents (e.g., restart, pause, resume).

    Args:
        agent_ids: List of agent IDs to orchestrate
        action: Action to perform (restart, pause, resume)

    Returns:
        Orchestration result.
    """
    try:
        results = []
        for agent_id in agent_ids:
            if action == "restart":
                # Restart agent (would require k8s API or similar)
                result = await admin_client.restart_agent(agent_id)
                results.append(
                    {
                        "agent_id": agent_id,
                        "action": action,
                        "result": result.payload,
                    }
                )
            elif action == "pause":
                # Pause agent
                result = await admin_client.pause_agent(agent_id)
                results.append(
                    {
                        "agent_id": agent_id,
                        "action": action,
                        "result": result.payload,
                    }
                )
            elif action == "resume":
                # Resume agent
                result = await admin_client.resume_agent(agent_id)
                results.append(
                    {
                        "agent_id": agent_id,
                        "action": action,
                        "result": result.payload,
                    }
                )

        return {
            "success": True,
            "results": results,
        }
    except Exception as e:
        logger.error("Failed to orchestrate agents: %s", e)
        return {
            "success": False,
            "error": str(e),
        }


# Register all tools
_register_tool(check_agent_status)
_register_tool(analyze_error)
_register_tool(restart_service)
_register_tool(scale_service)
_register_tool(get_service_logs)
_register_tool(fix_code_error)
_register_tool(evaluate_healing_result)
_register_tool(orchestrate_agents)
