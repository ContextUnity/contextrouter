"""
Self-Healing Subgraph — Automatic Error Detection & Recovery.

This subgraph monitors services, detects errors, and attempts automatic recovery.

Flow:
    check_agent_status → detect_errors → analyze_errors → get_recommendations →
    apply_healing_tool → evaluate_result → [retry|orchestrate|report] → END
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

from contextunity.analytics.error_detector import ErrorDetector
from contextunity.api.admin_client import AdminClient
from contextunity.healing.service_healer import ServiceHealer
from langgraph.graph import END, StateGraph

from contextrouter.core.registry import register_graph
from contextrouter.modules.tools import discover_all_tools

from .state import SelfHealingState

logger = logging.getLogger(__name__)

# Initialize clients
admin_client = AdminClient(
    endpoint="unity.contextunity.ts.net:50056",
    mode="grpc",
)
error_detector = ErrorDetector(admin_client)
service_healer = ServiceHealer(admin_client)


async def check_agent_status_node(state: SelfHealingState) -> Dict[str, Any]:
    """Check status of agents (if agent_id provided)."""
    agent_id = state.get("agent_id")

    if agent_id:
        logger.info("Checking status of agent %s...", agent_id)

        # Get agent activity
        activity_result = await admin_client.get_agent_activity(agent_id=agent_id, hours=1)
        activity = activity_result.payload

        return {
            "agent_status": {
                "agent_id": agent_id,
                "status": "healthy" if activity.get("error_count", 0) == 0 else "unhealthy",
                "error_count": activity.get("error_count", 0),
                "last_activity": activity.get("last_activity"),
            },
        }

    return {}


async def detect_errors_node(state: SelfHealingState) -> Dict[str, Any]:
    """Detect errors across all services."""
    logger.info("Detecting errors across services...")

    errors = await error_detector.detect_all()

    return {
        "detected_errors": errors,
        "error_count": len(errors),
    }


async def analyze_errors_node(state: SelfHealingState) -> Dict[str, Any]:
    """Analyze detected errors and prioritize them."""
    logger.info("Analyzing errors...")

    errors = state.get("detected_errors", [])

    # Categorize errors by severity
    critical = [e for e in errors if e.get("severity") == "critical"]
    high = [e for e in errors if e.get("severity") == "high"]
    medium = [e for e in errors if e.get("severity") == "medium"]
    low = [e for e in errors if e.get("severity") == "low"]

    # Analyze patterns
    error_patterns = error_detector.analyze_patterns(errors)

    return {
        "error_analysis": {
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
            "patterns": error_patterns,
        },
    }


async def get_recommendations_node(state: SelfHealingState) -> Dict[str, Any]:
    """Get recommendations for fixing errors with available tools."""
    logger.info("Getting recommendations...")

    errors = state.get("detected_errors", [])

    # Get available healing tools
    tools = discover_all_tools()
    healing_tools = [
        t
        for t in tools
        if hasattr(t, "name")
        and any(kw in t.name.lower() for kw in ["heal", "restart", "scale", "fix", "check"])
    ]

    recommendations = []
    for error in errors:
        error_type = error.get("error_type")
        service_endpoint = error.get("service_endpoint")
        severity = error.get("severity")

        # Generate recommendations based on error type
        if error_type == "service_unhealthy":
            recommendations.append(
                {
                    "error": error,
                    "recommended_tool": "restart_service",
                    "tool_params": {"service_endpoint": service_endpoint},
                    "priority": "critical" if severity == "critical" else "high",
                    "description": f"Restart service {service_endpoint}",
                }
            )
        elif error_type == "high_latency":
            recommendations.append(
                {
                    "error": error,
                    "recommended_tool": "scale_service",
                    "tool_params": {"service_endpoint": service_endpoint, "replicas": "+1"},
                    "priority": "high",
                    "description": f"Scale up {service_endpoint}",
                }
            )
        elif error_type == "code_error":
            recommendations.append(
                {
                    "error": error,
                    "recommended_tool": "fix_code_error",
                    "tool_params": {
                        "error": error.get("error_details"),
                        "file_path": error.get("file_path"),
                    },
                    "priority": "high",
                    "description": f"Fix code error in {error.get('file_path')}",
                }
            )

    return {
        "recommendations": recommendations,
        "available_tools": [t.name for t in healing_tools],
    }


async def apply_healing_tool_node(state: SelfHealingState) -> Dict[str, Any]:
    """Apply healing tool based on recommendations."""
    logger.info("Applying healing tools...")

    recommendations = state.get("recommendations", [])

    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_recs = sorted(
        recommendations,
        key=lambda r: priority_order.get(r.get("priority", "low"), 3),
    )

    healing_results = []

    # Get available tools
    tools = discover_all_tools()
    tool_map = {t.name: t for t in tools}

    for rec in sorted_recs[:5]:  # Limit to top 5
        tool_name = rec.get("recommended_tool")
        tool_params = rec.get("tool_params", {})
        error = rec.get("error")

        if tool_name in tool_map:
            tool = tool_map[tool_name]
            try:
                logger.info("Applying tool %s with params %s", tool_name, tool_params)
                result = await tool.ainvoke(tool_params)

                healing_results.append(
                    {
                        "error": error,
                        "tool": tool_name,
                        "tool_params": tool_params,
                        "result": result,
                        "success": result.get("success", False),
                    }
                )
            except Exception as e:
                logger.error("Failed to apply tool %s: %s", tool_name, e)
                healing_results.append(
                    {
                        "error": error,
                        "tool": tool_name,
                        "tool_params": tool_params,
                        "result": {"error": str(e)},
                        "success": False,
                    }
                )
        else:
            logger.warning("Tool %s not found", tool_name)
            healing_results.append(
                {
                    "error": error,
                    "tool": tool_name,
                    "result": {"error": f"Tool {tool_name} not available"},
                    "success": False,
                }
            )

    return {
        "healing_results": healing_results,
    }


async def evaluate_result_node(state: SelfHealingState) -> Dict[str, Any]:
    """Evaluate healing results and decide next action."""
    logger.info("Evaluating healing results...")

    healing_results = state.get("healing_results", [])

    # Get evaluation tool
    all_tools = discover_all_tools()
    eval_tool = next(
        (
            t
            for t in all_tools
            if hasattr(t, "name") and getattr(t, "name", None) == "evaluate_healing_result"
        ),
        None,
    )

    evaluations = []
    needs_retry = []
    needs_orchestration = []

    for result in healing_results:
        if not result.get("success"):
            # Evaluate why it failed
            if eval_tool:
                try:
                    eval_params = {
                        "healing_action": result.get("tool"),
                        "service_endpoint": result.get("error", {}).get("service_endpoint", ""),
                    }
                    # Use ainvoke if available
                    if hasattr(eval_tool, "ainvoke"):
                        eval_result = await eval_tool.ainvoke(eval_params)
                    else:
                        eval_result = eval_tool.invoke(eval_params)

                    evaluations.append(eval_result)

                    # Check if retry is recommended
                    recs = (
                        eval_result.get("recommendations", [])
                        if isinstance(eval_result, dict)
                        else []
                    )
                    if recs:
                        needs_retry.append(
                            {
                                "original_result": result,
                                "evaluation": eval_result,
                                "recommendations": recs,
                            }
                        )
                except Exception as e:
                    logger.error("Failed to evaluate result: %s", e)

            # Check if orchestration is needed (critical errors)
            error = result.get("error", {})
            if isinstance(error, dict) and error.get("severity") == "critical":
                needs_orchestration.append(
                    {
                        "error": error,
                        "failed_healing": result,
                    }
                )

    return {
        "evaluations": evaluations,
        "needs_retry": needs_retry,
        "needs_orchestration": needs_orchestration,
        "successful_healings": [r for r in healing_results if r.get("success")],
    }


async def orchestrate_agents_node(state: SelfHealingState) -> Dict[str, Any]:
    """Orchestrate agents (restart, pause, resume) for critical errors."""
    logger.info("Orchestrating agents...")

    needs_orchestration = state.get("needs_orchestration", [])

    if not needs_orchestration:
        return {}

    # Get orchestrate tool
    all_tools = discover_all_tools()
    orchestrate_tool = next(
        (
            t
            for t in all_tools
            if hasattr(t, "name") and getattr(t, "name", None) == "orchestrate_agents"
        ),
        None,
    )

    if not orchestrate_tool:
        logger.warning("Orchestrate tool not available")
        return {
            "orchestration_results": [],
            "message": "Orchestration tool not available",
        }

    orchestration_results = []

    for item in needs_orchestration:
        error = item.get("error", {})
        service_endpoint = error.get("service_endpoint", "")

        # Extract agent IDs from service endpoint (simplified)
        # In production, would map service to agent IDs
        agent_ids = [service_endpoint.split(".")[0]] if service_endpoint else []

        try:
            orchestrate_params = {
                "agent_ids": agent_ids,
                "action": "restart",
            }
            # Use ainvoke if available
            if hasattr(orchestrate_tool, "ainvoke"):
                result = await orchestrate_tool.ainvoke(orchestrate_params)
            else:
                result = orchestrate_tool.invoke(orchestrate_params)

            orchestration_results.append(
                {
                    "error": error,
                    "orchestration_result": result,
                }
            )
        except Exception as e:
            logger.error("Failed to orchestrate agents: %s", e)
            orchestration_results.append(
                {
                    "error": error,
                    "orchestration_result": {"error": str(e)},
                }
            )

    return {
        "orchestration_results": orchestration_results,
    }


async def report_node(state: SelfHealingState) -> Dict[str, Any]:
    """Generate healing report."""
    logger.info("Generating healing report...")

    healing_results = state.get("healing_results", [])
    evaluations = state.get("evaluations", [])
    orchestration_results = state.get("orchestration_results", [])

    successful = [r for r in healing_results if r.get("success")]
    failed = [r for r in healing_results if not r.get("success")]

    report = {
        "summary": {
            "total_errors": len(state.get("detected_errors", [])),
            "healing_attempts": len(healing_results),
            "successful_healings": len(successful),
            "failed_healings": len(failed),
            "orchestrations": len(orchestration_results),
        },
        "details": {
            "successful": successful,
            "failed": failed,
            "evaluations": evaluations,
            "orchestrations": orchestration_results,
        },
    }

    return {
        "healing_report": report,
    }


def _should_get_recommendations(
    state: SelfHealingState,
) -> Literal["get_recommendations", "report"]:
    """Decide whether to get recommendations."""
    errors = state.get("detected_errors", [])

    if errors:
        return "get_recommendations"

    return "report"


def _should_apply_healing(state: SelfHealingState) -> Literal["apply_healing", "report"]:
    """Decide whether to apply healing tools."""
    recommendations = state.get("recommendations", [])

    if recommendations:
        return "apply_healing"

    return "report"


def _should_evaluate(state: SelfHealingState) -> Literal["evaluate", "orchestrate", "report"]:
    """Decide whether to evaluate results or orchestrate."""
    healing_results = state.get("healing_results", [])

    if healing_results:
        return "evaluate"

    return "report"


def _should_orchestrate(state: SelfHealingState) -> Literal["orchestrate", "retry", "report"]:
    """Decide whether to orchestrate agents or retry."""
    needs_orchestration = state.get("needs_orchestration", [])
    needs_retry = state.get("needs_retry", [])

    if needs_orchestration:
        return "orchestrate"

    if needs_retry:
        return "retry"

    return "report"


@register_graph("self_healing")
def build_self_healing_graph():
    """Build Self-Healing subgraph.

    Flow:
        check_agent_status (optional) → detect_errors → analyze_errors →
        get_recommendations → apply_healing_tool → evaluate_result →
        [orchestrate|retry|report] → END
    """
    workflow = StateGraph(SelfHealingState)

    # Add nodes
    workflow.add_node("check_agent_status", check_agent_status_node)
    workflow.add_node("detect_errors", detect_errors_node)
    workflow.add_node("analyze_errors", analyze_errors_node)
    workflow.add_node("get_recommendations", get_recommendations_node)
    workflow.add_node("apply_healing", apply_healing_tool_node)
    workflow.add_node("evaluate", evaluate_result_node)
    workflow.add_node("orchestrate", orchestrate_agents_node)
    workflow.add_node("report", report_node)

    # Entry
    workflow.set_entry_point("check_agent_status")

    # Flow
    workflow.add_edge("check_agent_status", "detect_errors")
    workflow.add_edge("detect_errors", "analyze_errors")
    workflow.add_conditional_edges(
        "analyze_errors",
        _should_get_recommendations,
        {
            "get_recommendations": "get_recommendations",
            "report": "report",
        },
    )
    workflow.add_conditional_edges(
        "get_recommendations",
        _should_apply_healing,
        {
            "apply_healing": "apply_healing",
            "report": "report",
        },
    )
    workflow.add_conditional_edges(
        "apply_healing",
        _should_evaluate,
        {
            "evaluate": "evaluate",
            "report": "report",
        },
    )
    workflow.add_conditional_edges(
        "evaluate",
        _should_orchestrate,
        {
            "orchestrate": "orchestrate",
            "retry": "apply_healing",  # Retry healing
            "report": "report",
        },
    )
    workflow.add_edge("orchestrate", "report")
    workflow.add_edge("report", END)

    return workflow.compile()
