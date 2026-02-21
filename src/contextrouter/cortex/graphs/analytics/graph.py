"""
Analytics Agent Subgraph — System Analytics & Monitoring.

This subgraph provides analytics capabilities for monitoring and analyzing
ContextUnity ecosystem performance, errors, and usage patterns.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

from contextunity.analytics.analytics_agent import AnalyticsAgent
from contextunity.api.admin_client import AdminClient
from langgraph.graph import END, StateGraph

from contextrouter.core.registry import register_graph

from .state import AnalyticsState

logger = logging.getLogger(__name__)

# Initialize clients
admin_client = AdminClient(
    endpoint="unity.contextunity.ts.net:50056",
    mode="grpc",
)
analytics_agent = AnalyticsAgent(admin_client)


async def collect_metrics_node(state: AnalyticsState) -> Dict[str, Any]:
    """Collect metrics from all services."""
    logger.info("Collecting metrics...")

    hours = state.get("hours", 24)

    # Get system analytics
    system_analytics = await admin_client.get_system_analytics(hours=hours)

    # Get error analytics
    error_analytics = await admin_client.get_error_analytics(hours=hours)

    # Get performance analytics
    performance_analytics = await admin_client.get_performance_analytics(hours=hours)

    return {
        "system_analytics": system_analytics.payload,
        "error_analytics": error_analytics.payload,
        "performance_analytics": performance_analytics.payload,
    }


async def analyze_metrics_node(state: AnalyticsState) -> Dict[str, Any]:
    """Analyze collected metrics."""
    logger.info("Analyzing metrics...")

    system_analytics = state.get("system_analytics", {})
    error_analytics = state.get("error_analytics", {})
    performance_analytics = state.get("performance_analytics", {})

    # Analyze trends
    trends = analytics_agent.analyze_trends(
        system_analytics=system_analytics,
        error_analytics=error_analytics,
        performance_analytics=performance_analytics,
    )

    # Identify anomalies
    anomalies = analytics_agent.detect_anomalies(
        system_analytics=system_analytics,
        error_analytics=error_analytics,
        performance_analytics=performance_analytics,
    )

    # Generate insights
    insights = analytics_agent.generate_insights(
        trends=trends,
        anomalies=anomalies,
    )

    return {
        "trends": trends,
        "anomalies": anomalies,
        "insights": insights,
    }


async def generate_report_node(state: AnalyticsState) -> Dict[str, Any]:
    """Generate analytics report."""
    logger.info("Generating analytics report...")

    system_analytics = state.get("system_analytics", {})
    error_analytics = state.get("error_analytics", {})
    performance_analytics = state.get("performance_analytics", {})
    trends = state.get("trends", {})
    anomalies = state.get("anomalies", {})
    insights = state.get("insights", [])

    report = analytics_agent.generate_report(
        system_analytics=system_analytics,
        error_analytics=error_analytics,
        performance_analytics=performance_analytics,
        trends=trends,
        anomalies=anomalies,
        insights=insights,
    )

    return {
        "analytics_report": report,
    }


def _should_analyze(state: AnalyticsState) -> Literal["analyze", "report"]:
    """Decide whether to analyze metrics."""
    system_analytics = state.get("system_analytics")

    if system_analytics:
        return "analyze"

    return "report"


@register_graph("analytics")
def build_analytics_graph():
    """Build Analytics Agent subgraph.

    Flow:
        collect_metrics → analyze_metrics → generate_report → END
    """
    workflow = StateGraph(AnalyticsState)

    # Add nodes
    workflow.add_node("collect_metrics", collect_metrics_node)
    workflow.add_node("analyze_metrics", analyze_metrics_node)
    workflow.add_node("generate_report", generate_report_node)

    # Entry
    workflow.set_entry_point("collect_metrics")

    # Flow
    workflow.add_conditional_edges(
        "collect_metrics",
        _should_analyze,
        {
            "analyze": "analyze_metrics",
            "report": "generate_report",
        },
    )
    workflow.add_edge("analyze_metrics", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()
