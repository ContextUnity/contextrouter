"""State for Analytics subgraph."""

from typing import Any, Dict, List, TypedDict


class AnalyticsState(TypedDict):
    """State for Analytics subgraph."""

    # Configuration
    hours: int

    # Collected metrics
    system_analytics: Dict[str, Any]
    error_analytics: Dict[str, Any]
    performance_analytics: Dict[str, Any]

    # Analysis results
    trends: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    insights: List[str]

    # Report
    analytics_report: Dict[str, Any]
