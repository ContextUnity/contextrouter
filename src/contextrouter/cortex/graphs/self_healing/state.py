"""State for Self-Healing subgraph."""

from typing import Any, Dict, List, Optional, TypedDict


class SelfHealingState(TypedDict):
    """State for Self-Healing subgraph."""

    # Optional: specific agent to check
    agent_id: Optional[str]
    agent_status: Optional[Dict[str, Any]]

    # Error detection
    detected_errors: List[Dict[str, Any]]
    error_count: int

    # Error analysis
    error_analysis: Dict[str, Any]

    # Recommendations
    recommendations: List[Dict[str, Any]]
    available_tools: List[str]

    # Healing results
    healing_results: List[Dict[str, Any]]

    # Evaluation
    evaluations: List[Dict[str, Any]]
    needs_retry: List[Dict[str, Any]]
    needs_orchestration: List[Dict[str, Any]]
    successful_healings: List[Dict[str, Any]]

    # Orchestration
    orchestration_results: List[Dict[str, Any]]

    # Report
    healing_report: Dict[str, Any]
