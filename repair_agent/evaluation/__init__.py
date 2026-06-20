"""Evaluation: metrics, prediction validation, credit assignment, run summaries."""

from repair_agent.evaluation.metrics import (
    aggregate_run_metrics,
    denominator_counts,
    pass_at_k,
    summarize_model_gates,
    summarize_resources,
    validate_predictions_file,
)

__all__ = [
    "aggregate_run_metrics",
    "denominator_counts",
    "pass_at_k",
    "summarize_model_gates",
    "summarize_resources",
    "validate_predictions_file",
]
