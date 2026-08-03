from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionPolicy:
    # Source core chunk
    target_source_tokens: int = 1400
    max_source_tokens: int = 1800

    # Reference-resolution context only
    left_context_tokens: int = 120
    right_context_tokens: int = 120

    # Model output
    max_completion_tokens: int = 8000

    # Execution
    logical_batch_size: int = 10
    concurrency: int = 2

    # Failure handling
    max_api_retries: int = 3
    max_semantic_repairs: int = 2
    max_split_depth: int = 3

    # Output utilization warnings
    warning_utilization: float = 0.80
    critical_utilization: float = 0.95

    min_source_tokens: int = 500