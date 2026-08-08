from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionPolicy:
    # Source core chunk
    target_source_tokens: int = 1400
    max_source_tokens: int = 1800
    min_source_tokens: int = 500

    # Reference-resolution context only
    left_context_tokens: int = 120
    right_context_tokens: int = 120

    # Model output
    max_completion_tokens: int = 16000
    patch_completion_tokens: int = 6000

    # Execution
    logical_batch_size: int = 10
    concurrency: int = 2

    # Technical and recovery handling
    max_api_retries: int = 3
    max_generation_attempts: int = 2
    # Tiny-leaf recovery
    max_micro_reextract_attempts: int = 1
    micro_reextract_max_source_tokens: int = 400
    max_post_micro_patch_attempts: int = 1
    
    max_patch_attempts: int = 2
    max_patch_operations: int = 12
    max_split_depth: int = 3
    min_rechunk_source_tokens: int = 400

    # Semantic recovery routing
    semantic_rechunk_isolated_threshold: int = 5
    semantic_rechunk_issue_family_threshold: int = 3
    semantic_rechunk_undefined_endpoint_threshold: int = 2

    # Strict acceptance: no destructive graph edits by default.
    allow_destructive_patches: bool = False

    # Paper-level partial graph materialization.
    #
    # Keep local strict validators unchanged. These thresholds only decide
    # whether already strict-valid leaves can be materialized when sibling
    # leaves were quarantined. Source-token coverage is intentionally more
    # important than raw chunk-success counts.
    partial_acceptable_min_source_token_coverage: float = 0.95
    partial_acceptable_max_quarantine_token_fraction: float = 0.05
    partial_critical_min_source_token_coverage: float = 0.70

    # Output utilization warnings
    warning_utilization: float = 0.80
    critical_utilization: float = 0.953

