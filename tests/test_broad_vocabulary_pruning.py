from __future__ import annotations

from pathlib import Path

import yaml

from pipeline_core.corpus.extraction_vocabulary_context import (
    BROAD_METHODS_ONLY_CONTEXT_ID,
    build_broad_experiment_methods_vocabulary_context,
)
from pipeline_core.llm.llm_telemetry import estimate_tokens
from pipeline_core.corpus.vocab_registry import load_default_registries


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _legacy_full_context(experiment_registry, metric_registry) -> str:
    return "\n".join([
        "REGISTERED EXPERIMENT METHODS:",
        *experiment_registry.prompt_lines(metadata_keys=("family",)),
        "",
        "REGISTERED MEASUREMENT METRICS:",
        *metric_registry.prompt_lines(
            metadata_keys=("canonical_unit", "parameters")
        ),
    ])


def test_broad_pruned_context_keeps_methods_and_drops_metrics():
    experiment_registry, _ = load_default_registries(PROJECT_ROOT)
    context = build_broad_experiment_methods_vocabulary_context(
        experiment_registry
    )

    assert "REGISTERED EXPERIMENT METHODS:" in context
    assert "xps: X-ray photoelectron spectroscopy" in context
    assert "REGISTERED MEASUREMENT METRICS:" not in context
    assert "overpotential: Overpotential" not in context
    assert BROAD_METHODS_ONLY_CONTEXT_ID.endswith("-v1")


def test_broad_pruned_context_is_smaller_than_legacy_full_surface():
    experiment_registry, metric_registry = load_default_registries(PROJECT_ROOT)
    full = _legacy_full_context(experiment_registry, metric_registry)
    pruned = build_broad_experiment_methods_vocabulary_context(
        experiment_registry
    )
    full_tokens, _ = estimate_tokens(full)
    pruned_tokens, _ = estimate_tokens(pruned)

    assert pruned_tokens < full_tokens


