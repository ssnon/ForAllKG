from __future__ import annotations

from pathlib import Path

import yaml

from dac_her.broad_corpus_pipeline import (
    BroadCorpusPilotPipeline,
    BroadPilotOptions,
)
from dac_her.extraction_vocabulary_context import (
    BROAD_METHODS_ONLY_CONTEXT_ID,
    build_broad_experiment_methods_vocabulary_context,
)
from dac_her.llm_telemetry import estimate_tokens
from dac_her.vocab_registry import load_default_registries


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


def test_broad_pipeline_propagates_metric_vocab_pruning_flag(
    tmp_path: Path,
):
    papers = tmp_path / "papers.yaml"
    papers.write_text(
        yaml.safe_dump(
            {
                "version": 3,
                "papers": {
                    "broad_A": {
                        "enabled": True,
                        "documents": [],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    pipeline = BroadCorpusPilotPipeline(
        project_root=tmp_path,
        papers_yaml=papers,
        corpus_id="broad-vocab-pruning-smoke",
        options=BroadPilotOptions(
            data_root="data_broad",
            dry_run=True,
            broad_compact_schema=True,
            broad_prune_metric_vocabulary=True,
        ),
        requested_paper_ids=["broad_A"],
    )

    command = pipeline.paper_command("broad_A", "extract")
    assert "--broad-compact-schema" in command
    assert "--broad-prune-metric-vocabulary" in command
