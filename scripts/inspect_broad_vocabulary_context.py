from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.extraction_vocabulary_context import (
    BROAD_METHODS_ONLY_CONTEXT_ID,
    build_broad_experiment_methods_vocabulary_context,
)
from dac_her.llm_telemetry import component_fingerprint, estimate_tokens
from dac_her.vocab_registry import load_default_registries


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_CONTEXT_ID = "registry-context-full-v1"


def _full_context(experiment_registry, metric_registry) -> str:
    return "\n".join([
        "REGISTERED EXPERIMENT METHODS:",
        *experiment_registry.prompt_lines(metadata_keys=("family",)),
        "",
        "REGISTERED MEASUREMENT METRICS:",
        *metric_registry.prompt_lines(
            metadata_keys=("canonical_unit", "parameters")
        ),
    ])


def _metric_section(metric_registry) -> str:
    return "\n".join([
        "REGISTERED MEASUREMENT METRICS:",
        *metric_registry.prompt_lines(
            metadata_keys=("canonical_unit", "parameters")
        ),
    ])


def _summary(text: str, serialization_id: str) -> dict:
    tokens, estimator = estimate_tokens(text)
    return {
        "serialization_id": serialization_id,
        "characters": len(text),
        "estimated_tokens": tokens,
        "estimator": estimator,
        "fingerprint": component_fingerprint(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    experiment_registry, metric_registry = load_default_registries(PROJECT_ROOT)
    full = _full_context(experiment_registry, metric_registry)
    pruned = build_broad_experiment_methods_vocabulary_context(
        experiment_registry
    )
    metric_only = _metric_section(metric_registry)

    full_summary = _summary(full, FULL_CONTEXT_ID)
    pruned_summary = _summary(pruned, BROAD_METHODS_ONLY_CONTEXT_ID)
    metric_summary = _summary(
        metric_only,
        "measurement-metric-section-diagnostic",
    )

    delta = (
        full_summary["estimated_tokens"]
        - pruned_summary["estimated_tokens"]
    )
    payload = {
        "schema_version": "broad-vocabulary-serialization-audit-v1",
        "full": full_summary,
        "pruned": pruned_summary,
        "measurement_metric_section": metric_summary,
        "estimated_token_delta_per_call": delta,
        "estimated_reduction_fraction": (
            delta / full_summary["estimated_tokens"]
            if full_summary["estimated_tokens"]
            else 0.0
        ),
    }

    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    print(rendered)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
