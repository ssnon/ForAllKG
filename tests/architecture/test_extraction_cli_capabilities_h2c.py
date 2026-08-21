from __future__ import annotations

import sys
from pathlib import Path

from scripts.corpus import extract_paper


ROOT = Path(__file__).resolve().parents[2]


def test_extract_paper_orchestration_is_domain_name_free() -> None:
    path = (
        ROOT
        / "scripts"
        / "corpus"
        / "extract_paper.py"
    )

    source = path.read_text(
        encoding="utf-8"
    )

    assert "catalysis_mechanism" not in source
    assert "args.broad_" not in source

    assert (
        "extraction_adapter.compact_generation_response_model"
        in source
    )

    assert (
        "compact_domain_gate_recovery_response_model"
        in source
    )

    assert (
        "extraction_adapter.reduced_vocabulary_context_builder"
        in source
    )

    # Legacy CLI spellings remain accepted only as parser aliases.
    assert source.count("--broad-compact-schema") >= 1
    assert source.count("--broad-compact-domain-recovery") >= 1
    assert source.count("--broad-prune-metric-vocabulary") >= 1


def test_generic_capability_cli_names_parse_to_generic_destinations(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_paper",
            "--paper-id",
            "TEST",
            "--compact-generation-schema",
            "--compact-domain-gate-recovery",
            "--reduced-vocabulary-context",
        ],
    )

    args = extract_paper.parse_args()

    assert args.compact_generation_schema is True
    assert args.compact_domain_gate_recovery is True
    assert args.reduced_vocabulary_context is True

    assert not hasattr(
        args,
        "broad_compact_schema",
    )

    assert not hasattr(
        args,
        "broad_compact_domain_recovery",
    )

    assert not hasattr(
        args,
        "broad_prune_metric_vocabulary",
    )


def test_legacy_broad_cli_aliases_map_to_generic_destinations(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_paper",
            "--paper-id",
            "TEST",
            "--broad-compact-schema",
            "--broad-compact-domain-recovery",
            "--broad-prune-metric-vocabulary",
        ],
    )

    args = extract_paper.parse_args()

    assert args.compact_generation_schema is True
    assert args.compact_domain_gate_recovery is True
    assert args.reduced_vocabulary_context is True
