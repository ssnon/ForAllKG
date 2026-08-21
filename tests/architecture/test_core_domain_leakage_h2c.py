from __future__ import annotations

from pathlib import Path

from pipeline_core.llm.llm_telemetry import (
    normalize_stage_name,
)


ROOT = Path(__file__).resolve().parents[2]


def test_core_telemetry_has_no_broad_response_model_special_case() -> None:
    path = (
        ROOT
        / "pipeline_core"
        / "llm"
        / "llm_telemetry.py"
    )

    source = path.read_text(
        encoding="utf-8"
    )

    assert (
        "BroadMechanismGraphDraft"
        not in source
    )

    assert (
        'lowered.endswith("graphdraft")'
        in source
    )


def test_generic_graph_draft_stage_normalization_preserves_broad_parity() -> None:
    assert (
        normalize_stage_name(
            None,
            response_model="BroadMechanismGraphDraft",
        )
        == "graph_generation"
    )

    assert (
        normalize_stage_name(
            None,
            response_model="FutureDomainGraphDraft",
        )
        == "graph_generation"
    )

    assert (
        normalize_stage_name(
            None,
            response_model="KnowledgeGraphDraft",
        )
        == "graph_generation"
    )


def test_patch_and_micro_stage_normalization_remain_unchanged() -> None:
    assert (
        normalize_stage_name(
            None,
            response_model="KnowledgeGraphPatch",
        )
        == "semantic_patch"
    )

    assert (
        normalize_stage_name(
            None,
            response_model="SemanticRepairPatch",
        )
        == "semantic_patch"
    )

    assert (
        normalize_stage_name(
            None,
            response_model="MicroReextractResponse",
        )
        == "micro_reextract"
    )


def test_pipeline_core_has_no_current_catalysis_profile_identifier() -> None:
    violations = []

    for path in (
        ROOT
        / "pipeline_core"
    ).rglob("*.py"):
        source = path.read_text(
            encoding="utf-8"
        )

        if "catalysis_mechanism" in source:
            violations.append(
                str(
                    path.relative_to(ROOT)
                )
            )

    assert violations == []
