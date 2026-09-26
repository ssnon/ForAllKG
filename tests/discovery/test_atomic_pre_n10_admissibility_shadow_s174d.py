from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.atomic_pre_n10_admissibility_shadow import (
    assess_atomic_pre_n10_shadow_row,
    compile_atomic_pre_n10_admissibility_shadow,
)
from pipeline_core.discovery.atomic_scientific_specification import (
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
)
from pipeline_core.discovery.legacy_atomic_specification_shadow import (
    LegacyAtomicSpecificationShadowReport,
    LegacyAtomicSpecificationShadowRow,
)


def _sha(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _spec(claim_id: str = "claim:1") -> CompiledAtomicSpecification:
    return CompiledAtomicSpecification(
        local_id="c1",
        claim_id=claim_id,
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="Factor M moderates descriptor D and outcome O.",
        rationale="test",
        source_candidate_ids=[],
        premise_statement_ids=["statement:1"],
        gap_statement_ids=[],
        prior_art_identity_terms=["Factor M"],
        relation_endpoint_anchors=["descriptor D", "outcome O"],
        scope_qualifier_spans=[],
        directional_qualifier_spans=[],
        relation_nucleus_terms=["descriptor D", "outcome O"],
        distinguishing_terms=["Factor M"],
        required_bridge=(
            "Factor M moderates the relation between descriptor D and outcome O."
        ),
        observable="outcome O",
        predicted_observation="Outcome O changes with descriptor D.",
        falsification_condition="Outcome O does not change with descriptor D.",
        prediction_observation_id="prediction:1",
        falsification_criterion_id="falsifier:1",
        search_concepts=[],
        search_queries=["descriptor D outcome O"],
        scientific_structure=NoveltyClaimScientificStructure(),
        scientific_structure_reason_codes=[],
    )


def _row(
    *,
    claim_id: str = "claim:1",
    claim_rank: int = 1,
    role: str | None = "NOVELTY_BEARING",
    source_status: str = "READY",
    semantic_status: str = "PASS",
    specification_status: str = "COMPLETE",
    atomic_kind_status: str = "SUPPORTED",
    compilation_status: str = "COMPILED_SHADOW",
    semantic_reasons: list[str] | None = None,
    specification_reasons: list[str] | None = None,
) -> LegacyAtomicSpecificationShadowRow:
    compiled = compilation_status == "COMPILED_SHADOW"
    return LegacyAtomicSpecificationShadowRow(
        hypothesis_id="hypothesis:h1",
        claim_id=claim_id,
        claim_rank=claim_rank,
        claim_local_id=f"local:{claim_rank}",
        kind="moderator_interaction",
        novelty_selection_role=role,
        source_reference_status=source_status,
        proposition_fidelity_status=(
            "PASS" if semantic_status == "PASS" else "INVALID"
        ),
        bridge_fidelity_status=(
            "PASS" if semantic_status == "PASS" else "INVALID"
        ),
        semantic_fidelity_status=semantic_status,
        specification_status=specification_status,
        atomic_kind_status=atomic_kind_status,
        compilation_status=compilation_status,
        source_reference_reason_codes=(
            [] if source_status == "READY" else ["source_reference_problem"]
        ),
        proposition_fidelity_reason_codes=(
            []
            if semantic_status == "PASS"
            else ["atomic_claim_direction_qualifier_not_preserved"]
        ),
        bridge_fidelity_reason_codes=[],
        unclassified_semantic_reason_codes=[],
        specification_reason_codes=specification_reasons or [],
        semantic_fidelity_reason_codes=semantic_reasons or (
            []
            if semantic_status == "PASS"
            else ["atomic_claim_direction_qualifier_not_preserved"]
        ),
        semantic_fidelity_taxonomy_shadow={},
        prediction_observation_id="prediction:1",
        falsification_criterion_id="falsifier:1",
        source_observable="outcome O",
        specification=_spec(claim_id) if compiled else None,
    )


def _report(rows: list[LegacyAtomicSpecificationShadowRow]):
    body = {
        "schema_version": (
            "legacy-atomic-scientific-specification-shadow-report-v1"
        ),
        "hypothesis_id": "hypothesis:h1",
        "row_count": len(rows),
        "compiled_count": sum(
            row.compilation_status == "COMPILED_SHADOW"
            for row in rows
        ),
        "source_ready_count": sum(
            row.source_reference_status == "READY"
            for row in rows
        ),
        "semantic_pass_count": sum(
            row.semantic_fidelity_status == "PASS"
            for row in rows
        ),
        "semantic_review_count": sum(
            row.semantic_fidelity_status == "REVIEW"
            for row in rows
        ),
        "semantic_invalid_count": sum(
            row.semantic_fidelity_status == "INVALID"
            for row in rows
        ),
        "specification_complete_count": sum(
            row.specification_status == "COMPLETE"
            for row in rows
        ),
        "rows": [row.model_dump(mode="json") for row in rows],
        "diagnostic_only": True,
        "production_authority": False,
        "canonical_claim_mutated": False,
        "query_plan_mutated": False,
        "vpre_contract_changed": False,
        "retrieval_performed": False,
        "novelty_assessment_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "production_selection_changed": False,
    }
    digest = _sha(body)
    return LegacyAtomicSpecificationShadowReport(
        **body,
        report_id="legacy_atomic_specification_shadow:" + digest[:20],
        report_sha256=digest,
    )


def test_ready_requires_all_neutral_dimensions_to_pass():
    row = assess_atomic_pre_n10_shadow_row(_row())

    assert row.readiness_status == "READY_FOR_N10_SHADOW"
    assert row.blocking_dimensions == []
    assert row.compiled_specification_present is True
    assert row.production_authority is False
    assert row.router_authority is False


def test_source_ready_does_not_override_semantic_invalid():
    row = assess_atomic_pre_n10_shadow_row(
        _row(
            semantic_status="INVALID",
            semantic_reasons=[
                "atomic_claim_direction_qualifier_not_preserved",
                "required_bridge_relation_endpoint_alignment_unresolved",
            ],
        )
    )

    assert row.source_reference_status == "READY"
    assert row.semantic_fidelity_status == "INVALID"
    assert row.specification_status == "COMPLETE"
    assert row.atomic_kind_status == "SUPPORTED"
    assert row.structural_compilation_status == "COMPILED_SHADOW"
    assert row.blocking_dimensions == ["SEMANTIC_FIDELITY"]
    assert row.readiness_status == "NOT_READY_FOR_N10_SHADOW"


def test_independent_blockers_remain_distinct():
    row = assess_atomic_pre_n10_shadow_row(
        _row(
            role=None,
            source_status="INVALID",
            semantic_status="REVIEW",
            specification_status="INCOMPLETE",
            atomic_kind_status="UNSUPPORTED",
            compilation_status="ABSTAINED_SOURCE_REFERENCE",
            specification_reasons=["missing_required_bridge"],
        )
    )

    assert row.blocking_dimensions == [
        "SOURCE_REFERENCE",
        "SEMANTIC_FIDELITY",
        "SPECIFICATION",
        "ATOMIC_KIND",
        "NOVELTY_ROLE",
        "STRUCTURAL_COMPILATION",
    ]
    assert row.readiness_status == "NOT_READY_FOR_N10_SHADOW"


def test_report_requires_all_claims_ready_and_one_novelty_bearing_ready():
    ready_novelty = _row()
    blocked_supporting = _row(
        claim_id="claim:2",
        claim_rank=2,
        role="REQUIRED_ENABLING_RELATION",
        semantic_status="INVALID",
    )
    report = compile_atomic_pre_n10_admissibility_shadow(
        _report([ready_novelty, blocked_supporting])
    )

    assert report.claim_count == 2
    assert report.ready_claim_count == 1
    assert report.novelty_bearing_ready_claim_count == 1
    assert report.disposition == "INTERVENTION_REQUIRED_SHADOW"
    assert report.blocker_dimension_counts == {
        "SEMANTIC_FIDELITY": 1
    }
    assert report.exact_text_reconstruction_used_for_readiness is False
    assert report.vpre_contract_changed is False
    assert report.n10_performed is False
