from __future__ import annotations

import pytest

from dac_her.cross_context_trend import (
    CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
    CrossContextTrendAssessment,
    PairwiseTrendContrast,
    TrendContextDimension,
    TrendContextProfile,
    audit_cross_context_trends,
    classify_direction_relation,
    classify_evidence_kind_relation,
    stable_cross_context_assessment_id,
    stable_pairwise_trend_contrast_id,
    stable_trend_context_profile_id,
    stable_trend_relation_id,
)
from dac_her.trend_precision import PaperLocalTrendResult


CONTEXT_SEMANTICS = "test_context_v1"
ASSESSMENT_SEMANTICS = "test_assessment_v1"


def _result(
    result_id: str,
    paper_id: str,
    direction: str,
    *,
    shape: str = "monotonic",
    evidence_kinds: tuple[str, ...] = ("reported_claim",),
) -> PaperLocalTrendResult:
    return PaperLocalTrendResult(
        result_id=result_id,
        paper_id=paper_id,
        domain_profile_id="test_domain",
        trend_semantics_id="trend_test_v1",
        precision_semantics_id="precision_test_v1",
        result_lane=(
            "numeric"
            if evidence_kinds != ("reported_claim",)
            else "claim"
        ),
        independent_variable_key="shell_thickness",
        dependent_observable_key="raman_intensity",
        direction=direction,
        shape=shape,
        control_family="structural",
        observable_semantics="measured_signal_intensity",
        member_trend_ids=(f"{result_id}:member",),
        evidence_kinds=evidence_kinds,
    )


def _relation_id() -> str:
    return stable_trend_relation_id(
        independent_variable_key="shell_thickness",
        dependent_observable_key="raman_intensity",
        control_family="structural",
        observable_semantics="measured_signal_intensity",
    )


def _dimensions(
    *,
    substrate: str,
) -> tuple[TrendContextDimension, ...]:
    return (
        TrendContextDimension(
            name="substrate_condition",
            status="known",
            normalized_value=substrate,
            source_values=(substrate,),
            source_node_ids=(f"node:{substrate}",),
        ),
        TrendContextDimension(
            name="excitation_wavelength",
            status="unknown",
        ),
        TrendContextDimension(
            name="shell_thickness",
            status="varied_control",
        ),
    )


def _profile(
    result: PaperLocalTrendResult,
    *,
    substrate: str,
) -> TrendContextProfile:
    relation_id = _relation_id()
    return TrendContextProfile(
        context_profile_id=stable_trend_context_profile_id(
            context_semantics_id=CONTEXT_SEMANTICS,
            local_result_id=result.result_id,
        ),
        domain_profile_id=result.domain_profile_id,
        contract_semantics_id=
            CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
        context_semantics_id=CONTEXT_SEMANTICS,
        local_result_id=result.result_id,
        paper_id=result.paper_id,
        relation_id=relation_id,
        independent_variable_key=result.independent_variable_key,
        dependent_observable_key=result.dependent_observable_key,
        control_family=result.control_family,
        observable_semantics=result.observable_semantics,
        result_lane=result.result_lane,
        direction=result.direction,
        shape=result.shape,
        evidence_kinds=result.evidence_kinds,
        member_trend_ids=result.member_trend_ids,
        dimensions=_dimensions(substrate=substrate),
    )


def _contrast(
    left: TrendContextProfile,
    right: TrendContextProfile,
    *,
    context_relation: str = "same_context",
    mismatched: tuple[str, ...] = (),
    differentiating: tuple[str, ...] = (),
) -> PairwiseTrendContrast:
    return PairwiseTrendContrast(
        contrast_id=stable_pairwise_trend_contrast_id(
            assessment_semantics_id=ASSESSMENT_SEMANTICS,
            relation_id=left.relation_id,
            left_result_id=left.local_result_id,
            right_result_id=right.local_result_id,
        ),
        contract_semantics_id=
            CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
        assessment_semantics_id=ASSESSMENT_SEMANTICS,
        relation_id=left.relation_id,
        left_context_profile_id=left.context_profile_id,
        right_context_profile_id=right.context_profile_id,
        left_result_id=left.local_result_id,
        right_result_id=right.local_result_id,
        left_paper_id=left.paper_id,
        right_paper_id=right.paper_id,
        direction_relation=classify_direction_relation(
            left.direction,
            right.direction,
        ),
        shape_relation=(
            "same_shape"
            if left.shape == right.shape
            else "different_shape"
        ),
        evidence_kind_relation=classify_evidence_kind_relation(
            left.evidence_kinds,
            right.evidence_kinds,
        ),
        context_relation=context_relation,
        matched_dimensions=(
            ("substrate_condition",)
            if context_relation == "same_context"
            else ()
        ),
        mismatched_dimensions=mismatched,
        unknown_dimensions=("excitation_wavelength",),
        varied_control_dimensions=("shell_thickness",),
        differentiating_dimensions=differentiating,
        reason_codes=("synthetic_pair",),
    )


def _assessment(
    results: list[PaperLocalTrendResult],
    contrasts: list[PairwiseTrendContrast],
    *,
    status: str,
    repeated: tuple[str, ...] = (),
    reversal: tuple[str, ...] = (),
    context_specific: tuple[str, ...] = (),
    unresolved: tuple[str, ...] = (),
) -> CrossContextTrendAssessment:
    relation_id = _relation_id()
    direction_buckets = {
        "positive_result_ids": tuple(
            row.result_id
            for row in results
            if row.direction == "positive"
        ),
        "negative_result_ids": tuple(
            row.result_id
            for row in results
            if row.direction == "negative"
        ),
        "non_monotonic_result_ids": tuple(
            row.result_id
            for row in results
            if row.direction == "non_monotonic"
        ),
        "unchanged_result_ids": tuple(
            row.result_id
            for row in results
            if row.direction == "unchanged"
        ),
        "unspecified_result_ids": tuple(
            row.result_id
            for row in results
            if row.direction not in {
                "positive",
                "negative",
                "non_monotonic",
                "unchanged",
            }
        ),
    }
    return CrossContextTrendAssessment(
        assessment_id=stable_cross_context_assessment_id(
            assessment_semantics_id=ASSESSMENT_SEMANTICS,
            relation_id=relation_id,
            member_result_ids=(
                row.result_id for row in results
            ),
        ),
        domain_profile_id="test_domain",
        contract_semantics_id=
            CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
        assessment_semantics_id=ASSESSMENT_SEMANTICS,
        relation_id=relation_id,
        independent_variable_key="shell_thickness",
        dependent_observable_key="raman_intensity",
        control_family="structural",
        observable_semantics="measured_signal_intensity",
        member_result_ids=tuple(
            row.result_id for row in results
        ),
        paper_ids=tuple(
            sorted({row.paper_id for row in results})
        ),
        pairwise_contrast_ids=tuple(
            row.contrast_id for row in contrasts
        ),
        status=status,
        **direction_buckets,
        experimental_numeric_result_ids=tuple(
            row.result_id
            for row in results
            if "experimental_numeric" in row.evidence_kinds
        ),
        calculated_numeric_result_ids=tuple(
            row.result_id
            for row in results
            if "calculated_numeric" in row.evidence_kinds
        ),
        reported_claim_result_ids=tuple(
            row.result_id
            for row in results
            if "reported_claim" in row.evidence_kinds
        ),
        repeated_pair_ids=repeated,
        reversal_pair_ids=reversal,
        context_specific_pair_ids=context_specific,
        unresolved_pair_ids=unresolved,
        reason_codes=(f"status:{status}",),
    )


def test_relation_identity_excludes_direction_and_shape():
    first = _result("r1", "P1", "positive", shape="monotonic")
    second = _result(
        "r2",
        "P2",
        "negative",
        shape="single_optimum",
    )
    left = _profile(first, substrate="AuAg")
    right = _profile(second, substrate="AuAg")
    assert left.relation_id == right.relation_id


def test_direction_relation_positive_negative_is_strict_reversal():
    assert (
        classify_direction_relation("positive", "negative")
        == "opposite_direction"
    )
    assert (
        classify_direction_relation(
            "positive",
            "non_monotonic",
        )
        == "monotonic_vs_non_monotonic"
    )


def test_pairwise_contrast_is_cross_paper_only():
    left = _profile(
        _result("r1", "P1", "positive"),
        substrate="AuAg",
    )
    right = _profile(
        _result("r2", "P1", "positive"),
        substrate="AuAg",
    )
    with pytest.raises(ValueError, match="cross-paper only"):
        _contrast(left, right)


def test_context_dimension_buckets_are_disjoint():
    left = _profile(
        _result("r1", "P1", "positive"),
        substrate="AuAg",
    )
    right = _profile(
        _result("r2", "P2", "positive"),
        substrate="AuAg",
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        PairwiseTrendContrast(
            contrast_id=stable_pairwise_trend_contrast_id(
                assessment_semantics_id=ASSESSMENT_SEMANTICS,
                relation_id=left.relation_id,
                left_result_id=left.local_result_id,
                right_result_id=right.local_result_id,
            ),
            contract_semantics_id=
                CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
            assessment_semantics_id=ASSESSMENT_SEMANTICS,
            relation_id=left.relation_id,
            left_context_profile_id=left.context_profile_id,
            right_context_profile_id=right.context_profile_id,
            left_result_id=left.local_result_id,
            right_result_id=right.local_result_id,
            left_paper_id=left.paper_id,
            right_paper_id=right.paper_id,
            direction_relation="same_direction",
            shape_relation="same_shape",
            evidence_kind_relation="same_kind",
            context_relation="same_context",
            matched_dimensions=("shell_thickness",),
            varied_control_dimensions=("shell_thickness",),
        )


def test_valid_cross_paper_repeated_assessment_passes_audit():
    a = _result("a", "P1", "positive")
    b = _result("b", "P2", "positive")
    pa = _profile(a, substrate="AuAg")
    pb = _profile(b, substrate="AuAg")
    pair = _contrast(pa, pb)
    assessment = _assessment(
        [a, b],
        [pair],
        status="repeated",
        repeated=(pair.contrast_id,),
    )

    audit = audit_cross_context_trends(
        local_results=[a, b],
        profiles=[pa, pb],
        contrasts=[pair],
        assessments=[assessment],
    )
    assert audit.structural_gate is True
    assert audit.status_counts == {"repeated": 1}


def test_positive_positive_negative_cannot_be_majority_voted_repeated():
    a = _result("a", "P1", "positive")
    b = _result("b", "P2", "positive")
    c = _result("c", "P3", "negative")
    pa = _profile(a, substrate="AuAg")
    pb = _profile(b, substrate="AuAg")
    pc = _profile(c, substrate="AuAg")

    ab = _contrast(pa, pb)
    ac = _contrast(pa, pc)
    bc = _contrast(pb, pc)

    with pytest.raises(
        ValueError,
        match="Any reversal pair requires status='reversed'",
    ):
        _assessment(
            [a, b, c],
            [ab, ac, bc],
            status="repeated",
            repeated=(ab.contrast_id,),
            reversal=(ac.contrast_id, bc.contrast_id),
        )


def test_positive_positive_negative_reversed_assessment_passes_audit():
    a = _result("a", "P1", "positive")
    b = _result("b", "P2", "positive")
    c = _result("c", "P3", "negative")
    pa = _profile(a, substrate="AuAg")
    pb = _profile(b, substrate="AuAg")
    pc = _profile(c, substrate="AuAg")
    ab = _contrast(pa, pb)
    ac = _contrast(pa, pc)
    bc = _contrast(pb, pc)

    assessment = _assessment(
        [a, b, c],
        [ab, ac, bc],
        status="reversed",
        repeated=(ab.contrast_id,),
        reversal=(ac.contrast_id, bc.contrast_id),
    )
    audit = audit_cross_context_trends(
        local_results=[a, b, c],
        profiles=[pa, pb, pc],
        contrasts=[ab, ac, bc],
        assessments=[assessment],
    )
    assert audit.structural_gate is True
    assert audit.status_counts == {"reversed": 1}


def test_single_paper_result_is_insufficient_not_repeated():
    result = _result("a", "P1", "positive")
    profile = _profile(result, substrate="AuAg")
    assessment = _assessment(
        [result],
        [],
        status="insufficient",
    )
    audit = audit_cross_context_trends(
        local_results=[result],
        profiles=[profile],
        contrasts=[],
        assessments=[assessment],
    )
    assert audit.structural_gate is True
    assert audit.status_counts == {"insufficient": 1}


def test_evidence_modality_is_preserved_not_collapsed_to_support_count():
    a = _result(
        "a",
        "P1",
        "positive",
        evidence_kinds=("experimental_numeric",),
    )
    b = _result(
        "b",
        "P2",
        "positive",
        evidence_kinds=("reported_claim",),
    )
    pa = _profile(a, substrate="AuAg")
    pb = _profile(b, substrate="AuAg")
    pair = _contrast(pa, pb)
    assert pair.evidence_kind_relation == "cross_kind"

    assessment = _assessment(
        [a, b],
        [pair],
        status="repeated",
        repeated=(pair.contrast_id,),
    )
    audit = audit_cross_context_trends(
        local_results=[a, b],
        profiles=[pa, pb],
        contrasts=[pair],
        assessments=[assessment],
    )
    assert audit.structural_gate is True
    assert assessment.experimental_numeric_result_ids == ("a",)
    assert assessment.reported_claim_result_ids == ("b",)
