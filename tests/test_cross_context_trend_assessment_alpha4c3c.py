from __future__ import annotations

from dataclasses import replace

import pytest

from dac_her.cross_context_trend import (
    CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
    TrendContextDimension,
    TrendContextProfile,
    stable_trend_context_profile_id,
    stable_trend_relation_id,
)
from dac_her.cross_context_trend_assessment import (
    CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID,
    audit_deterministic_cross_context_assessments,
    build_deterministic_cross_context_assessments,
    build_pairwise_trend_contrast,
    classify_pair_role,
)
from dac_her.trend_precision import PaperLocalTrendResult


CONTEXT_SEMANTICS = "sers_au_ag_trend_context_v1_alpha4c3b"


def _relation_id() -> str:
    return stable_trend_relation_id(
        independent_variable_key="shell_thickness",
        dependent_observable_key="raman_intensity",
        control_family="structural",
        observable_semantics="measured_signal_intensity",
    )


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
        domain_profile_id="sers_au_ag",
        trend_semantics_id="sers_au_ag_trend_v5_alpha4c2121",
        precision_semantics_id=
            "sers_au_ag_trend_precision_v5_alpha4c21211",
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


def _dimension(
    name: str,
    *,
    status: str,
    value: str = "",
) -> TrendContextDimension:
    kwargs = {
        "name": name,
        "status": status,
    }
    if status == "known":
        kwargs.update({
            "normalized_value": value,
            "source_values": (value,),
            "source_node_ids": (f"node:{name}:{value}",),
        })
    elif status == "ambiguous":
        kwargs.update({
            "source_values": ("a", "b"),
            "source_node_ids": (f"node:{name}:ambiguous",),
        })
    elif status == "varied_control":
        kwargs.update({
            "provenance_scopes": (
                "trend_independent_variable",
            ),
        })
    return TrendContextDimension(**kwargs)


def _profile(
    result: PaperLocalTrendResult,
    *,
    analyte: tuple[str, str] = ("known", "atp"),
    excitation: tuple[str, str] = ("known", "532 nm"),
    substrate: tuple[str, str] = ("known", "AuAg"),
) -> TrendContextProfile:
    dimensions = (
        _dimension(
            "analyte",
            status=analyte[0],
            value=analyte[1],
        ),
        _dimension(
            "excitation_wavelength",
            status=excitation[0],
            value=excitation[1],
        ),
        _dimension(
            "shell_thickness",
            status="varied_control",
        ),
        _dimension(
            "substrate_condition",
            status=substrate[0],
            value=substrate[1],
        ),
    )
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
        relation_id=_relation_id(),
        independent_variable_key=result.independent_variable_key,
        dependent_observable_key=result.dependent_observable_key,
        control_family=result.control_family,
        observable_semantics=result.observable_semantics,
        result_lane=result.result_lane,
        direction=result.direction,
        shape=result.shape,
        evidence_kinds=result.evidence_kinds,
        member_trend_ids=result.member_trend_ids,
        dimensions=dimensions,
    )


def test_same_direction_cross_paper_pair_is_repeated():
    a = _profile(_result("a", "P1", "positive"))
    b = _profile(_result("b", "P2", "positive"))

    contrast = build_pairwise_trend_contrast(a, b)

    assert contrast.direction_relation == "same_direction"
    assert contrast.context_relation == "same_context"
    assert classify_pair_role(contrast) == "repeated"


def test_positive_negative_is_reversal_even_when_context_unknown():
    a = _profile(
        _result("a", "P1", "positive"),
        analyte=("unknown", ""),
        excitation=("unknown", ""),
        substrate=("unknown", ""),
    )
    b = _profile(
        _result("b", "P2", "negative"),
        analyte=("unknown", ""),
        excitation=("unknown", ""),
        substrate=("unknown", ""),
    )

    contrast = build_pairwise_trend_contrast(a, b)

    assert contrast.direction_relation == "opposite_direction"
    assert contrast.context_relation == "context_unknown"
    assert classify_pair_role(contrast) == "reversal"


def test_positive_non_monotonic_with_known_context_difference_is_context_specific():
    a = _profile(
        _result("a", "P1", "positive"),
        substrate=("known", "nanocube"),
    )
    b = _profile(
        _result(
            "b",
            "P2",
            "non_monotonic",
            shape="single_optimum",
        ),
        substrate=("known", "nanobox"),
    )

    contrast = build_pairwise_trend_contrast(a, b)

    assert (
        contrast.direction_relation
        == "monotonic_vs_non_monotonic"
    )
    assert contrast.context_relation == "context_different"
    assert contrast.mismatched_dimensions == (
        "substrate_condition",
    )
    assert contrast.differentiating_dimensions == (
        "substrate_condition",
    )
    assert classify_pair_role(contrast) == "context_specific"


def test_positive_non_monotonic_without_known_context_difference_is_unresolved():
    a = _profile(
        _result("a", "P1", "positive"),
        analyte=("unknown", ""),
        excitation=("unknown", ""),
        substrate=("unknown", ""),
    )
    b = _profile(
        _result(
            "b",
            "P2",
            "non_monotonic",
            shape="single_optimum",
        ),
        analyte=("unknown", ""),
        excitation=("unknown", ""),
        substrate=("unknown", ""),
    )

    contrast = build_pairwise_trend_contrast(a, b)

    assert contrast.context_relation == "context_unknown"
    assert classify_pair_role(contrast) == "unresolved"


def test_varied_control_is_partitioned_not_mismatched():
    a = _profile(_result("a", "P1", "positive"))
    b = _profile(_result("b", "P2", "positive"))

    contrast = build_pairwise_trend_contrast(a, b)

    assert contrast.varied_control_dimensions == (
        "shell_thickness",
    )
    assert "shell_thickness" not in (
        contrast.mismatched_dimensions
    )


def test_same_relation_varied_control_mask_disagreement_fails_closed():
    a = _profile(_result("a", "P1", "positive"))
    b = _profile(_result("b", "P2", "positive"))
    dims = tuple(
        _dimension(
            item.name,
            status=(
                "known"
                if item.name == "shell_thickness"
                else item.status
            ),
            value=(
                "8.4 nm"
                if item.name == "shell_thickness"
                else item.normalized_value
            ),
        )
        for item in b.dimensions
    )
    broken = replace(b, dimensions=dims)

    with pytest.raises(
        ValueError,
        match="disagree on varied_control mask",
    ):
        build_pairwise_trend_contrast(a, broken)


def test_positive_positive_negative_relation_is_reversed_not_majority_repeated():
    results = [
        _result("a", "P1", "positive"),
        _result("b", "P2", "positive"),
        _result("c", "P3", "negative"),
    ]
    profiles = [_profile(result) for result in results]

    contrasts, assessments = (
        build_deterministic_cross_context_assessments(
            profiles
        )
    )

    assert len(contrasts) == 3
    assert len(assessments) == 1
    assessment = assessments[0]
    assert assessment.status == "reversed"
    assert len(assessment.repeated_pair_ids) == 1
    assert len(assessment.reversal_pair_ids) == 2


def test_context_specific_has_priority_over_repeated_when_no_reversal():
    results = [
        _result("a", "P1", "positive"),
        _result("b", "P2", "positive"),
        _result(
            "c",
            "P3",
            "non_monotonic",
            shape="single_optimum",
        ),
    ]
    profiles = [
        _profile(
            results[0],
            substrate=("known", "nanocube"),
        ),
        _profile(
            results[1],
            substrate=("known", "nanocube"),
        ),
        _profile(
            results[2],
            substrate=("known", "nanobox"),
        ),
    ]

    _, assessments = (
        build_deterministic_cross_context_assessments(
            profiles
        )
    )

    assert assessments[0].status == "context_specific"
    assert assessments[0].repeated_pair_ids
    assert assessments[0].context_specific_pair_ids


def test_same_paper_multiple_results_produce_no_pair_and_insufficient():
    results = [
        _result(
            "a",
            "P1",
            "positive",
            evidence_kinds=("experimental_numeric",),
        ),
        _result(
            "b",
            "P1",
            "positive",
            shape="saturating",
            evidence_kinds=("reported_claim",),
        ),
    ]
    profiles = [_profile(result) for result in results]

    contrasts, assessments = (
        build_deterministic_cross_context_assessments(
            profiles
        )
    )

    assert contrasts == []
    assert len(assessments) == 1
    assessment = assessments[0]
    assert assessment.status == "insufficient"
    assert assessment.paper_ids == ("P1",)
    assert assessment.pairwise_contrast_ids == ()
    assert (
        "same_paper_support_not_replication"
        in assessment.reason_codes
    )


def test_cross_kind_repetition_preserves_modality_buckets():
    numeric = _result(
        "a",
        "P1",
        "positive",
        evidence_kinds=("experimental_numeric",),
    )
    claim = _result(
        "b",
        "P2",
        "positive",
        evidence_kinds=("reported_claim",),
    )
    profiles = [_profile(numeric), _profile(claim)]

    contrasts, assessments = (
        build_deterministic_cross_context_assessments(
            profiles
        )
    )

    assert contrasts[0].evidence_kind_relation == "cross_kind"
    assessment = assessments[0]
    assert assessment.status == "repeated"
    assert assessment.experimental_numeric_result_ids == ("a",)
    assert assessment.reported_claim_result_ids == ("b",)


def test_deterministic_audit_requires_complete_cross_paper_pair_set():
    results = [
        _result("a", "P1", "positive"),
        _result("b", "P2", "positive"),
        _result("c", "P3", "positive"),
    ]
    profiles = [_profile(result) for result in results]
    contrasts, assessments = (
        build_deterministic_cross_context_assessments(
            profiles
        )
    )

    audit = audit_deterministic_cross_context_assessments(
        local_results=results,
        profiles=profiles,
        contrasts=contrasts,
        assessments=assessments,
    )
    assert audit.structural_gate is True
    assert audit.expected_cross_paper_pair_count == 3

    broken_contrasts = contrasts[:-1]
    broken_assessment = replace(
        assessments[0],
        pairwise_contrast_ids=tuple(
            row.contrast_id
            for row in broken_contrasts
        ),
        repeated_pair_ids=tuple(
            row.contrast_id
            for row in broken_contrasts
        ),
    )
    broken_audit = (
        audit_deterministic_cross_context_assessments(
            local_results=results,
            profiles=profiles,
            contrasts=broken_contrasts,
            assessments=[broken_assessment],
        )
    )
    assert broken_audit.structural_gate is False
    assert (
        "cross_paper_pair_completeness_mismatch"
        in broken_audit.issues
    )


def test_assessment_semantics_id_is_frozen():
    assert CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID == (
        "cross_context_trend_assessment_v1_alpha4c3c"
    )
