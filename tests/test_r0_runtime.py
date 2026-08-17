from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from dac_her.external_novelty_contracts import ClaimPriorArtReview, ClaimSearchCoverage, PriorArtMatch
from dac_her.novelty_refinement_contracts import NoveltyGap
from dac_her.r0_contracts import R0ActionDecision
from dac_her.r0_runtime import (
    compile_r0_refinement_basis,
    compile_targeted_evidence_assessment,
    derive_r0_refinement_bases,
    decide_r0,
    validate_r0_action_decision,
    validate_r0_decision_against_sources,
    validate_targeted_evidence_assessment,
)

PLAN_ID = "novelty_gap_plan:test"
PLAN_SHA = "1" * 64
T1_RUN_ID = "sers_targeted_retrieval_t1_live_v2:test"
T1_FREEZE_ID = "sers_targeted_retrieval_t1_final_freeze_v2:test"
T1_SHA = "2" * 64
HID = "hypothesis:h1"


def _gap(action: str, claim_ids: list[str] | None = None, *, hid: str = HID) -> NoveltyGap:
    return NoveltyGap(
        gap_id="gap:g1",
        hypothesis_id=hid,
        source_external_status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        action=action,
        target_claim_ids=claim_ids or [],
        differentiator="synthetic differentiator",
        already_known_boundary=[],
        unresolved_boundary=[],
        contextual_conflict_work_ids=[],
        targeted_queries=[],
        reason_codes=[],
    )


def _match(
    work_id: str,
    relationship: str,
    *,
    scope_compatible_for_conflict: bool = True,
) -> PriorArtMatch:
    return PriorArtMatch(
        work_id=work_id,
        relationship=relationship,
        confidence=0.9,
        rationale="synthetic evidence relationship",
        relevance_score=0.9,
        semantic_similarity=0.9,
        lexical_coverage=0.9,
        reaction_domain_relevance=0.9,
        catalyst_scope_relevance=0.9,
        scope_compatible_for_conflict=scope_compatible_for_conflict,
        scope_reason_codes=[],
        title=f"Synthetic {work_id}",
        abstract_available=True,
    )


def _review(
    claim_id: str,
    status: str,
    *,
    hid: str = HID,
    unique_work_count: int = 1,
    abstract_work_count: int = 1,
    coverage_claim_id: str | None = None,
    work_id: str | None = None,
    relationship_override: str | None = None,
    scope_compatible_for_conflict: bool = True,
) -> ClaimPriorArtReview:
    relationship_by_status = {
        "DIRECT_PRIOR_ART": "DIRECT_PRIOR_ART",
        "PARTIAL_PRIOR_ART": "PARTIAL_PRIOR_ART",
        "COMPONENTS_ONLY": "COMPONENT_ONLY",
        "CONFLICTING_PRIOR_ART": "CONFLICTING_PRIOR_ART",
    }
    relationship = (
        relationship_override
        if relationship_override is not None
        else relationship_by_status.get(status)
    )
    resolved_work_id = work_id or f"work:{claim_id}"
    matches = (
        [
            _match(
                resolved_work_id,
                relationship,
                scope_compatible_for_conflict=scope_compatible_for_conflict,
            )
        ]
        if relationship
        else []
    )
    return ClaimPriorArtReview(
        hypothesis_id=hid,
        claim_id=claim_id,
        claim_text=f"synthetic {claim_id}",
        importance="core",
        status=status,
        matches=matches,
        coverage=ClaimSearchCoverage(
            claim_id=coverage_claim_id or claim_id,
            query_count=1,
            successful_query_count=1,
            unique_work_count=unique_work_count,
            abstract_work_count=abstract_work_count,
            reviewed_work_count=len(matches),
        ),
        reason_codes=[],
        reviewer_unknown_work_ids=[],
        interpretation="synthetic review",
    )


def _basis(review: ClaimPriorArtReview, basis_kind: str):
    assert review.matches
    return compile_r0_refinement_basis(
        review,
        basis_kind=basis_kind,
        source_work_ids=[review.matches[0].work_id],
    )


def _assessment(gap: NoveltyGap, reviews: list[ClaimPriorArtReview]):
    return compile_targeted_evidence_assessment(
        gap,
        reviews,
        source_gap_plan_id=PLAN_ID,
        source_gap_plan_sha256=PLAN_SHA,
        source_t1_run_id=T1_RUN_ID,
        source_t1_freeze_id=T1_FREEZE_ID,
        source_t1_manifest_sha256=T1_SHA,
    )


def _decision(gap: NoveltyGap, assessment=None):
    return decide_r0(
        gap,
        source_gap_plan_id=PLAN_ID,
        source_gap_plan_sha256=PLAN_SHA,
        assessment=assessment,
    )


@pytest.mark.parametrize(
    ("status", "state"),
    [
        ("DIRECT_PRIOR_ART", "directly_covered"),
        ("PARTIAL_PRIOR_ART", "partially_covered"),
        ("COMPONENTS_ONLY", "relational_gap_remains"),
        ("CONFLICTING_PRIOR_ART", "conflicted"),
        ("NO_DIRECT_MATCH_FOUND", "unresolved"),
        ("INSUFFICIENT_METADATA", "unresolved"),
        ("TITLE_ONLY_NEIGHBORS", "unresolved"),
    ],
)
def test_targeted_search_only_never_authorizes_r1(status: str, state: str) -> None:
    gap = _gap("targeted_search_only", ["claim:c1"])
    assessment = _assessment(gap, [_review("claim:c1", status)])
    decision = _decision(gap, assessment)
    assert assessment.evidence_state == state
    assert assessment.refinement_bases == []
    assert decision.route == "pass_original_to_r2"
    assert decision.r1_authorized is False
    assert decision.max_refinements_authorized == 0
    assert decision.r2_required is True
    assert decision.network_calls == 0
    assert decision.llm_calls == 0
    assert decision.hypothesis_rewritten is False
    assert decision.automatic_next_stage_authorized is False


def test_assessment_compiler_has_no_caller_controlled_refinement_basis_parameter() -> None:
    signature = inspect.signature(compile_targeted_evidence_assessment)
    assert "refinement_bases" not in signature.parameters


def test_targeted_search_only_derives_no_refinement_basis() -> None:
    gap = _gap("targeted_search_only", ["claim:c1"])
    review = _review("claim:c1", "PARTIAL_PRIOR_ART")
    assessment = _assessment(gap, [review])
    assert assessment.refinement_bases == []
    assert derive_r0_refinement_bases(gap, [review]) == []
    assert _decision(gap, assessment).r1_authorized is False


def test_targeted_search_then_refine_derives_boundary_basis_automatically() -> None:
    gap = _gap("targeted_search_then_refine", ["claim:c1"])
    review = _review("claim:c1", "PARTIAL_PRIOR_ART")
    assessment = _assessment(gap, [review])
    assert len(assessment.refinement_bases) == 1
    basis = assessment.refinement_bases[0]
    assert basis.basis_kind == "qualify_partial_relation"
    assert basis.epistemic_usage == "boundary_constraint_only_not_positive_premise"
    decision = _decision(gap, assessment)
    assert decision.route == "authorize_one_r1_then_r2"
    assert decision.r1_authorized is True
    assert decision.max_refinements_authorized == 1
    assert "R0D_BOUNDARY_EVIDENCE_BOUNDED_REFINEMENT" in decision.reason_codes
    assert all("POSITIVE" not in code for code in decision.reason_codes)


def test_partial_status_without_partial_match_fails_closed_without_r1() -> None:
    gap = _gap("targeted_search_then_refine", ["claim:c1"])
    review = _review(
        "claim:c1",
        "PARTIAL_PRIOR_ART",
        relationship_override="UNRELATED",
    )
    assessment = _assessment(gap, [review])
    assert assessment.evidence_state == "partially_covered"
    assert assessment.refinement_bases == []
    decision = _decision(gap, assessment)
    assert decision.route == "pass_original_to_r2"
    assert decision.r1_authorized is False
    assert "R0D_NO_BOUNDARY_REFINEMENT_BASIS" in decision.reason_codes


def test_component_basis_is_derived_and_authorizes_one_r1() -> None:
    gap = _gap("targeted_search_then_refine", ["claim:c1"])
    review = _review("claim:c1", "COMPONENTS_ONLY")
    assessment = _assessment(gap, [review])
    decision = _decision(gap, assessment)
    assert assessment.evidence_state == "relational_gap_remains"
    assert [row.basis_kind for row in assessment.refinement_bases] == [
        "separate_known_components"
    ]
    assert decision.route == "authorize_one_r1_then_r2"
    assert decision.max_refinements_authorized == 1


def test_no_direct_match_is_unresolved_not_novel_or_refinement_basis() -> None:
    gap = _gap("targeted_search_then_refine", ["claim:c1"])
    assessment = _assessment(gap, [_review("claim:c1", "NO_DIRECT_MATCH_FOUND")])
    decision = _decision(gap, assessment)
    assert assessment.evidence_state == "unresolved"
    assert assessment.literature_absence_claimed is False
    assert decision.route == "pass_original_to_r2"
    assert decision.r1_authorized is False
    assert "R0E_UNKNOWN_NOT_ABSENCE" in assessment.reason_codes


def test_review_limitation_dominates_positive_evidence() -> None:
    gap = _gap("targeted_search_then_refine", ["claim:c1", "claim:c2"])
    assessment = _assessment(
        gap,
        [
            _review("claim:c1", "PARTIAL_PRIOR_ART"),
            _review("claim:c2", "INSUFFICIENT_METADATA"),
        ],
    )
    assert assessment.evidence_state == "unresolved"
    assert _decision(gap, assessment).route == "pass_original_to_r2"


def test_post_targeted_conflict_is_deferred_to_r2_not_terminal_reject() -> None:
    gap = _gap("targeted_search_then_refine", ["claim:c1"])
    assessment = _assessment(gap, [_review("claim:c1", "CONFLICTING_PRIOR_ART")])
    decision = _decision(gap, assessment)
    assert assessment.evidence_state == "conflicted"
    assert decision.route == "pass_original_to_r2"
    assert decision.r2_required is True
    assert "R0D_POST_TARGETED_CONFLICT_DEFERRED_TO_R2" in decision.reason_codes


def test_refine_away_from_conflict_derives_scope_matched_basis() -> None:
    gap = _gap("refine_away_from_conflict", ["claim:c1"])
    review = _review("claim:c1", "CONFLICTING_PRIOR_ART")
    assessment = _assessment(gap, [review])
    decision = _decision(gap, assessment)
    assert [row.basis_kind for row in assessment.refinement_bases] == [
        "resolve_scope_matched_conflict"
    ]
    assert decision.route == "authorize_one_r1_then_r2"
    assert decision.max_refinements_authorized == 1


def test_conflict_basis_requires_scope_compatible_match() -> None:
    gap = _gap("refine_away_from_conflict", ["claim:c1"])
    review = _review(
        "claim:c1",
        "CONFLICTING_PRIOR_ART",
        scope_compatible_for_conflict=False,
    )
    assessment = _assessment(gap, [review])
    assert assessment.evidence_state == "conflicted"
    assert assessment.refinement_bases == []
    assert _decision(gap, assessment).route == "pass_original_to_r2"
    with pytest.raises(ValueError, match="scope-compatible"):
        _basis(review, "resolve_scope_matched_conflict")


def test_keep_and_reject_preserve_frozen_source_action() -> None:
    keep = _decision(_gap("keep"))
    assert keep.route == "pass_through_frozen"
    assert keep.r1_authorized is False
    assert keep.r2_required is True

    reject = _decision(_gap("reject"))
    assert reject.route == "terminal_reject"
    assert reject.r1_authorized is False
    assert reject.r2_required is False


def test_targeted_action_requires_exact_review_set() -> None:
    gap = _gap("targeted_search_only", ["claim:c1", "claim:c2"])
    with pytest.raises(ValueError, match="exactly match"):
        _assessment(gap, [_review("claim:c1", "DIRECT_PRIOR_ART")])


def test_review_hypothesis_and_coverage_lineage_are_fail_closed() -> None:
    gap = _gap("targeted_search_only", ["claim:c1"])
    with pytest.raises(ValueError, match="hypothesis_id drift"):
        _assessment(gap, [_review("claim:c1", "DIRECT_PRIOR_ART", hid="hypothesis:other")])
    with pytest.raises(ValueError, match="coverage claim_id mismatch"):
        _assessment(
            gap,
            [_review("claim:c1", "DIRECT_PRIOR_ART", coverage_claim_id="claim:other")],
        )


def test_input_order_is_canonical_and_deterministic() -> None:
    gap_a = _gap("targeted_search_then_refine", ["claim:c2", "claim:c1"])
    gap_b = _gap("targeted_search_then_refine", ["claim:c1", "claim:c2"])
    rows_a = [
        _review("claim:c2", "PARTIAL_PRIOR_ART"),
        _review("claim:c1", "DIRECT_PRIOR_ART"),
    ]
    rows_b = list(reversed(rows_a))
    a = _assessment(gap_a, rows_a)
    b = _assessment(gap_b, rows_b)
    assert a.assessment_sha256 == b.assessment_sha256
    da = _decision(gap_a, a)
    db = _decision(gap_b, b)
    assert da.route == db.route == "authorize_one_r1_then_r2"
    assert da.decision_sha256 == db.decision_sha256


def test_count_changes_do_not_change_scientific_route() -> None:
    gap = _gap("targeted_search_only", ["claim:c1"])
    a = _assessment(
        gap,
        [_review("claim:c1", "NO_DIRECT_MATCH_FOUND", unique_work_count=1, abstract_work_count=1)],
    )
    b = _assessment(
        gap,
        [_review("claim:c1", "NO_DIRECT_MATCH_FOUND", unique_work_count=100, abstract_work_count=100)],
    )
    assert a.evidence_state == b.evidence_state == "unresolved"
    assert _decision(gap, a).route == _decision(gap, b).route == "pass_original_to_r2"


def test_count_changes_do_not_change_auto_basis_or_r1_route() -> None:
    gap = _gap("targeted_search_then_refine", ["claim:c1"])
    low = _assessment(
        gap,
        [_review("claim:c1", "PARTIAL_PRIOR_ART", unique_work_count=1, abstract_work_count=1)],
    )
    high = _assessment(
        gap,
        [_review("claim:c1", "PARTIAL_PRIOR_ART", unique_work_count=100, abstract_work_count=100)],
    )
    assert [row.basis_id for row in low.refinement_bases] == [
        row.basis_id for row in high.refinement_bases
    ]
    assert _decision(gap, low).route == "authorize_one_r1_then_r2"
    assert _decision(gap, high).route == "authorize_one_r1_then_r2"


def test_id_renaming_does_not_change_route() -> None:
    gap_a = _gap("targeted_search_then_refine", ["claim:c1"], hid="hypothesis:a")
    gap_b = _gap("targeted_search_then_refine", ["claim:z9"], hid="hypothesis:b")
    review_a = _review("claim:c1", "PARTIAL_PRIOR_ART", hid="hypothesis:a")
    review_b = _review("claim:z9", "PARTIAL_PRIOR_ART", hid="hypothesis:b")
    a = _assessment(gap_a, [review_a])
    b = _assessment(gap_b, [review_b])
    assert _decision(gap_a, a).route == _decision(gap_b, b).route


def test_external_prior_art_is_never_promoted_to_positive_premise() -> None:
    gap = _gap("targeted_search_then_refine", ["claim:c1"])
    assessment = _assessment(gap, [_review("claim:c1", "PARTIAL_PRIOR_ART")])
    decision = _decision(gap, assessment)
    assert assessment.external_prior_art_can_be_positive_premise is False
    assert decision.external_prior_art_can_be_positive_premise is False
    assert all(
        row.epistemic_usage == "boundary_constraint_only_not_positive_premise"
        for row in assessment.refinement_bases
    )
    assert all("POSITIVE" not in code for code in assessment.reason_codes)
    assert all("POSITIVE" not in code for code in decision.reason_codes)


def test_assessment_validator_rejects_removed_auto_derived_basis() -> None:
    gap = _gap("targeted_search_then_refine", ["claim:c1"])
    assessment = _assessment(gap, [_review("claim:c1", "PARTIAL_PRIOR_ART")])
    bad = assessment.model_copy(update={"refinement_bases": []})
    # Re-hashing is deliberately omitted: either hash or deterministic-basis validation must fail.
    with pytest.raises(ValueError):
        validate_targeted_evidence_assessment(bad)


def test_hash_tampering_is_detected() -> None:
    gap = _gap("targeted_search_only", ["claim:c1"])
    assessment = _assessment(gap, [_review("claim:c1", "DIRECT_PRIOR_ART")])
    validate_targeted_evidence_assessment(assessment)
    bad_assessment = assessment.model_copy(update={"assessment_sha256": "0" * 64})
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_targeted_evidence_assessment(bad_assessment)

    decision = _decision(gap, assessment)
    validate_r0_action_decision(decision)
    bad_decision = decision.model_copy(update={"decision_sha256": "0" * 64})
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_r0_action_decision(bad_decision)



def test_decision_can_be_recomputed_exactly_from_sources() -> None:
    gap = _gap("targeted_search_then_refine", ["claim:c1"])
    review = _review("claim:c1", "PARTIAL_PRIOR_ART")
    assessment = _assessment(gap, [review])
    decision = _decision(gap, assessment)
    validate_r0_decision_against_sources(decision, gap, assessment=assessment)


def test_model_copy_cannot_bypass_decision_contract_validation() -> None:
    gap = _gap("targeted_search_only", ["claim:c1"])
    assessment = _assessment(gap, [_review("claim:c1", "DIRECT_PRIOR_ART")])
    decision = _decision(gap, assessment)
    bad = decision.model_copy(
        update={
            "route": "authorize_one_r1_then_r2",
            "r1_authorized": True,
            "max_refinements_authorized": 1,
        }
    )
    with pytest.raises(ValidationError, match="targeted_search_only"):
        validate_r0_action_decision(bad)

def test_contract_rejects_manual_targeted_search_only_r1_authorization() -> None:
    with pytest.raises(ValidationError, match="targeted_search_only"):
        R0ActionDecision(
            decision_id="r0_action_decision:x",
            decision_sha256="0" * 64,
            hypothesis_id=HID,
            gap_id="gap:g1",
            source_gap_plan_id=PLAN_ID,
            source_gap_plan_sha256=PLAN_SHA,
            source_gap_action="targeted_search_only",
            source_assessment_id="targeted_evidence_assessment:x",
            source_assessment_sha256="1" * 64,
            route="authorize_one_r1_then_r2",
            r1_authorized=True,
            r2_required=True,
            max_refinements_authorized=1,
            reason_codes=[],
        )


def test_contract_forbids_extra_fields_and_more_than_one_refinement() -> None:
    with pytest.raises(ValidationError):
        R0ActionDecision(
            decision_id="r0_action_decision:x",
            decision_sha256="0" * 64,
            hypothesis_id=HID,
            gap_id="gap:g1",
            source_gap_plan_id=PLAN_ID,
            source_gap_plan_sha256=PLAN_SHA,
            source_gap_action="targeted_search_then_refine",
            source_assessment_id="targeted_evidence_assessment:x",
            source_assessment_sha256="1" * 64,
            route="authorize_one_r1_then_r2",
            r1_authorized=True,
            r2_required=True,
            max_refinements_authorized=2,
            reason_codes=[],
            surprise="forbidden",
        )
