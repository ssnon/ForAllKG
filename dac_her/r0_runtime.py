from __future__ import annotations

import hashlib
import json
from typing import Iterable

from dac_her.external_novelty_contracts import ClaimPriorArtReview
from dac_her.novelty_refinement_contracts import NoveltyGap
from dac_her.r0_contracts import (
    R0ActionDecision,
    R0RefinementBasis,
    R0RefinementBasisKind,
    TargetedEvidenceAssessment,
    TargetedEvidenceState,
)


_TARGETED_ACTIONS = {
    "targeted_search_then_refine",
    "targeted_search_only",
    "refine_away_from_conflict",
}

_UNRESOLVED_STATUSES = {
    "INSUFFICIENT_METADATA",
    "TITLE_ONLY_NEIGHBORS",
    "NO_DIRECT_MATCH_FOUND",
}

_STATUS_REASON_CODES = {
    "DIRECT_PRIOR_ART": "R0E_STATUS_DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART": "R0E_STATUS_PARTIAL_PRIOR_ART",
    "COMPONENTS_ONLY": "R0E_STATUS_COMPONENTS_ONLY",
    "CONFLICTING_PRIOR_ART": "R0E_STATUS_CONFLICTING_PRIOR_ART",
    "NO_DIRECT_MATCH_FOUND": "R0E_STATUS_NO_DIRECT_MATCH_FOUND_BOUNDED",
    "TITLE_ONLY_NEIGHBORS": "R0E_STATUS_TITLE_ONLY_UNRESOLVED",
    "INSUFFICIENT_METADATA": "R0E_STATUS_INSUFFICIENT_METADATA",
}

_BASIS_RULES: dict[str, tuple[str, frozenset[str]]] = {
    "qualify_partial_relation": (
        "PARTIAL_PRIOR_ART",
        frozenset({"PARTIAL_PRIOR_ART"}),
    ),
    "separate_known_components": (
        "COMPONENTS_ONLY",
        frozenset({"COMPONENT_ONLY", "CONTEXTUAL_CONFLICT"}),
    ),
    "resolve_scope_matched_conflict": (
        "CONFLICTING_PRIOR_ART",
        frozenset({"CONFLICTING_PRIOR_ART"}),
    ),
}


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _artifact_id(prefix: str, sha256: str) -> str:
    return f"{prefix}:{sha256[:20]}"


def _derive_evidence_state(
    reviews: Iterable[ClaimPriorArtReview],
) -> TargetedEvidenceState:
    statuses = [row.status for row in reviews]
    if not statuses:
        return "unresolved"

    # Fail closed: unresolved/negative-search states dominate stronger routing states.
    if any(row in _UNRESOLVED_STATUSES for row in statuses):
        return "unresolved"
    if "CONFLICTING_PRIOR_ART" in statuses:
        return "conflicted"
    if all(row == "DIRECT_PRIOR_ART" for row in statuses):
        return "directly_covered"
    if "COMPONENTS_ONLY" in statuses:
        return "relational_gap_remains"
    if all(row in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"} for row in statuses):
        if "PARTIAL_PRIOR_ART" in statuses:
            return "partially_covered"
        return "directly_covered"
    return "unresolved"


def compile_r0_refinement_basis(
    review: ClaimPriorArtReview,
    *,
    basis_kind: R0RefinementBasisKind,
    source_work_ids: Iterable[str],
) -> R0RefinementBasis:
    work_input = list(source_work_ids)
    work_ids = sorted(set(work_input))
    if not work_ids:
        raise ValueError("R0 refinement basis requires at least one source work")
    if len(work_ids) != len(work_input):
        raise ValueError("duplicate source work IDs in R0 refinement basis")

    required_status, allowed_relationships = _BASIS_RULES[basis_kind]
    if review.status != required_status:
        raise ValueError(
            f"basis kind {basis_kind!r} requires claim status {required_status}, "
            f"got {review.status}"
        )

    matches = {row.work_id: row for row in review.matches}
    missing = sorted(set(work_ids) - set(matches))
    if missing:
        raise ValueError(f"R0 refinement basis references unreviewed works: {missing}")
    invalid = sorted(
        work_id
        for work_id in work_ids
        if matches[work_id].relationship not in allowed_relationships
    )
    if invalid:
        raise ValueError(
            "R0 refinement basis work relationship is not eligible for basis kind "
            f"{basis_kind!r}: {invalid}"
        )
    if basis_kind == "resolve_scope_matched_conflict":
        incompatible = sorted(
            work_id
            for work_id in work_ids
            if not matches[work_id].scope_compatible_for_conflict
        )
        if incompatible:
            raise ValueError(
                "R0 conflict basis requires scope-compatible prior art: "
                f"{incompatible}"
            )

    payload = {
        "schema_version": "r0-refinement-basis-v1",
        "target_claim_id": review.claim_id,
        "basis_kind": basis_kind,
        "source_work_ids": work_ids,
        "epistemic_usage": "boundary_constraint_only_not_positive_premise",
    }
    sha256 = _sha256_json(payload)
    return R0RefinementBasis(
        basis_id=_artifact_id("r0_refinement_basis", sha256),
        basis_sha256=sha256,
        **payload,
    )


def validate_r0_refinement_basis(
    basis: R0RefinementBasis,
    review: ClaimPriorArtReview,
) -> None:
    R0RefinementBasis.model_validate(basis.model_dump(mode="json"))
    payload = basis.model_dump(mode="json", exclude={"basis_id", "basis_sha256"})
    expected_sha256 = _sha256_json(payload)
    expected_id = _artifact_id("r0_refinement_basis", expected_sha256)
    if basis.basis_sha256 != expected_sha256:
        raise ValueError("R0 refinement basis SHA256 mismatch")
    if basis.basis_id != expected_id:
        raise ValueError("R0 refinement basis ID mismatch")
    if basis.target_claim_id != review.claim_id:
        raise ValueError("R0 refinement basis claim lineage mismatch")

    required_status, allowed_relationships = _BASIS_RULES[basis.basis_kind]
    if review.status != required_status:
        raise ValueError("R0 refinement basis claim status drift")
    matches = {row.work_id: row for row in review.matches}
    missing = sorted(set(basis.source_work_ids) - set(matches))
    if missing:
        raise ValueError(f"R0 refinement basis references unreviewed works: {missing}")
    invalid = sorted(
        work_id
        for work_id in basis.source_work_ids
        if matches[work_id].relationship not in allowed_relationships
    )
    if invalid:
        raise ValueError(f"R0 refinement basis relationship drift: {invalid}")
    if basis.basis_kind == "resolve_scope_matched_conflict":
        incompatible = sorted(
            work_id
            for work_id in basis.source_work_ids
            if not matches[work_id].scope_compatible_for_conflict
        )
        if incompatible:
            raise ValueError(
                "R0 conflict basis requires scope-compatible prior art: "
                f"{incompatible}"
            )


def _eligible_basis_work_ids(
    review: ClaimPriorArtReview,
    basis_kind: R0RefinementBasisKind,
) -> list[str]:
    required_status, allowed_relationships = _BASIS_RULES[basis_kind]
    if review.status != required_status:
        return []

    eligible: list[str] = []
    for match in review.matches:
        if match.relationship not in allowed_relationships:
            continue
        if (
            basis_kind == "resolve_scope_matched_conflict"
            and not match.scope_compatible_for_conflict
        ):
            continue
        eligible.append(match.work_id)
    return sorted(set(eligible))


def _derive_refinement_bases(
    *,
    action: str,
    reviews: list[ClaimPriorArtReview],
    state: TargetedEvidenceState,
) -> list[R0RefinementBasis]:
    # R0 bases are deterministic boundary constraints derived from trusted
    # claim-review matches. They are never caller supplied and never positive
    # scientific premises.
    if action == "targeted_search_only":
        return []

    if action == "targeted_search_then_refine":
        if state not in {"partially_covered", "relational_gap_remains"}:
            return []
        candidate_kinds: tuple[R0RefinementBasisKind, ...] = (
            "qualify_partial_relation",
            "separate_known_components",
        )
    elif action == "refine_away_from_conflict":
        if state != "conflicted":
            return []
        candidate_kinds = ("resolve_scope_matched_conflict",)
    else:
        return []

    bases: list[R0RefinementBasis] = []
    for review in sorted(reviews, key=lambda row: row.claim_id):
        for basis_kind in candidate_kinds:
            work_ids = _eligible_basis_work_ids(review, basis_kind)
            if not work_ids:
                continue
            bases.append(
                compile_r0_refinement_basis(
                    review,
                    basis_kind=basis_kind,
                    source_work_ids=work_ids,
                )
            )
    return sorted(bases, key=lambda row: row.basis_id)


def _normalize_targeted_reviews(
    gap: NoveltyGap,
    claim_reviews: Iterable[ClaimPriorArtReview],
) -> tuple[list[str], list[ClaimPriorArtReview]]:
    if gap.action not in _TARGETED_ACTIONS:
        raise ValueError(
            f"gap action {gap.action!r} does not require targeted evidence assessment"
        )

    target_ids = sorted(set(gap.target_claim_ids))
    if not target_ids:
        raise ValueError("targeted gap must bind at least one target claim")
    if len(target_ids) != len(gap.target_claim_ids):
        raise ValueError("gap contains duplicate target_claim_ids")

    reviews = sorted(list(claim_reviews), key=lambda row: row.claim_id)
    review_ids = [row.claim_id for row in reviews]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("duplicate claim reviews")
    if set(review_ids) != set(target_ids):
        missing = sorted(set(target_ids) - set(review_ids))
        unexpected = sorted(set(review_ids) - set(target_ids))
        raise ValueError(
            "claim review set must exactly match targeted claims: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for review in reviews:
        if review.hypothesis_id != gap.hypothesis_id:
            raise ValueError(
                f"claim review hypothesis_id drift for {review.claim_id}: "
                f"expected {gap.hypothesis_id}, got {review.hypothesis_id}"
            )
        if review.coverage.claim_id != review.claim_id:
            raise ValueError(
                f"claim review coverage claim_id mismatch for {review.claim_id}"
            )
    return target_ids, reviews


def derive_r0_refinement_bases(
    gap: NoveltyGap,
    claim_reviews: Iterable[ClaimPriorArtReview],
) -> list[R0RefinementBasis]:
    """Derive R0 boundary constraints without caller-controlled authorization input."""
    _, reviews = _normalize_targeted_reviews(gap, claim_reviews)
    state = _derive_evidence_state(reviews)
    return _derive_refinement_bases(
        action=gap.action,
        reviews=reviews,
        state=state,
    )


def _assessment_reason_codes(
    reviews: list[ClaimPriorArtReview],
    state: TargetedEvidenceState,
    bases: list[R0RefinementBasis],
) -> list[str]:
    reasons = {
        _STATUS_REASON_CODES[row.status]
        for row in reviews
        if row.status in _STATUS_REASON_CODES
    }
    reasons.add(f"R0E_STATE:{state}")
    if bases:
        reasons.add("R0E_BOUNDARY_REFINEMENT_BASIS_DERIVED")
    if state == "unresolved":
        reasons.add("R0E_UNKNOWN_NOT_ABSENCE")
    return sorted(reasons)


def _validate_basis_set_semantics(
    *,
    action: str,
    reviews: list[ClaimPriorArtReview],
    state: TargetedEvidenceState,
    bases: list[R0RefinementBasis],
) -> None:
    if not bases:
        return

    by_claim = {row.claim_id: row for row in reviews}
    if action == "targeted_search_only":
        raise ValueError("targeted_search_only may not carry a refinement basis")

    if action == "targeted_search_then_refine":
        if state not in {"partially_covered", "relational_gap_remains"}:
            raise ValueError(
                "targeted_search_then_refine basis requires a partial/component evidence state"
            )
        allowed_kinds = {"qualify_partial_relation", "separate_known_components"}
    elif action == "refine_away_from_conflict":
        if state != "conflicted":
            raise ValueError("conflict refinement basis requires conflicted evidence")
        allowed_kinds = {"resolve_scope_matched_conflict"}
    else:
        raise ValueError(f"gap action {action!r} cannot carry a refinement basis")

    invalid_kinds = sorted(
        basis.basis_id for basis in bases if basis.basis_kind not in allowed_kinds
    )
    if invalid_kinds:
        raise ValueError(
            "R0 refinement basis kind is not eligible for source action: "
            f"{invalid_kinds}"
        )
    for basis in bases:
        review = by_claim.get(basis.target_claim_id)
        if review is None:
            raise ValueError(
                f"R0 refinement basis targets unknown claim {basis.target_claim_id}"
            )
        validate_r0_refinement_basis(basis, review)


def compile_targeted_evidence_assessment(
    gap: NoveltyGap,
    claim_reviews: Iterable[ClaimPriorArtReview],
    *,
    source_gap_plan_id: str,
    source_gap_plan_sha256: str,
    source_t1_run_id: str,
    source_t1_freeze_id: str,
    source_t1_manifest_sha256: str,
) -> TargetedEvidenceAssessment:
    target_ids, reviews = _normalize_targeted_reviews(gap, claim_reviews)

    state = _derive_evidence_state(reviews)
    bases = _derive_refinement_bases(
        action=gap.action,
        reviews=reviews,
        state=state,
    )
    _validate_basis_set_semantics(
        action=gap.action,
        reviews=reviews,
        state=state,
        bases=bases,
    )
    reasons = _assessment_reason_codes(reviews, state, bases)

    payload = {
        "schema_version": "targeted-evidence-assessment-v1",
        "hypothesis_id": gap.hypothesis_id,
        "gap_id": gap.gap_id,
        "source_gap_plan_id": source_gap_plan_id,
        "source_gap_plan_sha256": source_gap_plan_sha256,
        "source_gap_action": gap.action,
        "source_t1_run_id": source_t1_run_id,
        "source_t1_freeze_id": source_t1_freeze_id,
        "source_t1_manifest_sha256": source_t1_manifest_sha256,
        "target_claim_ids": target_ids,
        "claim_reviews": [row.model_dump(mode="json") for row in reviews],
        "evidence_state": state,
        "refinement_bases": [row.model_dump(mode="json") for row in bases],
        "reason_codes": reasons,
        "literature_absence_claimed": False,
        "external_prior_art_can_be_positive_premise": False,
        "policy_version": "r0-targeted-evidence-policy-v1",
    }
    sha256 = _sha256_json(payload)
    return TargetedEvidenceAssessment(
        assessment_id=_artifact_id("targeted_evidence_assessment", sha256),
        assessment_sha256=sha256,
        **payload,
    )


def validate_targeted_evidence_assessment(
    assessment: TargetedEvidenceAssessment,
) -> None:
    TargetedEvidenceAssessment.model_validate(assessment.model_dump(mode="json"))
    payload = assessment.model_dump(
        mode="json",
        exclude={"assessment_id", "assessment_sha256"},
    )
    expected_sha256 = _sha256_json(payload)
    expected_id = _artifact_id("targeted_evidence_assessment", expected_sha256)
    if assessment.assessment_sha256 != expected_sha256:
        raise ValueError("targeted evidence assessment SHA256 mismatch")
    if assessment.assessment_id != expected_id:
        raise ValueError("targeted evidence assessment ID mismatch")

    state = _derive_evidence_state(assessment.claim_reviews)
    if assessment.evidence_state != state:
        raise ValueError(
            "targeted evidence assessment state drift: "
            f"expected {state}, got {assessment.evidence_state}"
        )
    _validate_basis_set_semantics(
        action=assessment.source_gap_action,
        reviews=assessment.claim_reviews,
        state=state,
        bases=assessment.refinement_bases,
    )
    expected_bases = _derive_refinement_bases(
        action=assessment.source_gap_action,
        reviews=assessment.claim_reviews,
        state=state,
    )
    if [row.model_dump(mode="json") for row in assessment.refinement_bases] != [
        row.model_dump(mode="json") for row in expected_bases
    ]:
        raise ValueError(
            "targeted evidence assessment refinement-basis derivation drift"
        )
    expected_reasons = _assessment_reason_codes(
        assessment.claim_reviews,
        state,
        assessment.refinement_bases,
    )
    if assessment.reason_codes != expected_reasons:
        raise ValueError("targeted evidence assessment reason-code drift")


def _validate_assessment_lineage(
    gap: NoveltyGap,
    assessment: TargetedEvidenceAssessment,
    *,
    source_gap_plan_id: str,
    source_gap_plan_sha256: str,
) -> None:
    validate_targeted_evidence_assessment(assessment)
    mismatches: list[str] = []
    if assessment.hypothesis_id != gap.hypothesis_id:
        mismatches.append("hypothesis_id")
    if assessment.gap_id != gap.gap_id:
        mismatches.append("gap_id")
    if assessment.source_gap_action != gap.action:
        mismatches.append("source_gap_action")
    if assessment.source_gap_plan_id != source_gap_plan_id:
        mismatches.append("source_gap_plan_id")
    if assessment.source_gap_plan_sha256 != source_gap_plan_sha256:
        mismatches.append("source_gap_plan_sha256")
    if assessment.target_claim_ids != sorted(gap.target_claim_ids):
        mismatches.append("target_claim_ids")
    if mismatches:
        raise ValueError(f"R0 assessment lineage mismatch: {sorted(mismatches)}")


def decide_r0(
    gap: NoveltyGap,
    *,
    source_gap_plan_id: str,
    source_gap_plan_sha256: str,
    assessment: TargetedEvidenceAssessment | None = None,
) -> R0ActionDecision:
    if gap.action in {"keep", "reject"}:
        if assessment is not None:
            raise ValueError(f"frozen {gap.action} action may not consume targeted assessment")
    else:
        if assessment is None:
            raise ValueError(f"targeted gap action {gap.action!r} requires an assessment")
        _validate_assessment_lineage(
            gap,
            assessment,
            source_gap_plan_id=source_gap_plan_id,
            source_gap_plan_sha256=source_gap_plan_sha256,
        )

    reasons: set[str] = set()
    source_assessment_id: str | None = None
    source_assessment_sha256: str | None = None

    if gap.action == "keep":
        route = "pass_through_frozen"
        r1_authorized = False
        r2_required = True
        max_refinements = 0
        reasons.add("R0D_SOURCE_KEEP")
    elif gap.action == "reject":
        route = "terminal_reject"
        r1_authorized = False
        r2_required = False
        max_refinements = 0
        reasons.add("R0D_SOURCE_REJECT")
    else:
        assert assessment is not None
        source_assessment_id = assessment.assessment_id
        source_assessment_sha256 = assessment.assessment_sha256
        state = assessment.evidence_state
        has_basis = bool(assessment.refinement_bases)

        route = "pass_original_to_r2"
        r1_authorized = False
        r2_required = True
        max_refinements = 0

        if gap.action == "targeted_search_only":
            reasons.add("R0D_TARGETED_SEARCH_ONLY_R1_FORBIDDEN")
        elif gap.action == "targeted_search_then_refine":
            if state in {"partially_covered", "relational_gap_remains"} and has_basis:
                route = "authorize_one_r1_then_r2"
                r1_authorized = True
                max_refinements = 1
                reasons.add("R0D_BOUNDARY_EVIDENCE_BOUNDED_REFINEMENT")
                reasons.add("R0D_R1_ONCE_AUTHORIZED")
            elif state in {"partially_covered", "relational_gap_remains"}:
                reasons.add("R0D_NO_BOUNDARY_REFINEMENT_BASIS")
        elif gap.action == "refine_away_from_conflict":
            if state == "conflicted" and has_basis:
                route = "authorize_one_r1_then_r2"
                r1_authorized = True
                max_refinements = 1
                reasons.add("R0D_BOUNDARY_EVIDENCE_BOUNDED_REFINEMENT")
                reasons.add("R0D_R1_ONCE_AUTHORIZED")
            else:
                reasons.add("R0D_NO_BOUNDARY_REFINEMENT_BASIS")

        if state == "directly_covered":
            reasons.add("R0D_DIRECT_EVIDENCE_NO_POSTHOC_REWRITE")
        elif state == "unresolved":
            reasons.add("R0D_REVIEW_LIMITATION_NO_REFINEMENT")
        elif state == "conflicted" and route != "authorize_one_r1_then_r2":
            reasons.add("R0D_POST_TARGETED_CONFLICT_DEFERRED_TO_R2")

        if route == "pass_original_to_r2":
            reasons.add("R0D_PASS_ORIGINAL_TO_R2")

    reason_codes = sorted(reasons)
    payload = {
        "schema_version": "r0-action-decision-v1",
        "hypothesis_id": gap.hypothesis_id,
        "gap_id": gap.gap_id,
        "source_gap_plan_id": source_gap_plan_id,
        "source_gap_plan_sha256": source_gap_plan_sha256,
        "source_gap_action": gap.action,
        "source_assessment_id": source_assessment_id,
        "source_assessment_sha256": source_assessment_sha256,
        "route": route,
        "r1_authorized": r1_authorized,
        "r2_required": r2_required,
        "max_refinements_authorized": max_refinements,
        "reason_codes": reason_codes,
        "network_calls": 0,
        "llm_calls": 0,
        "hypothesis_rewritten": False,
        "automatic_next_stage_authorized": False,
        "external_prior_art_can_be_positive_premise": False,
        "policy_version": "r0-action-policy-v1",
    }
    sha256 = _sha256_json(payload)
    decision = R0ActionDecision(
        decision_id=_artifact_id("r0_action_decision", sha256),
        decision_sha256=sha256,
        **payload,
    )

    # Second mechanical barrier for the most important source action.
    if gap.action == "targeted_search_only":
        assert decision.route == "pass_original_to_r2"
        assert decision.r1_authorized is False
        assert decision.max_refinements_authorized == 0

    return decision


def validate_r0_action_decision(decision: R0ActionDecision) -> None:
    R0ActionDecision.model_validate(decision.model_dump(mode="json"))
    payload = decision.model_dump(
        mode="json",
        exclude={"decision_id", "decision_sha256"},
    )
    expected_sha256 = _sha256_json(payload)
    expected_id = _artifact_id("r0_action_decision", expected_sha256)
    if decision.decision_sha256 != expected_sha256:
        raise ValueError("R0 action decision SHA256 mismatch")
    if decision.decision_id != expected_id:
        raise ValueError("R0 action decision ID mismatch")


def validate_r0_decision_against_sources(
    decision: R0ActionDecision,
    gap: NoveltyGap,
    *,
    assessment: TargetedEvidenceAssessment | None = None,
) -> None:
    """Recompute the deterministic R0 decision from its frozen source objects."""
    validate_r0_action_decision(decision)
    expected = decide_r0(
        gap,
        source_gap_plan_id=decision.source_gap_plan_id,
        source_gap_plan_sha256=decision.source_gap_plan_sha256,
        assessment=assessment,
    )
    if decision.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise ValueError("R0 action decision does not match deterministic source recomputation")
