from __future__ import annotations

from pipeline_core.discovery.positive_nonobviousness_adjudication import (
    PositiveNonObviousnessClaimDraft,
    PositiveNonObviousnessHypothesisDraft,
    _deterministic_skip_gate,
    _gate_from_reviews,
    _scope_model_draft_to_call,
    compile_positive_nonobviousness_claim_review,
)
from pipeline_core.discovery.positive_nonobviousness_basis import (
    ClaimPositiveNonObviousnessBasisRecord,
    HypothesisPositiveNonObviousnessBasisSummary,
    PositiveNonObviousnessBasisEvidence,
)


def _evidence(work_id="w1"):
    return PositiveNonObviousnessBasisEvidence(
        hypothesis_id="h1",
        claim_id="c1",
        work_id=work_id,
        relationship="DIRECTIONAL_COUNTEREVIDENCE",
        basis_kind="DOCUMENTED_DIRECTIONAL_TENSION",
        evidence_span=(
            "The reported relation was decoupled under matched conditions."
        ),
        rationale="test",
        confidence=0.9,
    )


def _record(
    *,
    role="NOVELTY_BEARING",
    evidence=True,
    direct=False,
    coverage=1.0,
):
    basis = [_evidence()] if evidence else []
    relevance = {
        "NOVELTY_BEARING": "PRIMARY_NOVELTY_BEARING",
        "REQUIRED_ENABLING_RELATION": "REQUIRED_ENABLING",
        "TESTING_PREDICTION": "TESTING_ONLY",
        "AUXILIARY": "AUXILIARY_ONLY",
    }[role]
    qualifying = bool(
        basis
        and relevance
        in {
            "PRIMARY_NOVELTY_BEARING",
            "REQUIRED_ENABLING",
        }
    )
    if not basis:
        state = "NO_POSITIVE_BASIS_CANDIDATE"
    elif relevance == "PRIMARY_NOVELTY_BEARING":
        state = "PRIMARY_POSITIVE_BASIS_CANDIDATE"
    elif relevance == "REQUIRED_ENABLING":
        state = "ENABLING_POSITIVE_BASIS_CANDIDATE"
    else:
        state = "NONQUALIFYING_TENSION_ONLY"

    return ClaimPositiveNonObviousnessBasisRecord(
        hypothesis_id="h1",
        claim_id="c1",
        novelty_selection_role=role,
        role_relevance=relevance,
        structural_centrality_score=1.0,
        centrality_rank_within_hypothesis=1,
        highest_centrality_within_hypothesis=True,
        classification_fraction=coverage,
        classification_coverage_state=(
            "FULLY_CLASSIFIED"
            if coverage == 1.0
            else "PARTIALLY_CLASSIFIED"
        ),
        basis_evidence=basis,
        basis_evidence_count=len(basis),
        basis_kinds=[
            row.basis_kind for row in basis
        ],
        basis_work_ids=[
            row.work_id for row in basis
        ],
        direct_prior_art_work_ids=(
            ["w-direct"] if direct else []
        ),
        lower_order_prior_art_work_ids=[],
        direct_prior_art_blocker_present=direct,
        lower_order_prior_art_pressure_present=False,
        basis_state=state,
        qualifying_for_future_nonobviousness_adjudication=qualifying,
    )


def _summary(
    *,
    positive_state,
    readiness,
    qualifying=None,
    nonqualifying=None,
    no_basis=None,
    direct=None,
    fully=True,
):
    qualifying = qualifying or []
    nonqualifying = nonqualifying or []
    no_basis = no_basis or []
    direct = direct or []
    return HypothesisPositiveNonObviousnessBasisSummary(
        hypothesis_id="h1",
        claim_ids=["c1"],
        qualifying_basis_claim_ids=qualifying,
        nonqualifying_tension_claim_ids=nonqualifying,
        no_basis_claim_ids=no_basis,
        basis_work_ids=(["w1"] if qualifying or nonqualifying else []),
        basis_kind_counts=(
            {"DOCUMENTED_DIRECTIONAL_TENSION": 1}
            if qualifying or nonqualifying
            else {}
        ),
        direct_prior_art_blocker_claim_ids=direct,
        lower_order_pressure_claim_ids=[],
        positive_basis_state=positive_state,
        future_adjudication_readiness=readiness,
        epistemic_coverage_profile=(
            "FULL_CLASSIFICATION_COVERAGE"
            if fully
            else "PARTIAL_CLASSIFICATION_COVERAGE"
        ),
        all_claims_fully_classified=fully,
        any_unclassified_presented_work=not fully,
        direct_prior_art_blocks_future_certification=bool(
            direct
        ),
    )


def test_no_basis_skips_without_llm_authority():
    summary = _summary(
        positive_state="NO_POSITIVE_BASIS_CANDIDATE",
        readiness="NO_POSITIVE_BASIS",
        no_basis=["c1"],
    )
    assert _deterministic_skip_gate(summary) == (
        "NO_POSITIVE_BASIS_UNRESOLVED"
    )


def test_partial_coverage_skips_even_with_qualifying_basis():
    summary = _summary(
        positive_state=(
            "QUALIFYING_POSITIVE_BASIS_CANDIDATE_PRESENT"
        ),
        readiness=(
            "BASIS_PRESENT_BUT_EPISTEMICALLY_PARTIAL"
        ),
        qualifying=["c1"],
        fully=False,
    )
    assert _deterministic_skip_gate(summary) == (
        "EPISTEMICALLY_PARTIAL_UNRESOLVED"
    )


def test_direct_prior_art_blocker_precedes_basis_adjudication():
    summary = _summary(
        positive_state=(
            "QUALIFYING_POSITIVE_BASIS_CANDIDATE_PRESENT"
        ),
        readiness=(
            "BASIS_PRESENT_BUT_DIRECT_PRIOR_ART_BLOCKS_CERTIFICATION"
        ),
        qualifying=["c1"],
        direct=["c1"],
    )
    assert _deterministic_skip_gate(summary) == (
        "DIRECT_PRIOR_ART_BLOCKED"
    )


def test_productive_tension_with_valid_basis_creates_bounded_authority():
    record = _record()
    draft = PositiveNonObviousnessClaimDraft(
        claim_id="c1",
        disposition="PRODUCTIVE_EXPECTATION_TENSION",
        confidence=0.9,
        supporting_basis_work_ids=["w1"],
        fatal_basis_work_ids=[],
        rationale=(
            "The exact decoupling evidence establishes a non-routine "
            "expectation relevant to the proposed bridge."
        ),
    )

    compiled = compile_positive_nonobviousness_claim_review(
        basis_record=record,
        draft=draft,
    )

    assert compiled.compiled_state == (
        "POSITIVE_NONOBVIOUSNESS_SUPPORTED"
    )
    assert compiled.positive_nonobviousness_authority is True
    assert compiled.novelty_authority is False
    assert _gate_from_reviews([compiled]) == (
        "POSITIVE_NONOBVIOUSNESS_AUTHORIZED"
    )


def test_fatal_contradiction_blocks_even_though_same_work_was_basis_candidate():
    record = _record()
    draft = PositiveNonObviousnessClaimDraft(
        claim_id="c1",
        disposition="FATAL_OR_DIRECT_CONTRADICTION",
        confidence=0.95,
        supporting_basis_work_ids=[],
        fatal_basis_work_ids=["w1"],
        rationale=(
            "The cited evidence directly contradicts the proposed relation "
            "in overlapping scope."
        ),
    )

    compiled = compile_positive_nonobviousness_claim_review(
        basis_record=record,
        draft=draft,
    )

    assert compiled.compiled_state == (
        "FATAL_CONTRADICTION_FOUND"
    )
    assert compiled.fatal_contradiction_authority is True
    assert compiled.positive_nonobviousness_authority is False
    assert _gate_from_reviews([compiled]) == (
        "FATAL_CONTRADICTION_BLOCKED"
    )


def test_hallucinated_basis_work_downgrades_fail_closed():
    record = _record()
    # Build an otherwise-valid draft then mutate after validation so the
    # deterministic compiler is tested against untrusted model output.
    draft = PositiveNonObviousnessClaimDraft(
        claim_id="c1",
        disposition="PRODUCTIVE_EXPECTATION_TENSION",
        confidence=0.8,
        supporting_basis_work_ids=["w1"],
        fatal_basis_work_ids=[],
        rationale="test",
    ).model_copy(
        update={
            "supporting_basis_work_ids": [
                "hallucinated-work"
            ]
        }
    )

    compiled = compile_positive_nonobviousness_claim_review(
        basis_record=record,
        draft=draft,
    )

    assert compiled.compiled_state == (
        "INSUFFICIENT_FOR_ADJUDICATION"
    )
    assert compiled.positive_nonobviousness_authority is False
    assert (
        "unsupported_supporting_basis_work_ids_dropped"
        in compiled.deterministic_reason_codes
    )
    assert (
        "productive_tension_requires_valid_basis_work"
        in compiled.deterministic_reason_codes
    )


def test_routine_extension_does_not_create_positive_authority():
    record = _record()
    draft = PositiveNonObviousnessClaimDraft(
        claim_id="c1",
        disposition="ROUTINE_OR_EXPECTED_EXTENSION",
        confidence=0.8,
        supporting_basis_work_ids=[],
        fatal_basis_work_ids=[],
        rationale=(
            "The cited tension does not make the proposed contextual "
            "extension non-routine."
        ),
    )

    compiled = compile_positive_nonobviousness_claim_review(
        basis_record=record,
        draft=draft,
    )

    assert compiled.compiled_state == (
        "POSITIVE_BASIS_NOT_ESTABLISHED"
    )
    assert _gate_from_reviews([compiled]) == (
        "POSITIVE_BASIS_NOT_ESTABLISHED"
    )


def test_model_hypothesis_id_echo_is_bound_to_deterministic_call_scope():
    draft = PositiveNonObviousnessHypothesisDraft(
        hypothesis_id="hypothesis:wrong-echo",
        claim_assessments=[
            PositiveNonObviousnessClaimDraft(
                claim_id="c1",
                disposition="ROUTINE_OR_EXPECTED_EXTENSION",
                confidence=0.8,
                supporting_basis_work_ids=[],
                fatal_basis_work_ids=[],
                rationale="eligible claim",
            ),
            PositiveNonObviousnessClaimDraft(
                claim_id="foreign-claim",
                disposition="ROUTINE_OR_EXPECTED_EXTENSION",
                confidence=0.7,
                supporting_basis_work_ids=[],
                fatal_basis_work_ids=[],
                rationale="must be dropped",
            ),
        ],
        interpretation="test",
    )

    scoped, reasons = _scope_model_draft_to_call(
        draft=draft,
        expected_hypothesis_id="hypothesis:h1",
        eligible_claim_ids=["c1"],
    )

    assert set(scoped) == {"c1"}
    assert (
        "model_hypothesis_id_mismatch_bound_to_call_scope"
        in reasons
    )
    assert "model_unknown_claim_ids_dropped" in reasons
