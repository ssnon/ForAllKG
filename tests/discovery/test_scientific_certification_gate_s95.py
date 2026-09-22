from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.scientific_certification_gate import (
    _decision,
    _external_distinctness_state,
    _fatal_blocker_state,
)


def _coverage(sufficient: bool):
    return SimpleNamespace(
        sufficient_for_absence_based_novelty=sufficient
    )


def _card(status: str, sufficient: bool = True):
    return SimpleNamespace(
        status=status,
        coverage=_coverage(sufficient),
    )


def _nonob(
    *,
    direct=False,
    fatal=False,
):
    return SimpleNamespace(
        direct_prior_art_blocker_claim_ids=(
            ["c1"] if direct else []
        ),
        fatal_contradiction_blocker=fatal,
        fatal_contradiction_claim_ids=(
            ["c1"] if fatal else []
        ),
    )


def test_certifies_only_when_all_four_gate_conditions_hold():
    decision, reasons = _decision(
        closure_state="BOUNDED_REVIEW_CLOSED",
        external_distinctness_state=(
            "BOUNDED_EXTERNAL_DISTINCTNESS_SUPPORTED"
        ),
        positive_nonobviousness_authority_state="AUTHORIZED",
        fatal_blocker_state="NONE",
    )

    assert decision == "CERTIFIED"
    assert reasons == [
        "bounded_review_closed",
        "bounded_external_distinctness_supported",
        "positive_nonobviousness_authorized",
        "no_fatal_or_direct_prior_art_blocker",
    ]


def test_partial_review_coverage_stays_unresolved():
    decision, reasons = _decision(
        closure_state="PARTIAL_REVIEW_COVERAGE",
        external_distinctness_state=(
            "BOUNDED_EXTERNAL_DISTINCTNESS_SUPPORTED"
        ),
        positive_nonobviousness_authority_state="AUTHORIZED",
        fatal_blocker_state="NONE",
    )

    assert decision == "UNRESOLVED"
    assert "bounded_evidence_review_not_closed" in reasons


def test_absence_or_external_distinctness_without_positive_nonobviousness_cannot_certify():
    decision, reasons = _decision(
        closure_state="BOUNDED_REVIEW_CLOSED",
        external_distinctness_state=(
            "BOUNDED_EXTERNAL_DISTINCTNESS_SUPPORTED"
        ),
        positive_nonobviousness_authority_state="NOT_AUTHORIZED",
        fatal_blocker_state="NONE",
    )

    assert decision == "UNRESOLVED"
    assert (
        "positive_nonobviousness_not_authorized"
        in reasons
    )


def test_direct_prior_art_is_deterministic_rejection_blocker():
    blocker = _fatal_blocker_state(
        _nonob(direct=True)
    )
    assert blocker == "DIRECT_PRIOR_ART"

    decision, reasons = _decision(
        closure_state="BOUNDED_REVIEW_CLOSED",
        external_distinctness_state=(
            "BOUNDED_EXTERNAL_DISTINCTNESS_SUPPORTED"
        ),
        positive_nonobviousness_authority_state="AUTHORIZED",
        fatal_blocker_state=blocker,
    )

    assert decision == "REJECTED"
    assert (
        "direct_prior_art_blocker_present"
        in reasons
    )


def test_fatal_contradiction_is_deterministic_rejection_blocker():
    blocker = _fatal_blocker_state(
        _nonob(fatal=True)
    )
    assert blocker == "FATAL_CONTRADICTION"

    decision, reasons = _decision(
        closure_state="BOUNDED_REVIEW_CLOSED",
        external_distinctness_state=(
            "BOUNDED_EXTERNAL_DISTINCTNESS_SUPPORTED"
        ),
        positive_nonobviousness_authority_state="AUTHORIZED",
        fatal_blocker_state=blocker,
    )

    assert decision == "REJECTED"
    assert (
        "fatal_contradiction_blocker_present"
        in reasons
    )


def test_plausibly_novel_requires_search_coverage_for_bounded_distinctness():
    assert _external_distinctness_state(
        _card("PLAUSIBLY_NOVEL", sufficient=True)
    ) == "BOUNDED_EXTERNAL_DISTINCTNESS_SUPPORTED"

    assert _external_distinctness_state(
        _card("PLAUSIBLY_NOVEL", sufficient=False)
    ) == "INSUFFICIENT_EXTERNAL_SEARCH_EVIDENCE"


def test_legacy_relation_backed_status_does_not_become_new_rejection_authority_by_itself():
    state = _external_distinctness_state(
        _card(
            "LITERATURE_SUPPORTED_EXTENSION",
            sufficient=True,
        )
    )
    assert state == "RELATION_BACKED_EXTERNAL_PRIOR_ART"

    decision, reasons = _decision(
        closure_state="BOUNDED_REVIEW_CLOSED",
        external_distinctness_state=state,
        positive_nonobviousness_authority_state="AUTHORIZED",
        fatal_blocker_state="NONE",
    )

    # The new subsystem rejects only positive blockers from the new typed
    # adjudication path. Legacy external status merely prevents certification.
    assert decision == "UNRESOLVED"
    assert (
        "bounded_external_distinctness_not_established"
        in reasons
    )


def test_missing_external_card_fails_closed_to_unresolved():
    assert _external_distinctness_state(
        None
    ) == "MISSING_EXTERNAL_NOVELTY_CARD"

    decision, _ = _decision(
        closure_state="BOUNDED_REVIEW_CLOSED",
        external_distinctness_state=(
            "MISSING_EXTERNAL_NOVELTY_CARD"
        ),
        positive_nonobviousness_authority_state="AUTHORIZED",
        fatal_blocker_state="NONE",
    )
    assert decision == "UNRESOLVED"
