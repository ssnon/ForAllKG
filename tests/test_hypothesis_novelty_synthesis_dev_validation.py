from __future__ import annotations

from types import SimpleNamespace

from dac_her.external_novelty_contracts import ExternalNoveltyPolicy
from dac_her.hypothesis_novelty_synthesis_dev_validation import (
    CLAIM_STATUSES,
    _synthetic_coverage,
    characterize_status_lattice,
    make_policy_assessor,
)


def _reviews(*statuses: str):
    return [
        SimpleNamespace(
            importance="core",
            status=status,
            claim_id=f"c{i}",
        )
        for i, status in enumerate(statuses, start=1)
    ]


def test_status_lattice_is_deterministic_and_complete():
    audit = characterize_status_lattice(
        ExternalNoveltyPolicy(),
        max_core_claims=4,
    )
    expected = 2 * sum(
        len(CLAIM_STATUSES) ** n
        for n in range(1, 5)
    )
    assert audit["case_count"] == expected
    assert len(audit["rows"]) == expected
    assert audit["scientific_semantic_outcome"] == (
        "MANUAL_REVIEW_REQUIRED"
    )


def test_conflict_has_precedence():
    assessor = make_policy_assessor()
    status, _, _ = assessor._status(
        _reviews(
            "PARTIAL_PRIOR_ART",
            "CONFLICTING_PRIOR_ART",
        ),
        _synthetic_coverage(True),
    )
    assert status == "CONFLICTING_PRIOR_ART"


def test_direct_and_partial_only_is_literature_supported_extension():
    assessor = make_policy_assessor()
    status, _, _ = assessor._status(
        _reviews(
            "DIRECT_PRIOR_ART",
            "PARTIAL_PRIOR_ART",
        ),
        _synthetic_coverage(True),
    )
    assert status == "LITERATURE_SUPPORTED_EXTENSION"


def test_absence_dependent_status_fails_closed_without_coverage():
    assessor = make_policy_assessor()
    status, _, _ = assessor._status(
        _reviews(
            "PARTIAL_PRIOR_ART",
            "COMPONENTS_ONLY",
        ),
        _synthetic_coverage(False),
    )
    assert status == "INSUFFICIENT_SEARCH_EVIDENCE"


def test_all_no_direct_requires_coverage_for_plausibly_novel():
    assessor = make_policy_assessor()
    insufficient, _, _ = assessor._status(
        _reviews(
            "NO_DIRECT_MATCH_FOUND",
            "NO_DIRECT_MATCH_FOUND",
        ),
        _synthetic_coverage(False),
    )
    sufficient, _, _ = assessor._status(
        _reviews(
            "NO_DIRECT_MATCH_FOUND",
            "NO_DIRECT_MATCH_FOUND",
        ),
        _synthetic_coverage(True),
    )
    assert insufficient == "INSUFFICIENT_SEARCH_EVIDENCE"
    assert sufficient == "PLAUSIBLY_NOVEL"


def test_all_components_current_policy_is_characterized_not_approved():
    audit = characterize_status_lattice(
        ExternalNoveltyPolicy(),
        max_core_claims=2,
    )
    probe = next(
        row
        for row in audit["manual_semantic_review_probes"]
        if row["name"] == "all_components_sufficient"
    )
    # This historical v1 harness characterizes the current production policy
    # but must never authorize the scientific status automatically. The exact
    # production status is intentionally allowed to evolve in later
    # hypothesis-level taxonomy hardening.
    assert probe["automatic_scientific_approval"] is False
    assert isinstance(probe["observed_status"], str)
    assert probe["observed_status"]
