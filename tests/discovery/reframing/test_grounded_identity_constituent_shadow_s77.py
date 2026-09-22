from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.reframing.grounded_identity_constituent_shadow import (
    GroundedIdentityAnnotation,
    GroundedIdentityConstituentGroup,
    GroundedIdentityConstituentSpan,
    _identity_matches_abstract,
    _shadow_state,
    _validate_identity_constituent_lexical_contract,
)


def _annotation() -> GroundedIdentityAnnotation:
    return GroundedIdentityAnnotation(
        claim_id="c1",
        identity_term="architecture-conditioned hotspot accessibility",
        groups=[
            GroundedIdentityConstituentGroup(
                label="architecture",
                spans=[
                    GroundedIdentityConstituentSpan(
                        candidate_id="a",
                        candidate_ref="CANDIDATE_01",
                        exact_source_text="substrate architecture",
                        matched_source_paths=["scientific_proposal"],
                    )
                ],
            ),
            GroundedIdentityConstituentGroup(
                label="accessibility",
                spans=[
                    GroundedIdentityConstituentSpan(
                        candidate_id="b",
                        candidate_ref="CANDIDATE_02",
                        exact_source_text="hotspot accessibility",
                        matched_source_paths=["scientific_proposal"],
                    )
                ],
            ),
        ],
        rationale="test",
    )


def test_grounded_identity_requires_all_constituent_groups():
    annotation = _annotation()
    assert _identity_matches_abstract(
        abstract=(
            "The substrate architecture controls morphology and hotspot "
            "accessibility across the active SERS region."
        ),
        annotation=annotation,
    )
    assert not _identity_matches_abstract(
        abstract="The substrate architecture controls morphology.",
        annotation=annotation,
    )


def test_negative_shadow_never_uses_constituents_to_override_positive_state():
    assert _shadow_state(
        successful_query_count=3,
        positive_work_ids=["w1"],
        eligible_count=0,
        minimum_negative_abstracts=3,
    ) == "ESTABLISHED"


def test_negative_shadow_requires_minimum_grounded_coverage():
    assert _shadow_state(
        successful_query_count=3,
        positive_work_ids=[],
        eligible_count=2,
        minimum_negative_abstracts=3,
    ) == "UNASSESSED"
    assert _shadow_state(
        successful_query_count=3,
        positive_work_ids=[],
        eligible_count=3,
        minimum_negative_abstracts=3,
    ) == "NOT_FOUND"


def test_identity_constituent_contract_rejects_complete_identity_reuse():
    annotation = _annotation()
    bad_groups = [
        GroundedIdentityConstituentGroup(
            label="identity copied whole",
            spans=[
                GroundedIdentityConstituentSpan(
                    candidate_id="a",
                    candidate_ref="CANDIDATE_01",
                    exact_source_text=(
                        "architecture-conditioned hotspot accessibility"
                    ),
                    matched_source_paths=["scientific_proposal"],
                )
            ],
        ),
        annotation.groups[1],
    ]
    try:
        _validate_identity_constituent_lexical_contract(
            identity_term=annotation.identity_term,
            groups=bad_groups,
        )
    except ValueError as exc:
        assert "complete synthetic identity" in str(exc)
    else:
        raise AssertionError("expected complete identity reuse rejection")


def test_identity_constituent_contract_rejects_outcome_only_group():
    annotation = _annotation()
    bad_groups = [
        annotation.groups[0],
        GroundedIdentityConstituentGroup(
            label="outcome",
            spans=[
                GroundedIdentityConstituentSpan(
                    candidate_id="b",
                    candidate_ref="CANDIDATE_02",
                    exact_source_text="lower calibration transfer error",
                    matched_source_paths=["scientific_proposal"],
                )
            ],
        ),
    ]
    try:
        _validate_identity_constituent_lexical_contract(
            identity_term=annotation.identity_term,
            groups=bad_groups,
        )
    except ValueError as exc:
        assert "no lexical content" in str(exc)
    else:
        raise AssertionError("expected outcome-only group rejection")


def test_identity_constituent_contract_requires_full_identity_token_coverage():
    annotation = _annotation()
    incomplete = [
        GroundedIdentityConstituentGroup(
            label="architecture",
            spans=[
                GroundedIdentityConstituentSpan(
                    candidate_id="a",
                    candidate_ref="CANDIDATE_01",
                    exact_source_text="substrate architecture",
                    matched_source_paths=["scientific_proposal"],
                )
            ],
        ),
        GroundedIdentityConstituentGroup(
            label="hotspot",
            spans=[
                GroundedIdentityConstituentSpan(
                    candidate_id="b",
                    candidate_ref="CANDIDATE_02",
                    exact_source_text="hotspot population",
                    matched_source_paths=["scientific_proposal"],
                )
            ],
        ),
    ]
    try:
        _validate_identity_constituent_lexical_contract(
            identity_term=annotation.identity_term,
            groups=incomplete,
        )
    except ValueError as exc:
        assert "do not cover all lexical identity content tokens" in str(exc)
    else:
        raise AssertionError("expected incomplete identity coverage rejection")


def test_identity_constituent_contract_accepts_true_decomposition():
    groups = [
        GroundedIdentityConstituentGroup(
            label="architecture conditioning",
            spans=[
                GroundedIdentityConstituentSpan(
                    candidate_id="a",
                    candidate_ref="CANDIDATE_01",
                    exact_source_text="architecture-conditioned",
                    matched_source_paths=["scientific_proposal"],
                )
            ],
        ),
        GroundedIdentityConstituentGroup(
            label="hotspot accessibility",
            spans=[
                GroundedIdentityConstituentSpan(
                    candidate_id="b",
                    candidate_ref="CANDIDATE_02",
                    exact_source_text="hotspot accessibility",
                    matched_source_paths=["scientific_proposal"],
                )
            ],
        ),
    ]
    _validate_identity_constituent_lexical_contract(
        identity_term="architecture-conditioned hotspot accessibility",
        groups=groups,
    )
