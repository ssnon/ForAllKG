import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaim,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    PredictedObservation,
)
from pipeline_core.discovery.novelty_selection_role_annotation import (
    NoveltySelectionRoleAnnotationDraft,
    NoveltySelectionRoleAssignmentDraft,
    build_role_annotation_prompt,
    compile_role_annotation,
)


def _hypothesis():
    return HypothesisCard(
        hypothesis_id="hypothesis:1",
        domain_profile_id="synthetic",
        source_context_id="context:1",
        source_context_sha256="ctxsha",
        source_report_id="report:1",
        source_report_sha256="repsha",
        title="Synthetic hypothesis",
        hypothesis_statement=(
            "Factor Z moderates the relationship "
            "between descriptor X and outcome Y."
        ),
        hypothesis_type="context_dependency",
        premise_statement_ids=["statement:1"],
        inferential_bridge=(
            "Descriptor X affects outcome Y, while "
            "factor Z may condition that relation."
        ),
        predicted_observations=[
            PredictedObservation(
                observation_id="obs:1",
                observable=(
                    "The X-Y slope differs across Z."
                ),
                expected_direction="qualitative_change",
                rationale=(
                    "This tests the proposed moderation."
                ),
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:1",
                observable=(
                    "The X-Y slope is invariant across Z."
                ),
                falsifying_outcome=(
                    "No detectable moderation by Z."
                ),
            )
        ],
        assumptions=[
            "Descriptor X is measurable."
        ],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )


def _claim(
    claim_id,
    *,
    kind="moderator_interaction",
    importance="core",
    text=(
        "Factor Z moderates the relationship "
        "between descriptor X and outcome Y."
    ),
):
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id="hypothesis:1",
        claim_rank=1,
        kind=kind,
        importance=importance,
        text=text,
        rationale="Synthetic atomic claim.",
    )


def test_role_prompt_hides_importance_and_outcomes():
    hypothesis = _hypothesis()

    claim = _claim(
        "claim:1",
        importance="core",
    )

    system, user = build_role_annotation_prompt(
        hypothesis,
        [claim],
    )

    assert "importance:" not in user
    assert "shadow_state" not in user
    assert "final_verdict" not in user
    assert "external_status" not in user
    assert "prior_art" not in user.lower()

    assert claim.claim_id in user
    assert claim.text in user

    assert (
        "Do not use claim importance."
        in system
    )


def test_role_compiler_preserves_canonical_claim_order():
    hypothesis = _hypothesis()

    claims = [
        _claim("claim:a"),
        _claim(
            "claim:b",
            kind="distinctive_prediction",
            importance="supporting",
            text=(
                "The X-Y slope differs across Z."
            ),
        ),
    ]

    draft = NoveltySelectionRoleAnnotationDraft(
        assignments=[
            NoveltySelectionRoleAssignmentDraft(
                claim_id="claim:b",
                novelty_selection_role=(
                    "TESTING_PREDICTION"
                ),
                rationale="Operational test.",
            ),
            NoveltySelectionRoleAssignmentDraft(
                claim_id="claim:a",
                novelty_selection_role=(
                    "NOVELTY_BEARING"
                ),
                rationale="Central relation.",
            ),
        ]
    )

    compiled = compile_role_annotation(
        hypothesis=hypothesis,
        claims=claims,
        draft=draft,
    )

    assert [
        row["claim_id"]
        for row in compiled
    ] == [
        "claim:a",
        "claim:b",
    ]

    assert (
        compiled[0]["novelty_selection_role"]
        == "NOVELTY_BEARING"
    )

    assert (
        compiled[1]["novelty_selection_role"]
        == "TESTING_PREDICTION"
    )

    assert all(
        row["outcome_blind"] is True
        for row in compiled
    )


def test_null_role_is_preserved_fail_closed():
    hypothesis = _hypothesis()
    claims = [_claim("claim:1")]

    draft = NoveltySelectionRoleAnnotationDraft(
        assignments=[
            NoveltySelectionRoleAssignmentDraft(
                claim_id="claim:1",
                novelty_selection_role=None,
                rationale=(
                    "Role cannot be assigned without "
                    "inventing structure."
                ),
            )
        ]
    )

    compiled = compile_role_annotation(
        hypothesis=hypothesis,
        claims=claims,
        draft=draft,
    )

    assert (
        compiled[0]["novelty_selection_role"]
        is None
    )


def test_missing_claim_id_is_rejected():
    hypothesis = _hypothesis()

    claims = [
        _claim("claim:a"),
        _claim("claim:b"),
    ]

    draft = NoveltySelectionRoleAnnotationDraft(
        assignments=[
            NoveltySelectionRoleAssignmentDraft(
                claim_id="claim:a",
                novelty_selection_role=(
                    "NOVELTY_BEARING"
                ),
                rationale="Central relation.",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing claim_id",
    ):
        compile_role_annotation(
            hypothesis=hypothesis,
            claims=claims,
            draft=draft,
        )


def test_unknown_claim_id_is_rejected():
    hypothesis = _hypothesis()
    claims = [_claim("claim:a")]

    draft = NoveltySelectionRoleAnnotationDraft(
        assignments=[
            NoveltySelectionRoleAssignmentDraft(
                claim_id="claim:other",
                novelty_selection_role=(
                    "NOVELTY_BEARING"
                ),
                rationale="Invalid ID.",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="unknown claim_id",
    ):
        compile_role_annotation(
            hypothesis=hypothesis,
            claims=claims,
            draft=draft,
        )


def test_duplicate_annotation_claim_id_is_rejected():
    hypothesis = _hypothesis()
    claims = [_claim("claim:a")]

    draft = NoveltySelectionRoleAnnotationDraft(
        assignments=[
            NoveltySelectionRoleAssignmentDraft(
                claim_id="claim:a",
                novelty_selection_role=(
                    "NOVELTY_BEARING"
                ),
                rationale="First.",
            ),
            NoveltySelectionRoleAssignmentDraft(
                claim_id="claim:a",
                novelty_selection_role=(
                    "AUXILIARY"
                ),
                rationale="Second.",
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="duplicate claim_id",
    ):
        compile_role_annotation(
            hypothesis=hypothesis,
            claims=claims,
            draft=draft,
        )
