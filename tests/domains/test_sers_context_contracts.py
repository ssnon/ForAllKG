from __future__ import annotations

import pytest
from pydantic import ValidationError

from domains.sers.context_contracts import (
    SERSContextFact,
    SERSContextFinding,
    SERSContextProvenance,
    SERSContextReview,
    SERSContextSignature,
    expected_context_finding_severity,
    expected_context_review_status,
)


def _axis_provenance(
    node_id: str,
) -> SERSContextProvenance:
    return SERSContextProvenance(
        kind="axis_structural_edge",
        node_ids=[node_id],
        paper_ids=["paper:axis"],
    )


def _hypothesis_provenance(
    hypothesis_id: str,
) -> SERSContextProvenance:
    return SERSContextProvenance(
        kind="hypothesis_assertion",
        hypothesis_ids=[hypothesis_id],
        excerpt="hypothesis assertion",
    )


def _explicit_fact(
    *,
    fact_id: str,
    dimension: str,
    role: str,
    value: str,
    provenance: SERSContextProvenance,
) -> SERSContextFact:
    return SERSContextFact(
        fact_id=fact_id,
        dimension=dimension,
        scientific_role=role,
        knowledge_state="explicit",
        value=value,
        provenance=[provenance],
    )


def test_explicit_fact_requires_value() -> None:
    with pytest.raises(
        ValidationError
    ):
        SERSContextFact(
            fact_id="fact:bad",
            dimension="material_identity",
            scientific_role="component",
            knowledge_state="explicit",
            provenance=[
                _axis_provenance(
                    "node:cu"
                )
            ],
        )


def test_unknown_fact_cannot_invent_value() -> None:
    with pytest.raises(
        ValidationError
    ):
        SERSContextFact(
            fact_id="fact:bad",
            dimension="material_state",
            scientific_role="material_state",
            knowledge_state="unknown",
            value="CuO",
            provenance=[
                _axis_provenance(
                    "node:cu"
                )
            ],
        )


def test_severity_mapping_is_deterministic() -> None:
    assert (
        expected_context_finding_severity(
            "match"
        )
        == "info"
    )

    assert (
        expected_context_finding_severity(
            "intentional_variation"
        )
        == "info"
    )

    assert (
        expected_context_finding_severity(
            "compatible_extension"
        )
        == "info"
    )

    assert (
        expected_context_finding_severity(
            "unknown"
        )
        == "warning"
    )

    assert (
        expected_context_finding_severity(
            "role_mismatch"
        )
        == "actionable"
    )

    assert (
        expected_context_finding_severity(
            "context_conflation"
        )
        == "actionable"
    )

    assert (
        expected_context_finding_severity(
            "conflict"
        )
        == "actionable"
    )


def test_h1_context_conflation_is_actionable() -> None:
    axis = SERSContextSignature(
        signature_id="sig:h1:axis",
        domain_profile_id="sers_au_ag",
        scope="axis_inspiration",
        source_ref_id="candidate:h1",
        facts=[
            _explicit_fact(
                fact_id="fact:h1:pyramid",
                dimension="morphology",
                role="morphology",
                value=(
                    "3D-Si with inserted small pyramid"
                ),
                provenance=_axis_provenance(
                    "node:3d-si"
                ),
            ),
            _explicit_fact(
                fact_id="fact:h1:gap",
                dimension="gap_regime",
                role="gap_regime",
                value="10 nm nanoparticle gap",
                provenance=_axis_provenance(
                    "node:auag-np"
                ),
            ),
        ],
    )

    hypothesis = SERSContextSignature(
        signature_id="sig:h1:hypothesis",
        domain_profile_id="sers_au_ag",
        scope="hypothesis",
        source_ref_id="hypothesis:h1",
        facts=[
            _explicit_fact(
                fact_id="fact:h1:hyp-gap",
                dimension="gap_regime",
                role="gap_regime",
                value=(
                    "inserted-pyramid-like "
                    "nanogap geometry"
                ),
                provenance=(
                    _hypothesis_provenance(
                        "hypothesis:h1"
                    )
                ),
            ),
        ],
    )

    finding = SERSContextFinding(
        finding_id="finding:h1:conflation",
        dimension="gap_regime",
        status="context_conflation",
        severity="actionable",
        left_signature_id=axis.signature_id,
        right_signature_id=hypothesis.signature_id,
        left_fact_ids=[
            "fact:h1:pyramid",
            "fact:h1:gap",
        ],
        right_fact_ids=[
            "fact:h1:hyp-gap",
        ],
        rationale=(
            "The source assigns inserted-pyramid "
            "geometry to the 3D-Si morphology while "
            "the nanoparticle gap is a distinct "
            "structural context."
        ),
        tags=[
            "support_morphology",
            "nanoparticle_gap",
        ],
    )

    review = SERSContextReview(
        review_id="review:h1",
        hypothesis_id="hypothesis:h1",
        signatures=[
            axis,
            hypothesis,
        ],
        findings=[
            finding
        ],
        status="reframe_required",
    )

    assert (
        review.status
        == "reframe_required"
    )


def test_h2_unknown_material_state_is_not_rejection() -> None:
    axis = SERSContextSignature(
        signature_id="sig:h2:axis",
        domain_profile_id="sers_au_ag",
        scope="axis_inspiration",
        source_ref_id="candidate:h2",
        facts=[
            _explicit_fact(
                fact_id="fact:h2:substrate",
                dimension="substrate",
                role="plasmonic_substrate",
                value=(
                    "Copper substrate with "
                    "nanostructured gold/silver"
                ),
                provenance=_axis_provenance(
                    "node:cu-substrate"
                ),
            ),
            _explicit_fact(
                fact_id="fact:h2:cu",
                dimension="material_identity",
                role="component",
                value="Copper",
                provenance=_axis_provenance(
                    "node:cu"
                ),
            ),
            SERSContextFact(
                fact_id="fact:h2:state",
                dimension="material_state",
                scientific_role="material_state",
                knowledge_state="unknown",
                provenance=[
                    _axis_provenance(
                        "node:cu"
                    )
                ],
            ),
        ],
    )

    hypothesis = SERSContextSignature(
        signature_id="sig:h2:hypothesis",
        domain_profile_id="sers_au_ag",
        scope="hypothesis",
        source_ref_id="hypothesis:h2",
        facts=[
            _explicit_fact(
                fact_id="fact:h2:hyp-substrate",
                dimension="substrate",
                role="plasmonic_substrate",
                value=(
                    "copper-supported Au-Ag "
                    "nanostructure"
                ),
                provenance=(
                    _hypothesis_provenance(
                        "hypothesis:h2"
                    )
                ),
            ),
            SERSContextFact(
                fact_id="fact:h2:hyp-state",
                dimension="material_state",
                scientific_role="material_state",
                knowledge_state="unknown",
                provenance=[
                    _hypothesis_provenance(
                        "hypothesis:h2"
                    )
                ],
            ),
        ],
    )

    variation = SERSContextFinding(
        finding_id="finding:h2:variation",
        dimension="substrate",
        status="intentional_variation",
        severity="info",
        left_signature_id=axis.signature_id,
        right_signature_id=hypothesis.signature_id,
        left_fact_ids=[
            "fact:h2:substrate"
        ],
        right_fact_ids=[
            "fact:h2:hyp-substrate"
        ],
        rationale=(
            "Copper-associated substrate context "
            "is the proposed experimental moderator."
        ),
    )

    unknown = SERSContextFinding(
        finding_id="finding:h2:state",
        dimension="material_state",
        status="unknown",
        severity="warning",
        left_signature_id=axis.signature_id,
        right_signature_id=hypothesis.signature_id,
        left_fact_ids=[
            "fact:h2:state"
        ],
        right_fact_ids=[
            "fact:h2:hyp-state"
        ],
        rationale=(
            "Candidate-local provenance does not "
            "establish copper oxidation state."
        ),
    )

    review = SERSContextReview(
        review_id="review:h2",
        hypothesis_id="hypothesis:h2",
        signatures=[
            axis,
            hypothesis,
        ],
        findings=[
            variation,
            unknown,
        ],
        status="pass_with_unknowns",
    )

    assert (
        review.status
        == "pass_with_unknowns"
    )


def test_invalid_finding_severity_is_rejected() -> None:
    with pytest.raises(
        ValidationError
    ):
        SERSContextFinding(
            finding_id="finding:bad",
            dimension="material_state",
            status="unknown",
            severity="info",
            left_signature_id="sig:left",
            right_signature_id="sig:right",
            left_fact_ids=["fact:left"],
            right_fact_ids=["fact:right"],
            rationale="missing material state",
        )


def test_review_status_is_deterministic() -> None:
    finding = SERSContextFinding(
        finding_id="finding:unknown",
        dimension="material_state",
        status="unknown",
        severity="warning",
        left_signature_id="sig:left",
        right_signature_id="sig:right",
        left_fact_ids=["fact:left"],
        right_fact_ids=["fact:right"],
        rationale="material state is unknown",
    )

    assert (
        expected_context_review_status(
            [finding]
        )
        == "pass_with_unknowns"
    )


def test_review_rejects_foreign_fact_reference() -> None:
    left = SERSContextSignature(
        signature_id="sig:left",
        domain_profile_id="sers_au_ag",
        scope="axis_inspiration",
        source_ref_id="candidate:left",
        facts=[
            _explicit_fact(
                fact_id="fact:left",
                dimension="material_identity",
                role="component",
                value="Copper",
                provenance=_axis_provenance(
                    "node:left"
                ),
            )
        ],
    )

    right = SERSContextSignature(
        signature_id="sig:right",
        domain_profile_id="sers_au_ag",
        scope="hypothesis",
        source_ref_id="hypothesis:right",
        facts=[
            _explicit_fact(
                fact_id="fact:right",
                dimension="material_identity",
                role="component",
                value="Copper",
                provenance=(
                    _hypothesis_provenance(
                        "hypothesis:right"
                    )
                ),
            )
        ],
    )

    finding = SERSContextFinding(
        finding_id="finding:bad-ref",
        dimension="material_identity",
        status="match",
        severity="info",
        left_signature_id="sig:left",
        right_signature_id="sig:right",
        left_fact_ids=[
            "fact:not-left"
        ],
        right_fact_ids=[
            "fact:right"
        ],
        rationale="test invalid reference",
    )

    with pytest.raises(
        ValidationError
    ):
        SERSContextReview(
            review_id="review:bad",
            hypothesis_id="hypothesis:right",
            signatures=[
                left,
                right,
            ],
            findings=[
                finding
            ],
            status="pass",
        )
