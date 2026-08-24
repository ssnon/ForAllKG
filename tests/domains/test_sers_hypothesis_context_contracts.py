from __future__ import annotations

import pytest

from domains.sers.context_contracts import (
    SERSContextBinding,
    SERSContextFact,
    SERSContextProvenance,
    SERSContextSignature,
)
from domains.sers.hypothesis_context_contracts import (
    HypothesisContextAssertionDraft,
    HypothesisContextInterpretationCompiler,
    HypothesisContextInterpretationDraft,
    HypothesisContextInterpretationValidationError,
    HypothesisContextMentionDraft,
    expected_hypothesis_context_assertions,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisEvidenceProfile,
    PredictedObservation,
)


def _profile() -> HypothesisEvidenceProfile:
    return HypothesisEvidenceProfile(
        premise_count=1,
        gap_count=0,
        source_paper_count=1,
        candidate_premise_count=0,
        reported_premise_count=1,
        synthesis_premise_count=0,
    )


def _card() -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="hypothesis:test",
        domain_profile_id="sers_au_ag",
        source_context_id="context:test",
        source_context_sha256="sha-context",
        source_report_id="report:test",
        source_report_sha256="sha-report",
        title="test",
        hypothesis_statement=(
            "An inserted-pyramid-like nanogap "
            "geometry changes the local field."
        ),
        hypothesis_type="context_dependency",
        premise_statement_ids=[
            "stmt:1"
        ],
        inferential_bridge=(
            "The inserted pyramid is proposed "
            "within the nanogap."
        ),
        predicted_observations=[
            PredictedObservation(
                observation_id="prediction:1",
                observable=(
                    "Local field changes with "
                    "inserted-pyramid nanogap geometry"
                ),
                expected_direction=(
                    "qualitative_change"
                ),
                rationale=(
                    "Geometry is treated as a "
                    "field-control variable."
                ),
            )
        ],
        falsification_criteria=[],
        assumptions=[
            (
                "Composition is held comparable "
                "while geometry changes."
            )
        ],
        evidence_profile=_profile(),
    )


def _source_signature() -> SERSContextSignature:
    return SERSContextSignature(
        signature_id="sig:axis",
        domain_profile_id="sers_au_ag",
        scope="axis_inspiration",
        source_ref_id="axis:test",
        facts=[
            SERSContextFact(
                fact_id="fact:pyramid",
                dimension="morphology",
                scientific_role="morphology",
                knowledge_state="explicit",
                value=(
                    "3D-Si with inserted small pyramid"
                ),
                binding=SERSContextBinding(
                    basis="direct_edge",
                    owner_ref_id="node:si",
                    owner_label="3D-Si substrate",
                    owner_type="PlasmonicSubstrate",
                    relation="HAS_MORPHOLOGY",
                ),
                provenance=[
                    SERSContextProvenance(
                        kind="axis_structural_edge",
                        node_ids=[
                            "node:si",
                            "node:pyramid",
                        ],
                        edge_ids=[
                            "edge:pyramid"
                        ],
                    )
                ],
            ),
            SERSContextFact(
                fact_id="fact:gap",
                dimension="gap_regime",
                scientific_role="gap_regime",
                knowledge_state="explicit",
                value="10 nm nanoparticle gap",
                binding=SERSContextBinding(
                    basis="direct_edge",
                    owner_ref_id="node:auag",
                    owner_label="Au@Ag nanoparticle",
                    owner_type="Nanostructure",
                    relation="HAS_STRUCTURAL_MOTIF",
                ),
                provenance=[
                    SERSContextProvenance(
                        kind="axis_structural_edge",
                        node_ids=[
                            "node:auag",
                            "node:gap",
                        ],
                        edge_ids=[
                            "edge:gap"
                        ],
                    )
                ],
            ),
        ],
    )


def _empty_assertions(
    card: HypothesisCard,
) -> list[
    HypothesisContextAssertionDraft
]:
    return [
        HypothesisContextAssertionDraft(
            assertion_id=row[
                "assertion_id"
            ],
            assertion_kind=row[
                "assertion_kind"
            ],
            assertion_text=row[
                "assertion_text"
            ],
            mentions=[],
        )
        for row in (
            expected_hypothesis_context_assertions(
                card
            )
        )
    ]


def test_expected_assertions_are_stable() -> None:
    card = _card()

    rows = (
        expected_hypothesis_context_assertions(
            card
        )
    )

    assert [
        row["assertion_id"]
        for row in rows
    ] == [
        "central:hypothesis:test",
        "bridge:hypothesis:test",
        "prediction:1",
        "assumption:hypothesis:test:0",
    ]


def test_h1_style_reattachment_is_representable() -> None:
    card = _card()
    signature = _source_signature()

    assertions = _empty_assertions(
        card
    )

    assertions[0].mentions = [
        HypothesisContextMentionDraft(
            mention_id="mention:h1:pyramid",
            mention_text=(
                "inserted-pyramid-like nanogap "
                "geometry"
            ),
            source_fact_ids=[
                "fact:pyramid"
            ],
            asserted_dimension="gap_regime",
            asserted_role="gap_regime",
            asserted_owner_label="nanogap geometry",
            asserted_owner_type="StructuralMotif",
            treatment="reattach",
            experimental_role=(
                "experimental_variable"
            ),
            rationale=(
                "The source pyramid morphology is "
                "attached to the nanogap context."
            ),
        )
    ]

    draft = HypothesisContextInterpretationDraft(
        hypothesis_id=card.hypothesis_id,
        source_signature_ids=[
            signature.signature_id
        ],
        assertions=assertions,
    )

    result = (
        HypothesisContextInterpretationCompiler()
        .compile(
            card=card,
            source_signatures=[
                signature
            ],
            draft=draft,
        )
    )

    mention = (
        result.assertions[0].mentions[0]
    )

    assert (
        mention.treatment
        == "reattach"
    )

    assert (
        mention.asserted_dimension
        == "gap_regime"
    )


def test_h2_style_intentional_variation_is_representable() -> None:
    card = _card()

    source = SERSContextSignature(
        signature_id="sig:cu",
        domain_profile_id="sers_au_ag",
        scope="axis_inspiration",
        source_ref_id="axis:cu",
        facts=[
            SERSContextFact(
                fact_id="fact:cu-substrate",
                dimension="substrate",
                scientific_role=(
                    "plasmonic_substrate"
                ),
                knowledge_state="explicit",
                value=(
                    "Copper substrate with "
                    "nanostructured gold"
                ),
                binding=SERSContextBinding(
                    basis="node",
                    owner_ref_id="node:cu-au",
                    owner_label=(
                        "Copper substrate with "
                        "nanostructured gold"
                    ),
                    owner_type=(
                        "PlasmonicSubstrate"
                    ),
                ),
                provenance=[
                    SERSContextProvenance(
                        kind="axis_anchor",
                        node_ids=[
                            "node:cu-au"
                        ],
                    )
                ],
            )
        ],
    )

    assertions = _empty_assertions(
        card
    )

    # Use an exact span that exists in this synthetic central assertion.
    assertions[0].mentions = [
        HypothesisContextMentionDraft(
            mention_id="mention:cu",
            mention_text="nanogap geometry",
            source_fact_ids=[
                "fact:cu-substrate"
            ],
            asserted_dimension="substrate",
            asserted_role=(
                "plasmonic_substrate"
            ),
            asserted_owner_label=(
                "experimental substrate condition"
            ),
            asserted_owner_type=(
                "PlasmonicSubstrate"
            ),
            treatment=(
                "intentionally_vary"
            ),
            experimental_role="moderator",
            rationale=(
                "The substrate context is treated "
                "as the comparison variable."
            ),
        )
    ]

    draft = HypothesisContextInterpretationDraft(
        hypothesis_id=card.hypothesis_id,
        source_signature_ids=[
            source.signature_id
        ],
        assertions=assertions,
    )

    result = (
        HypothesisContextInterpretationCompiler()
        .compile(
            card=card,
            source_signatures=[
                source
            ],
            draft=draft,
        )
    )

    assert (
        result.assertions[
            0
        ].mentions[
            0
        ].treatment
        == "intentionally_vary"
    )


def test_preserve_cannot_change_dimension() -> None:
    card = _card()
    signature = _source_signature()
    assertions = _empty_assertions(
        card
    )

    assertions[0].mentions = [
        HypothesisContextMentionDraft(
            mention_id="mention:bad",
            mention_text="nanogap geometry",
            source_fact_ids=[
                "fact:pyramid"
            ],
            asserted_dimension="gap_regime",
            asserted_role="gap_regime",
            treatment="preserve",
            rationale="invalid preservation",
        )
    ]

    draft = HypothesisContextInterpretationDraft(
        hypothesis_id=card.hypothesis_id,
        source_signature_ids=[
            signature.signature_id
        ],
        assertions=assertions,
    )

    with pytest.raises(
        HypothesisContextInterpretationValidationError,
        match="cannot silently change context dimension",
    ):
        (
            HypothesisContextInterpretationCompiler()
            .compile(
                card=card,
                source_signatures=[
                    signature
                ],
                draft=draft,
            )
        )


def test_unknown_source_fact_is_rejected() -> None:
    card = _card()
    signature = _source_signature()
    assertions = _empty_assertions(
        card
    )

    assertions[0].mentions = [
        HypothesisContextMentionDraft(
            mention_id="mention:unknown",
            mention_text="nanogap geometry",
            source_fact_ids=[
                "fact:not-real"
            ],
            asserted_dimension="gap_regime",
            asserted_role="gap_regime",
            treatment="generalize",
            rationale="bad source reference",
        )
    ]

    draft = HypothesisContextInterpretationDraft(
        hypothesis_id=card.hypothesis_id,
        source_signature_ids=[
            signature.signature_id
        ],
        assertions=assertions,
    )

    with pytest.raises(
        HypothesisContextInterpretationValidationError,
        match="unknown source fact",
    ):
        (
            HypothesisContextInterpretationCompiler()
            .compile(
                card=card,
                source_signatures=[
                    signature
                ],
                draft=draft,
            )
        )


def test_missing_assertion_is_rejected() -> None:
    card = _card()
    signature = _source_signature()

    assertions = _empty_assertions(
        card
    )[:-1]

    draft = HypothesisContextInterpretationDraft(
        hypothesis_id=card.hypothesis_id,
        source_signature_ids=[
            signature.signature_id
        ],
        assertions=assertions,
    )

    with pytest.raises(
        HypothesisContextInterpretationValidationError,
        match="missing assertions",
    ):
        (
            HypothesisContextInterpretationCompiler()
            .compile(
                card=card,
                source_signatures=[
                    signature
                ],
                draft=draft,
            )
        )


def test_non_span_mention_is_rejected() -> None:
    card = _card()
    signature = _source_signature()
    assertions = _empty_assertions(
        card
    )

    assertions[0].mentions = [
        HypothesisContextMentionDraft(
            mention_id="mention:invented",
            mention_text=(
                "phrase that does not occur"
            ),
            source_fact_ids=[
                "fact:gap"
            ],
            asserted_dimension="gap_regime",
            asserted_role="gap_regime",
            treatment="generalize",
            rationale="invented span",
        )
    ]

    draft = HypothesisContextInterpretationDraft(
        hypothesis_id=card.hypothesis_id,
        source_signature_ids=[
            signature.signature_id
        ],
        assertions=assertions,
    )

    with pytest.raises(
        HypothesisContextInterpretationValidationError,
        match="mention_text is not an exact assertion span",
    ):
        (
            HypothesisContextInterpretationCompiler()
            .compile(
                card=card,
                source_signatures=[
                    signature
                ],
                draft=draft,
            )
        )


def test_combine_requires_multiple_source_facts() -> None:
    with pytest.raises(
        Exception,
        match=(
            "combine requires at least "
            "two source facts"
        ),
    ):
        HypothesisContextMentionDraft(
            mention_id="mention:combine",
            mention_text="nanogap geometry",
            source_fact_ids=[
                "fact:gap"
            ],
            asserted_dimension="gap_regime",
            asserted_role="gap_regime",
            treatment="combine",
            rationale="invalid combine",
        )


def test_introduce_cannot_claim_source_fact_support() -> None:
    with pytest.raises(
        Exception,
        match=(
            "introduce must not claim "
            "source fact support"
        ),
    ):
        HypothesisContextMentionDraft(
            mention_id="mention:introduce",
            mention_text="new context",
            source_fact_ids=[
                "fact:gap"
            ],
            asserted_dimension="environment",
            asserted_role="environment",
            treatment="introduce",
            rationale="invalid introduction",
        )
