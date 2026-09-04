from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
)
from pipeline_core.discovery.external_novelty_llm import (
    _DECOMPOSE_SYSTEM,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    PredictedObservation,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    NoveltyClaimDecomposer,
    _compile_higher_order_relation_basis,
)


class _Backend:
    def __init__(self, draft):
        self.draft = draft

    def decompose(
        self,
        hypothesis,
        *,
        max_claims,
    ):
        return self.draft


def _hypothesis():
    return HypothesisCard(
        hypothesis_id="hypothesis:composition",
        domain_profile_id="synthetic",
        source_context_id="context:1",
        source_context_sha256="ctxsha",
        source_report_id="report:1",
        source_report_sha256="repsha",
        title="Explicit mediation",
        hypothesis_statement=(
            "Descriptor A may affect outcome C through "
            "mediator B as a linked mediation relation."
        ),
        hypothesis_type="descriptor_mediation",
        premise_statement_ids=["statement:1"],
        inferential_bridge=(
            "The proposed linked relation is A to B to C, "
            "with B mediating the connection."
        ),
        predicted_observations=[
            PredictedObservation(
                observation_id="obs:1",
                observable=(
                    "Mediator B provides information about "
                    "outcome C beyond descriptor A alone."
                ),
                expected_direction="unspecified",
                rationale=(
                    "This tests the linked mediation relation."
                ),
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:1",
                observable=(
                    "Mediator B provides information about "
                    "outcome C beyond descriptor A alone."
                ),
                falsifying_outcome=(
                    "B adds no information about C once A "
                    "is considered."
                ),
            )
        ],
        assumptions=[],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )


def test_prompt_explicitly_forbids_component_to_composite_invention():
    text = _DECOMPOSE_SYSTEM

    assert (
        "EXPLICIT HIGHER-ORDER RELATION "
        "PRESERVATION CONTRACT:"
        in text
    )

    assert (
        "A -> B plus B -> C does not authorize "
        "invention of A -> B -> C."
        in text
    )

    assert (
        "EXACT CONTIGUOUS source spans"
        in text
    )


def test_explicit_composite_basis_is_preserved():
    hypothesis = _hypothesis()

    basis = (
        "The proposed linked relation is A to B to C, "
        "with B mediating the connection."
    )

    accepted, reasons = (
        _compile_higher_order_relation_basis(
            kind="composite",
            values=[basis],
            source_texts=[
                hypothesis.hypothesis_statement,
                hypothesis.inferential_bridge,
            ],
        )
    )

    assert accepted == [basis]
    assert reasons == []


def test_pairwise_components_cannot_manufacture_composite():
    accepted, reasons = (
        _compile_higher_order_relation_basis(
            kind="composite",
            values=[
                "A is related to C through B."
            ],
            source_texts=[
                "A is related to B.",
                "B is related to C.",
            ],
        )
    )

    assert accepted == []

    assert (
        "unsupported_higher_order_relation_basis"
        in reasons
    )

    assert (
        "composite_missing_valid_"
        "higher_order_relation_basis"
        in reasons
    )


def test_higher_order_basis_rejected_on_non_composite_claim():
    basis = (
        "The proposed linked relation is A to B to C, "
        "with B mediating the connection."
    )

    accepted, reasons = (
        _compile_higher_order_relation_basis(
            kind="mediator",
            values=[basis],
            source_texts=[basis],
        )
    )

    assert accepted == []

    assert reasons == [
        "higher_order_basis_rejected_on_non_composite_claim"
    ]


def test_composite_without_basis_fails_closed():
    accepted, reasons = (
        _compile_higher_order_relation_basis(
            kind="composite",
            values=[],
            source_texts=[
                "A is related to B.",
                "B is related to C.",
            ],
        )
    )

    assert accepted == []

    assert reasons == [
        "composite_missing_valid_higher_order_relation_basis"
    ]


def test_decomposer_preserves_explicit_composite_provenance():
    hypothesis = _hypothesis()

    basis = (
        "The proposed linked relation is A to B to C, "
        "with B mediating the connection."
    )

    draft = NoveltyClaimDecompositionDraft(
        claims=[
            NoveltyClaimDraft(
                local_id="composite",
                kind="composite",
                importance="core",
                novelty_selection_role=(
                    "NOVELTY_BEARING"
                ),
                text=(
                    "Descriptor A relates to outcome C "
                    "through mediator B."
                ),
                rationale=(
                    "The supplied hypothesis explicitly "
                    "proposes the linked mediation."
                ),
                higher_order_relation_basis=[
                    basis
                ],
                predicted_observation=(
                    "Mediator B provides information about "
                    "outcome C beyond descriptor A alone."
                ),
                falsification_condition=(
                    "B adds no information about C once A "
                    "is considered."
                ),
            )
        ]
    )

    result = NoveltyClaimDecomposer(
        _Backend(draft),
        max_claims_per_hypothesis=4,
        max_queries_per_claim=2,
    ).decompose(
        hypothesis
    )

    assert len(result.claims) == 1

    claim = result.claims[0]

    assert claim.kind == "composite"

    assert (
        claim.novelty_selection_role
        == "NOVELTY_BEARING"
    )

    assert claim.higher_order_relation_basis == [
        basis
    ]

    assert (
        claim.higher_order_relation_reason_codes
        == []
    )


def test_unsourced_composite_provenance_is_removed():
    hypothesis = _hypothesis()

    invented = (
        "A and C exhibit a new threshold mediated by B."
    )

    draft = NoveltyClaimDecompositionDraft(
        claims=[
            NoveltyClaimDraft(
                local_id="composite",
                kind="composite",
                importance="core",
                novelty_selection_role=(
                    "NOVELTY_BEARING"
                ),
                text=(
                    "Descriptor A relates to outcome C "
                    "through mediator B."
                ),
                rationale="Synthetic control.",
                higher_order_relation_basis=[
                    invented
                ],
            )
        ]
    )

    result = NoveltyClaimDecomposer(
        _Backend(draft)
    ).decompose(
        hypothesis
    )

    claim = result.claims[0]

    assert claim.higher_order_relation_basis == []

    assert (
        "unsupported_higher_order_relation_basis"
        in claim.higher_order_relation_reason_codes
    )

    assert (
        "composite_missing_valid_"
        "higher_order_relation_basis"
        in claim.higher_order_relation_reason_codes
    )
