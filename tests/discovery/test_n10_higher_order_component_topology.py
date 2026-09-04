import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
)
from pipeline_core.discovery.external_novelty_llm import (
    _DECOMPOSE_SYSTEM,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisEvidenceProfile,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    NoveltyClaimDecomposer,
    _compile_higher_order_component_claim_ids,
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
        hypothesis_id="hypothesis:topology",
        domain_profile_id="synthetic",
        source_context_id="context:1",
        source_context_sha256="ctxsha",
        source_report_id="report:1",
        source_report_sha256="repsha",
        title="Explicit linked relation",
        hypothesis_statement=(
            "A may relate to C through B as one "
            "linked relation."
        ),
        hypothesis_type="descriptor_mediation",
        premise_statement_ids=["statement:1"],
        inferential_bridge=(
            "The proposed linked relation connects "
            "A to B to C."
        ),
        predicted_observations=[],
        falsification_criteria=[],
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


def test_prompt_defines_explicit_component_topology():
    text = _DECOMPOSE_SYSTEM

    assert (
        "COMPOSITE-COMPONENT TOPOLOGY CONTRACT:"
        in text
    )

    assert (
        "higher_order_component_local_ids"
        in text
    )

    assert (
        "structural membership only"
        in text
    )


def test_component_ids_resolve_without_semantic_inference():
    mapping = {
        "a-b": "claim:ab",
        "b-c": "claim:bc",
        "abc": "claim:abc",
    }

    result = (
        _compile_higher_order_component_claim_ids(
            kind="composite",
            local_id="abc",
            component_local_ids=[
                "a-b",
                "b-c",
            ],
            claim_id_by_local_id=mapping,
        )
    )

    assert result == [
        "claim:ab",
        "claim:bc",
    ]


def test_non_composite_cannot_declare_components():
    with pytest.raises(
        ValueError,
        match="non-composite",
    ):
        _compile_higher_order_component_claim_ids(
            kind="mediator",
            local_id="a-b",
            component_local_ids=["b-c"],
            claim_id_by_local_id={
                "a-b": "claim:ab",
                "b-c": "claim:bc",
            },
        )


def test_composite_cannot_reference_itself():
    with pytest.raises(
        ValueError,
        match="cannot reference itself",
    ):
        _compile_higher_order_component_claim_ids(
            kind="composite",
            local_id="abc",
            component_local_ids=["abc"],
            claim_id_by_local_id={
                "abc": "claim:abc",
            },
        )


def test_unknown_component_reference_is_rejected():
    with pytest.raises(
        ValueError,
        match="unknown higher-order component",
    ):
        _compile_higher_order_component_claim_ids(
            kind="composite",
            local_id="abc",
            component_local_ids=["missing"],
            claim_id_by_local_id={
                "abc": "claim:abc",
            },
        )


def test_duplicate_component_reference_is_rejected():
    with pytest.raises(
        ValueError,
        match="duplicate higher-order component",
    ):
        _compile_higher_order_component_claim_ids(
            kind="composite",
            local_id="abc",
            component_local_ids=[
                "a-b",
                "a-b",
            ],
            claim_id_by_local_id={
                "abc": "claim:abc",
                "a-b": "claim:ab",
            },
        )


def test_empty_component_list_is_valid_for_composite():
    result = (
        _compile_higher_order_component_claim_ids(
            kind="composite",
            local_id="abc",
            component_local_ids=[],
            claim_id_by_local_id={
                "abc": "claim:abc",
            },
        )
    )

    assert result == []


def test_decomposer_preserves_component_topology():
    hypothesis = _hypothesis()

    basis = (
        "The proposed linked relation connects "
        "A to B to C."
    )

    draft = NoveltyClaimDecompositionDraft(
        claims=[
            NoveltyClaimDraft(
                local_id="a-b",
                kind="mediator",
                importance="supporting",
                novelty_selection_role=(
                    "REQUIRED_ENABLING_RELATION"
                ),
                text="A is associated with B.",
                rationale="First component.",
            ),
            NoveltyClaimDraft(
                local_id="b-c",
                kind="mechanistic_link",
                importance="supporting",
                novelty_selection_role=(
                    "REQUIRED_ENABLING_RELATION"
                ),
                text="B is associated with C.",
                rationale="Second component.",
            ),
            NoveltyClaimDraft(
                local_id="abc",
                kind="composite",
                importance="core",
                novelty_selection_role=(
                    "NOVELTY_BEARING"
                ),
                text="A relates to C through B.",
                rationale="Explicit higher-order relation.",
                higher_order_relation_basis=[
                    basis
                ],
                higher_order_component_local_ids=[
                    "a-b",
                    "b-c",
                ],
            ),
        ]
    )

    result = NoveltyClaimDecomposer(
        _Backend(draft)
    ).decompose(
        hypothesis
    )

    by_kind = {
        claim.kind: claim
        for claim in result.claims
    }

    composite = by_kind["composite"]

    component_ids = {
        claim.claim_id
        for claim in result.claims
        if claim.kind != "composite"
    }

    assert set(
        composite.higher_order_component_claim_ids
    ) == component_ids

    assert composite.claim_id not in (
        composite.higher_order_component_claim_ids
    )
