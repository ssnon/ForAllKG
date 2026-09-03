from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    NoveltyClaimDecomposer,
)


def test_raw_specification_is_preserved_only_in_diagnostic_record() -> None:
    raw_bridge = (
        "iCOHP controls hydrogen adsorption free energy."
    )

    hypothesis_bridge = (
        "Metal identity conditions the mapping from M-H iCOHP "
        "to hydrogen adsorption free energy."
    )

    class Backend:
        def decompose(
            self,
            hypothesis,
            *,
            max_claims,
        ):
            return NoveltyClaimDecompositionDraft(
                claims=[
                    NoveltyClaimDraft(
                        local_id="c1",
                        kind="moderator_interaction",
                        importance="core",
                        text=(
                            "Metal identity conditions the "
                            "relationship between iCOHP and "
                            "hydrogen adsorption free energy."
                        ),
                        rationale="test",
                        prior_art_identity_terms=[
                            "metal identity",
                        ],
                        relation_nucleus_terms=[
                            "iCOHP",
                            "hydrogen adsorption free energy",
                        ],
                        required_bridge=raw_bridge,
                    )
                ]
            )

    decomposer = NoveltyClaimDecomposer(
        Backend(),
        max_claims_per_hypothesis=1,
        max_queries_per_claim=2,
    )

    result = decomposer.decompose(
        SimpleNamespace(
            hypothesis_id=(
                "hypothesis:test-sanitization"
            ),
            title="test",
            hypothesis_statement=(
                "Metal identity may moderate "
                "adsorption energetics."
            ),
            inferential_bridge=hypothesis_bridge,
            assumptions=[],
            predicted_observations=[],
            falsification_criteria=[],
        )
    )

    claim = result.claims[0]

    # Canonical scientific contract remains sanitized.
    assert claim.required_bridge == ""

    assert (
        "required_bridge_rejected_branch_identity"
        in claim.specification_sanitization_reason_codes
    )

    # Raw content survives only in diagnostic provenance.
    records = (
        decomposer.specification_sanitization_records
    )

    assert len(records) == 1

    record = records[0]

    assert record["diagnostic_only"] is True
    assert record["claim_id"] == claim.claim_id
    assert (
        record["required_bridge_source"]
        == "draft"
    )
    assert (
        record["raw_required_bridge"]
        == raw_bridge
    )
    assert (
        record["sanitized_required_bridge"]
        == ""
    )
    assert record["prior_art_identity_terms"] == [
        "metal identity",
    ]

    assert (
        "required_bridge_rejected_branch_identity"
        in record["reason_codes"]
    )

    # Raw rejected text must not enter the canonical claim.
    payload = claim.model_dump(mode="json")

    assert "raw_required_bridge" not in payload
    assert "raw_predicted_observation" not in payload
    assert "raw_falsification_condition" not in payload


def test_required_bridge_prompt_requires_self_contained_contiguous_branch_span() -> None:
    from pipeline_core.discovery.external_novelty_llm import (
        _DECOMPOSE_SYSTEM,
    )

    prompt = _DECOMPOSE_SYSTEM

    assert (
        "REQUIRED-BRIDGE RETURN SELF-CHECK"
        in prompt
    )

    assert (
        "RETURNED BRIDGE STRING ITSELF"
        in prompt
    )

    assert (
        "ONE CONTIGUOUS EXTRACTIVE SPAN"
        in prompt
    )

    assert (
        "Do not stitch together non-contiguous fragments"
        in prompt
    )

    assert (
        "Do not expand to a larger umbrella sentence"
        in prompt
    )

    assert (
        "return required_bridge as an empty string"
        in prompt
    )


def test_empty_atomic_bridge_does_not_fallback_to_hypothesis_bridge() -> None:
    from types import SimpleNamespace

    from pipeline_core.discovery.external_novelty_contracts import (
        NoveltyClaimDecompositionDraft,
        NoveltyClaimDraft,
    )
    from pipeline_core.discovery.novelty_claim_decomposition import (
        NoveltyClaimDecomposer,
    )

    hypothesis_bridge = (
        "Metal identity conditions the relationship between "
        "iCOHP and hydrogen adsorption free energy."
    )

    class Backend:
        def decompose(
            self,
            hypothesis,
            *,
            max_claims,
        ):
            return NoveltyClaimDecompositionDraft(
                claims=[
                    NoveltyClaimDraft(
                        local_id="c1",
                        kind="moderator_interaction",
                        importance="core",
                        text=(
                            "Metal identity moderates the "
                            "relationship between iCOHP and "
                            "hydrogen adsorption free energy."
                        ),
                        rationale="test",
                        prior_art_identity_terms=[
                            "metal identity",
                        ],
                        relation_nucleus_terms=[
                            "iCOHP",
                            "hydrogen adsorption free energy",
                        ],
                        required_bridge="",
                    )
                ]
            )

    decomposer = NoveltyClaimDecomposer(
        Backend(),
        max_claims_per_hypothesis=1,
        max_queries_per_claim=2,
    )

    result = decomposer.decompose(
        SimpleNamespace(
            hypothesis_id=(
                "hypothesis:no-bridge-fallback"
            ),
            title="test",
            hypothesis_statement=(
                "Metal identity may moderate "
                "adsorption energetics."
            ),
            inferential_bridge=hypothesis_bridge,
            assumptions=[],
            predicted_observations=[],
            falsification_criteria=[],
        )
    )

    claim = result.claims[0]

    # Atomic empty is authoritative. The umbrella hypothesis
    # bridge must not be promoted into the claim.
    assert claim.required_bridge == ""

    records = (
        decomposer.specification_sanitization_records
    )

    assert len(records) == 1

    record = records[0]

    assert (
        record["required_bridge_source"]
        == "empty"
    )
    assert record["raw_required_bridge"] == ""
    assert (
        record["sanitized_required_bridge"]
        == ""
    )

    assert (
        "required_bridge_source_empty"
        in record["reason_codes"]
    )

    assert (
        "required_bridge_source_hypothesis_fallback"
        not in record["reason_codes"]
    )

    payload = claim.model_dump(mode="json")

    assert (
        payload["required_bridge"]
        == ""
    )
