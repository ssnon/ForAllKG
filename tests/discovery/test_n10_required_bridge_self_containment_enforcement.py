from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
)
from pipeline_core.discovery.novelty_atomic_semantic_fidelity import (
    has_anaphoric_relation_reference,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    NoveltyClaimDecomposer,
)


def _run_bridge(raw_bridge: str):
    class Backend:
        def decompose(self, hypothesis, *, max_claims):
            return NoveltyClaimDecompositionDraft(
                claims=[
                    NoveltyClaimDraft(
                        local_id="c1",
                        kind="moderator_interaction",
                        importance="core",
                        text=(
                            "Scaffold localization conditions the association "
                            "between activation-deactivation compatibility "
                            "and cell survival."
                        ),
                        rationale="test",
                        prior_art_identity_terms=[
                            "activation-deactivation compatibility",
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
            hypothesis_id="hypothesis:self-containment",
            title="test",
            hypothesis_statement=(
                "Scaffold localization may condition signaling compatibility."
            ),
            inferential_bridge=raw_bridge,
            assumptions=[],
            predicted_observations=[],
            falsification_criteria=[],
        )
    )

    return result.claims[0], decomposer.specification_sanitization_records[0]


def test_demonstrative_relation_reference_is_rejected_even_when_exact_and_branch_specific():
    raw = (
        "I further propose that this localization-dependent modulation "
        "conditions the association between activation-deactivation "
        "compatibility and cell survival."
    )
    claim, record = _run_bridge(raw)

    assert claim.required_bridge == ""
    assert (
        "required_bridge_rejected_anaphoric_relation_reference"
        in claim.specification_sanitization_reason_codes
    )
    assert (
        "required_bridge_rejected_anaphoric_relation_reference"
        in record["reason_codes"]
    )
    assert record["raw_required_bridge"] == raw
    assert record["sanitized_required_bridge"] == ""


def test_complementizer_that_does_not_trigger_anaphora_rejection():
    raw = (
        "I propose that activation-deactivation compatibility "
        "modulates cell survival."
    )
    claim, record = _run_bridge(raw)

    assert claim.required_bridge == raw
    assert (
        "required_bridge_rejected_anaphoric_relation_reference"
        not in record["reason_codes"]
    )


def test_definite_self_contained_relation_phrase_is_not_anaphoric():
    assert not has_anaphoric_relation_reference(
        "The relationship between activation-deactivation compatibility "
        "and cell survival changes across scaffold-localization states."
    )


def test_demonstrative_relation_heads_are_anaphoric():
    assert has_anaphoric_relation_reference("this relationship")
    assert has_anaphoric_relation_reference("that modulation")
    assert has_anaphoric_relation_reference("these pathway interactions")
    assert has_anaphoric_relation_reference("such localization-dependent effect")
