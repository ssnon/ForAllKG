from pipeline_core.discovery.external_novelty_contracts import NoveltyClaim
from pipeline_core.discovery.prospective_routed_primary_materializer_v2 import (
    SourceAlignmentAuditDraft,
    _aligned_claim,
    _audit_passes,
)
from pipeline_core.discovery.prospective_routed_route_compiler_v2 import SourceAlignmentCandidatePairV2


def _claim():
    return NoveltyClaim(
        claim_id="claim:1",
        hypothesis_id="hypothesis:1",
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="X modulates Y.",
        rationale="rationale",
        prior_art_identity_terms=["X", "Y"],
        relation_nucleus_terms=["X", "Y"],
        required_bridge="X modulates Y.",
        predicted_observation="Old prediction.",
        falsification_condition="Old falsifier.",
    )


def _candidate():
    return SourceAlignmentCandidatePairV2(
        prediction_observation_id="prediction:1",
        prediction_observable="Y response",
        falsification_criterion_id="falsifier:1",
        falsifier_observable="Y response",
        falsifying_outcome="Y response is unchanged.",
    )


def test_alignment_uses_existing_source_surfaces_only():
    source = _claim()
    aligned = _aligned_claim(claim=source, candidate=_candidate())

    assert aligned.predicted_observation == "Y response"
    assert aligned.falsification_condition == "Y response is unchanged."

    # semantic_fidelity_binding belongs to NoveltyClaimDraft, not the
    # canonical NoveltyClaim. Source IDs are preserved in the alignment
    # lineage result instead of being injected into the scientific claim.
    assert not hasattr(aligned, "semantic_fidelity_binding")

    # Source alignment must not alter the scientific proposition itself.
    assert aligned.claim_id == source.claim_id
    assert aligned.hypothesis_id == source.hypothesis_id
    assert aligned.kind == source.kind
    assert aligned.novelty_selection_role == source.novelty_selection_role
    assert aligned.text == source.text
    assert aligned.rationale == source.rationale
    assert aligned.required_bridge == source.required_bridge
    assert aligned.prior_art_identity_terms == source.prior_art_identity_terms
    assert aligned.relation_nucleus_terms == source.relation_nucleus_terms


def test_audit_requires_every_zero_delta_dimension():
    audit = SourceAlignmentAuditDraft(
        claim_id="claim:1",
        zero_scientific_delta=True,
        same_relation_commitment=True,
        same_scope=True,
        same_direction=True,
        no_new_mechanism=True,
        no_new_moderator=True,
        no_new_scientific_entity=True,
        rationale="same",
    )
    assert _audit_passes(audit)
    assert not _audit_passes(audit.model_copy(update={"same_scope": False}))


def test_candidate_pair_is_source_surface_only():
    pair = _candidate()
    assert pair.shared_observable_identity is True
    assert pair.source_surfaces_only is True
