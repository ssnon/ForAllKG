from __future__ import annotations

import copy
import inspect

import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    NoveltyClaim,
    NoveltyClaimDraft,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    NoveltyClaimDecomposer,
    _apply_existing_bridge_scope_alignment_canonical_action_opt_in,
)


SOURCE = (
    "Under the condition of comparable overall metal–hydrogen coupling, "
    "I further propose that a more balanced bonding–antibonding distribution "
    "promotes compatibility between hydrogen adsorption and H2 desorption."
)
CLAIM_TEXT = (
    "At comparable overall metal–hydrogen coupling, a more balanced "
    "bonding–antibonding distribution promotes compatibility between "
    "hydrogen adsorption and H2 desorption."
)
SOURCE_SCOPE = (
    "Under the condition of comparable overall metal–hydrogen coupling"
)
PREDICTION = (
    "At comparable overall metal–hydrogen coupling, a more balanced "
    "bonding–antibonding distribution is expected to be more compatible "
    "with both hydrogen adsorption and H2 desorption."
)
FALSIFIER = (
    "Changing the bonding–antibonding distribution at comparable overall "
    "metal–hydrogen coupling produces no qualitative difference in hydrogen "
    "adsorption–desorption compatibility."
)


def _card() -> HypothesisCard:
    return HypothesisCard.model_validate(
        {
            "hypothesis_id": "hypothesis:s15n",
            "domain_profile_id": "domain:s15n",
            "source_context_id": "context:s15n",
            "source_context_sha256": "context-sha",
            "source_report_id": "report:s15n",
            "source_report_sha256": "report-sha",
            "title": "Alignment canonical opt-in fixture",
            "hypothesis_statement": SOURCE,
            "hypothesis_type": "context_dependency",
            "premise_statement_ids": ["premise:1"],
            "gap_statement_ids": ["gap:1"],
            "inferential_bridge": SOURCE,
            "predicted_observations": [
                {
                    "observation_id": "obs:1",
                    "observable": (
                        "hydrogen adsorption–desorption compatibility"
                    ),
                    "expected_direction": "qualitative_change",
                    "rationale": PREDICTION,
                }
            ],
            "falsification_criteria": [
                {
                    "criterion_id": "fals:1",
                    "observable": (
                        "hydrogen adsorption–desorption compatibility"
                    ),
                    "falsifying_outcome": FALSIFIER,
                }
            ],
            "assumptions": [],
            "source_paper_ids": ["paper:synthetic"],
            "gap_paper_ids": [],
            "cross_paper_synthesis": False,
            "candidate_dependency": "none",
            "evidence_profile": {
                "premise_count": 1,
                "gap_count": 1,
                "source_paper_count": 1,
                "candidate_premise_count": 0,
                "reported_premise_count": 1,
                "synthesis_premise_count": 0,
            },
            "status": "hypothesized",
            "novelty_status": "not_assessed",
        }
    )


def _draft(
    *,
    text: str = CLAIM_TEXT,
    bridge: str = SOURCE,
) -> NoveltyClaimDraft:
    return NoveltyClaimDraft.model_validate(
        {
            "local_id": "claim_1",
            "kind": "mechanistic_link",
            "importance": "core",
            "novelty_selection_role": "NOVELTY_BEARING",
            "text": text,
            "rationale": "Synthetic alignment fixture only.",
            "search_concepts": [
                "bonding antibonding distribution"
            ],
            "search_queries": [
                "bonding antibonding distribution hydrogen"
            ],
            "distinguishing_terms": [
                "comparable overall metal hydrogen coupling"
            ],
            "prior_art_identity_terms": [
                "bonding antibonding distribution"
            ],
            "relation_nucleus_terms": [
                "overall metal hydrogen coupling",
                "hydrogen adsorption",
                "H2 desorption",
                "compatibility",
            ],
            "semantic_fidelity_binding": {
                "proposition_basis": SOURCE,
                "relation_endpoint_anchors": [
                    "bonding-antibonding distribution",
                    "hydrogen adsorption",
                    "H2 desorption",
                ],
                "scope_qualifier_spans": [SOURCE_SCOPE],
                "directional_qualifier_spans": [
                    "a more balanced bonding-antibonding distribution",
                    "promotes compatibility",
                ],
                "prediction_observation_id": "obs:1",
                "falsification_criterion_id": "fals:1",
            },
            "required_bridge": bridge,
            "predicted_observation": PREDICTION,
            "falsification_condition": FALSIFIER,
        }
    )


def _canonical(
    draft: NoveltyClaimDraft,
    *,
    claim_id: str = "external_novelty_claim:s15n",
    text: str | None = None,
    bridge: str | None = None,
) -> NoveltyClaim:
    return NoveltyClaim.model_validate(
        {
            "claim_id": claim_id,
            "hypothesis_id": "hypothesis:s15n",
            "claim_rank": 1,
            "kind": draft.kind,
            "importance": draft.importance,
            "novelty_selection_role": (
                draft.novelty_selection_role
            ),
            "text": draft.text if text is None else text,
            "rationale": draft.rationale,
            "search_concepts": list(draft.search_concepts),
            "search_queries": list(draft.search_queries),
            "distinguishing_terms": list(
                draft.distinguishing_terms
            ),
            "prior_art_identity_terms": list(
                draft.prior_art_identity_terms
            ),
            "relation_nucleus_terms": list(
                draft.relation_nucleus_terms
            ),
            "required_bridge": (
                draft.required_bridge
                if bridge is None
                else bridge
            ),
            "predicted_observation": (
                draft.predicted_observation
            ),
            "falsification_condition": (
                draft.falsification_condition
            ),
        }
    )


def _container(
    canonical: NoveltyClaim,
) -> HypothesisNoveltyClaims:
    return HypothesisNoveltyClaims.model_validate(
        {
            "hypothesis_id": "hypothesis:s15n",
            "title": "Alignment canonical opt-in fixture",
            "claims": [
                canonical.model_dump(mode="json")
            ],
            "decomposition_notes": "synthetic",
        }
    )


def test_alignment_canonical_action_opt_in_is_text_only_and_auditable():
    card = _card()
    draft = _draft()
    canonical = _canonical(draft)
    container = _container(canonical)

    draft_before = draft.model_dump(mode="json")
    container_before = container.model_dump(mode="json")

    result, action = (
        _apply_existing_bridge_scope_alignment_canonical_action_opt_in(
            card,
            container,
            draft,
            claim_id=canonical.claim_id,
        )
    )

    assert result.claims[0].text == SOURCE
    assert result.claims[0].required_bridge == SOURCE
    assert result.claims[0].claim_id == canonical.claim_id
    assert result.claims[0].claim_rank == canonical.claim_rank
    assert (
        result.claims[0].novelty_selection_role
        == "NOVELTY_BEARING"
    )

    before_claim = container_before["claims"][0]
    after_claim = result.model_dump(mode="json")["claims"][0]
    changed = sorted(
        key
        for key in set(before_claim) | set(after_claim)
        if before_claim.get(key) != after_claim.get(key)
    )
    assert changed == ["text"]

    assert action["action_mode"] == "EXPLICIT_OPT_IN_ONLY"
    assert action["explicit_opt_in_required"] is True
    assert action["automatic_runtime_enabled"] is False
    assert action["canonical_action_performed"] is True
    assert (
        action["source_recovery_path"]
        == "EXISTING_BRIDGE_SCOPE_ALIGNMENT"
    )
    assert (
        action["shadow_authority_status"]
        == "AUTHORIZED_SHADOW"
    )
    assert action["shadow_authority_reason_codes"] == []
    assert action["canonical_changed_fields"] == ["text"]
    assert (
        action["all_non_text_canonical_fields_preserved"]
        is True
    )
    assert action["input_objects_mutated"] is False

    assert draft.model_dump(mode="json") == draft_before
    assert container.model_dump(mode="json") == container_before


def test_alignment_canonical_action_opt_in_fails_on_stale_canonical_text():
    card = _card()
    draft = _draft()
    canonical = _canonical(
        draft,
        text="A stale canonical text that was not previewed.",
    )
    container = _container(canonical)

    with pytest.raises(
        ValueError,
        match="stale source/canonical text mismatch",
    ):
        _apply_existing_bridge_scope_alignment_canonical_action_opt_in(
            card,
            container,
            draft,
            claim_id=canonical.claim_id,
        )


def test_alignment_canonical_action_opt_in_fails_if_current_shadow_denies():
    card = _card()
    draft = _draft(
        text=(
            "At higher overall metal–hydrogen coupling, a more balanced "
            "bonding–antibonding distribution promotes compatibility between "
            "hydrogen adsorption and H2 desorption."
        )
    )
    canonical = _canonical(draft)
    container = _container(canonical)

    with pytest.raises(ValueError):
        _apply_existing_bridge_scope_alignment_canonical_action_opt_in(
            card,
            container,
            draft,
            claim_id=canonical.claim_id,
        )


def test_alignment_canonical_action_opt_in_fails_on_canonical_bridge_mismatch():
    card = _card()
    draft = _draft()
    canonical = _canonical(
        draft,
        bridge="",
    )
    container = _container(canonical)

    with pytest.raises(
        ValueError,
        match="canonical bridge does not match current sanitizer output",
    ):
        _apply_existing_bridge_scope_alignment_canonical_action_opt_in(
            card,
            container,
            draft,
            claim_id=canonical.claim_id,
        )


def test_alignment_canonical_action_opt_in_fails_on_unknown_claim_id():
    card = _card()
    draft = _draft()
    canonical = _canonical(draft)
    container = _container(canonical)

    with pytest.raises(
        ValueError,
        match="requires exactly one canonical target",
    ):
        _apply_existing_bridge_scope_alignment_canonical_action_opt_in(
            card,
            container,
            draft,
            claim_id="external_novelty_claim:missing",
        )


def test_default_decompose_path_keeps_alignment_action_disabled_by_default():
    signature = inspect.signature(NoveltyClaimDecomposer.__init__)
    parameter = signature.parameters[
        "enable_existing_bridge_scope_alignment_canonical_action"
    ]

    assert parameter.default is False

    source = inspect.getsource(NoveltyClaimDecomposer.decompose)
    assert (
        "_apply_existing_bridge_scope_alignment_canonical_action_opt_in"
        in source
    )
    assert (
        "if not ("
        in source
    )
    assert (
        "self.enable_existing_bridge_scope_alignment_canonical_action"
        in source
    )
