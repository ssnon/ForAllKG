from __future__ import annotations

import copy

import pytest

import pipeline_core.discovery.novelty_claim_decomposition as decomposition_module
from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard
from pipeline_core.discovery.novelty_claim_decomposition import (
    NoveltyClaimDecomposer,
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
            "hypothesis_id": "hypothesis:s15p",
            "domain_profile_id": "domain:s15p",
            "source_context_id": "context:s15p",
            "source_context_sha256": "context-sha",
            "source_report_id": "report:s15p",
            "source_report_sha256": "report-sha",
            "title": "Disabled-by-default alignment runtime fixture",
            "hypothesis_statement": SOURCE,
            "hypothesis_type": "context_dependency",
            "premise_statement_ids": ["premise:1"],
            "gap_statement_ids": ["gap:1"],
            "inferential_bridge": SOURCE,
            "predicted_observations": [
                {
                    "observation_id": "obs:1",
                    "observable": "hydrogen adsorption–desorption compatibility",
                    "expected_direction": "qualitative_change",
                    "rationale": PREDICTION,
                }
            ],
            "falsification_criteria": [
                {
                    "criterion_id": "fals:1",
                    "observable": "hydrogen adsorption–desorption compatibility",
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


def _draft(*, text: str = CLAIM_TEXT) -> NoveltyClaimDraft:
    return NoveltyClaimDraft.model_validate(
        {
            "local_id": "claim_1",
            "kind": "mechanistic_link",
            "importance": "core",
            "novelty_selection_role": "NOVELTY_BEARING",
            "text": text,
            "rationale": "Synthetic runtime-option fixture only.",
            "search_concepts": ["bonding antibonding distribution"],
            "search_queries": ["bonding antibonding distribution hydrogen"],
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
            "required_bridge": SOURCE,
            "predicted_observation": PREDICTION,
            "falsification_condition": FALSIFIER,
        }
    )


class _Backend:
    def __init__(self, draft: NoveltyClaimDraft) -> None:
        self.draft = draft

    def decompose(self, hypothesis, *, max_claims):
        return NoveltyClaimDecompositionDraft(
            claims=[
                NoveltyClaimDraft.model_validate(
                    self.draft.model_dump(mode="json")
                )
            ],
            decomposition_notes="synthetic",
        )


def _decomposer(
    draft: NoveltyClaimDraft,
    *,
    enabled: bool | None = None,
) -> NoveltyClaimDecomposer:
    kwargs = {
        "max_claims_per_hypothesis": 1,
        "max_queries_per_claim": 2,
    }
    if enabled is not None:
        kwargs[
            "enable_existing_bridge_scope_alignment_canonical_action"
        ] = enabled
    return NoveltyClaimDecomposer(_Backend(draft), **kwargs)


def test_runtime_option_defaults_false_and_does_not_invoke_helper(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "disabled default path must not invoke canonical action helper"
        )

    monkeypatch.setattr(
        decomposition_module,
        "_apply_existing_bridge_scope_alignment_canonical_action_opt_in",
        forbidden,
    )

    card = _card()
    draft = _draft()
    card_before = card.model_dump(mode="json")
    draft_before = draft.model_dump(mode="json")

    decomposer = _decomposer(draft)
    result = decomposer.decompose(card)

    assert (
        decomposer.enable_existing_bridge_scope_alignment_canonical_action
        is False
    )
    assert result.claims[0].text == CLAIM_TEXT
    assert result.claims[0].required_bridge == SOURCE
    assert decomposer.canonical_action_records == []
    assert card.model_dump(mode="json") == card_before
    assert draft.model_dump(mode="json") == draft_before


def test_explicit_false_matches_omitted_default_byte_for_byte():
    card = _card()
    draft = _draft()

    omitted = _decomposer(draft).decompose(card)
    explicit_false = _decomposer(draft, enabled=False).decompose(card)

    assert (
        omitted.model_dump(mode="json")
        == explicit_false.model_dump(mode="json")
    )


def test_enabled_option_applies_one_text_only_action():
    card = _card()
    draft = _draft()

    control = _decomposer(draft, enabled=False).decompose(card)

    enabled = _decomposer(draft, enabled=True)
    treatment = enabled.decompose(card)

    assert treatment.claims[0].text == SOURCE
    assert treatment.claims[0].required_bridge == SOURCE
    assert treatment.claims[0].claim_id == control.claims[0].claim_id
    assert treatment.claims[0].claim_rank == control.claims[0].claim_rank
    assert (
        treatment.claims[0].novelty_selection_role
        == control.claims[0].novelty_selection_role
        == "NOVELTY_BEARING"
    )

    before = control.claims[0].model_dump(mode="json")
    after = treatment.claims[0].model_dump(mode="json")
    changed = sorted(
        key
        for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    )
    assert changed == ["text"]

    assert len(enabled.canonical_action_records) == 1
    action = enabled.canonical_action_records[0]
    assert action["action_mode"] == "EXPLICIT_OPT_IN_ONLY"
    assert action["automatic_runtime_enabled"] is False
    assert action["canonical_changed_fields"] == ["text"]
    assert action["shadow_authority_status"] == "AUTHORIZED_SHADOW"
    assert action["shadow_authority_reason_codes"] == []


def test_enabled_noneligible_alignment_is_noop():
    card = _card()
    draft = _draft(
        text=(
            "At higher overall metal–hydrogen coupling, a more balanced "
            "bonding–antibonding distribution promotes compatibility between "
            "hydrogen adsorption and H2 desorption."
        )
    )
    enabled = _decomposer(draft, enabled=True)

    result = enabled.decompose(card)

    assert result.claims[0].text == draft.text
    assert enabled.canonical_action_records == []


def test_enabled_path_does_not_swallow_unexpected_integrity_failure(monkeypatch):
    card = _card()
    draft = _draft()

    def corrupted(*args, **kwargs):
        raise ValueError("unexpected integrity failure")

    monkeypatch.setattr(
        decomposition_module,
        "_apply_existing_bridge_scope_alignment_canonical_action_opt_in",
        corrupted,
    )

    enabled = _decomposer(draft, enabled=True)

    with pytest.raises(ValueError, match="unexpected integrity failure"):
        enabled.decompose(card)


def test_action_records_reset_per_decompose_call():
    card = _card()
    draft = _draft()
    enabled = _decomposer(draft, enabled=True)

    first = enabled.decompose(card)
    first_records = copy.deepcopy(enabled.canonical_action_records)
    second = enabled.decompose(card)

    assert first.claims[0].text == SOURCE
    assert second.claims[0].text == SOURCE
    assert len(first_records) == 1
    assert len(enabled.canonical_action_records) == 1
    assert (
        enabled.canonical_action_records[0]["claim_id"]
        == first_records[0]["claim_id"]
    )
