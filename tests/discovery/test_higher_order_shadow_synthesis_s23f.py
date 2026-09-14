from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.higher_order_shadow_synthesis import (
    HigherOrderShadowHypothesisDraft,
    audit_higher_order_shadow_hypothesis,
    select_shadow_synthesis_contexts,
)


def _safe_draft() -> HigherOrderShadowHypothesisDraft:
    return HigherOrderShadowHypothesisDraft(
        decision="propose",
        hypothesis_statement=(
            "The relationship between interparticle separation and "
            "SERS intensity may depend on surface roughness."
        ),
        mechanistic_bridge=(
            "The recorded separation-to-field, field-to-SERS, and "
            "roughness-to-field relations motivate a test of context "
            "dependence while keeping their mediator expressions distinct."
        ),
        predicted_observation=(
            "Across controlled surface-roughness conditions, test whether "
            "the separation-SERS relationship differs in form or magnitude."
        ),
        falsification_condition=(
            "The hypothesis is weakened if the separation-SERS relationship "
            "is indistinguishable across the compared roughness conditions."
        ),
        assumptions=[
            "The recorded relations remain applicable in the compared system."
        ],
    )


def test_proposal_contract_requires_complete_shadow_hypothesis():
    with pytest.raises(ValidationError):
        HigherOrderShadowHypothesisDraft(
            decision="propose",
            hypothesis_statement="Maybe context dependent.",
        )


def test_abstention_contract_forbids_proposal_fields():
    with pytest.raises(ValidationError):
        HigherOrderShadowHypothesisDraft(
            decision="abstain",
            hypothesis_statement="Should not exist",
            abstention_reason="Insufficient coherence.",
        )


def test_direction_field_is_fail_closed_to_unspecified():
    payload = _safe_draft().model_dump(mode="json")
    payload["expected_direction"] = "increase"

    with pytest.raises(ValidationError):
        HigherOrderShadowHypothesisDraft.model_validate(
            payload
        )


def test_authority_audit_passes_bounded_open_direction_hypothesis():
    audit = audit_higher_order_shadow_hypothesis(
        _safe_draft()
    )

    assert audit.authority_safe is True
    assert audit.novelty_language_hits == []
    assert audit.evidence_upgrade_hits == []
    assert audit.directional_interaction_hits == []
    assert audit.mediator_identity_language_hits == []


def test_authority_audit_flags_novelty_evidence_and_direction_language():
    draft = HigherOrderShadowHypothesisDraft(
        decision="propose",
        hypothesis_statement=(
            "This novel interaction demonstrates that the modifier "
            "increases SERS response."
        ),
        mechanistic_bridge=(
            "The topology confirms the interaction."
        ),
        predicted_observation=(
            "A higher response is expected."
        ),
        falsification_condition=(
            "No difference would falsify it."
        ),
    )

    audit = audit_higher_order_shadow_hypothesis(
        draft
    )

    assert audit.authority_safe is False
    assert audit.novelty_language_hits
    assert audit.evidence_upgrade_hits
    assert audit.directional_interaction_hits


def test_authority_audit_ignores_explicitly_negated_guard_language():
    draft = HigherOrderShadowHypothesisDraft(
        decision="propose",
        hypothesis_statement=(
            "The relationship may differ under the modifier."
        ),
        mechanistic_bridge=(
            "The three mediator expressions remain distinct. "
            "No identity, causal link, or mediator path among these "
            "expressions is assumed, without establishing a shared "
            "mediator."
        ),
        predicted_observation=(
            "Test whether the source-target relationship differs."
        ),
        falsification_condition=(
            "No difference across modifier states would falsify it."
        ),
    )

    audit = audit_higher_order_shadow_hypothesis(draft)

    assert audit.authority_safe is True
    assert audit.mediator_identity_language_hits == []
    assert audit.evidence_upgrade_hits == []


def test_authority_audit_ignores_rather_than_exclusion_scope():
    draft = HigherOrderShadowHypothesisDraft(
        decision="propose",
        hypothesis_statement=(
            "The relationship may differ under the modifier."
        ),
        mechanistic_bridge=(
            "The mediator expressions are treated as separate compatibility "
            "expressions rather than as identical entities or as a supported "
            "mediator path."
        ),
        predicted_observation=(
            "Test whether the source-target relationship differs."
        ),
        falsification_condition=(
            "No difference across modifier states would falsify it."
        ),
    )

    audit = audit_higher_order_shadow_hypothesis(draft)

    assert audit.authority_safe is True
    assert audit.mediator_identity_language_hits == []


def test_authority_audit_rather_than_scope_stops_at_comma():
    draft = HigherOrderShadowHypothesisDraft(
        decision="propose",
        hypothesis_statement=(
            "The relationship may differ under the modifier."
        ),
        mechanistic_bridge=(
            "Rather than treating the expressions as identical, "
            "the topology establishes a mediator path."
        ),
        predicted_observation=(
            "Test whether the source-target relationship differs."
        ),
        falsification_condition=(
            "No difference across modifier states would falsify it."
        ),
    )

    audit = audit_higher_order_shadow_hypothesis(draft)

    assert audit.authority_safe is False
    assert audit.evidence_upgrade_hits
    assert audit.mediator_identity_language_hits


def test_authority_audit_negation_scope_resets_after_adversative():
    draft = HigherOrderShadowHypothesisDraft(
        decision="propose",
        hypothesis_statement=(
            "The relationship may differ under the modifier."
        ),
        mechanistic_bridge=(
            "No mediator identity is assumed, but the topology establishes "
            "a mediator path."
        ),
        predicted_observation=(
            "Test whether the source-target relationship differs."
        ),
        falsification_condition=(
            "No difference across modifier states would falsify it."
        ),
    )

    audit = audit_higher_order_shadow_hypothesis(draft)

    assert audit.authority_safe is False
    assert audit.evidence_upgrade_hits
    assert audit.mediator_identity_language_hits


def test_authority_audit_flags_unsupported_mediator_identity_language():
    draft = HigherOrderShadowHypothesisDraft(
        decision="propose",
        hypothesis_statement=(
            "The relationship may depend on surface roughness."
        ),
        mechanistic_bridge=(
            "The three relations form a possible mediator path through "
            "their shared electric-field mediator."
        ),
        predicted_observation=(
            "Test whether the source-target relationship differs."
        ),
        falsification_condition=(
            "No difference across modifier states would falsify it."
        ),
    )

    audit = audit_higher_order_shadow_hypothesis(draft)

    assert audit.authority_safe is False
    assert audit.mediator_identity_language_hits


def _context(
    context_id: str,
    modifier_id: str,
):
    return SimpleNamespace(
        context_id=context_id,
        lineage=SimpleNamespace(
            modifier_component_id=modifier_id
        ),
    )


def test_selection_keeps_one_context_per_modifier_without_quality_ranking():
    contexts = [
        _context("context:z", "modifier:a"),
        _context("context:a", "modifier:a"),
        _context("context:b", "modifier:b"),
        _context("context:c", "modifier:c"),
    ]

    selected = select_shadow_synthesis_contexts(
        contexts=contexts,
        max_contexts=3,
    )

    modifier_ids = [
        row.lineage.modifier_component_id
        for row in selected
    ]

    assert len(selected) == 3
    assert len(set(modifier_ids)) == 3

    # For modifier:a, context:a must win over context:z before hash sampling.
    selected_for_a = [
        row.context_id
        for row in selected
        if (
            row.lineage.modifier_component_id
            == "modifier:a"
        )
    ]
    assert selected_for_a == ["context:a"]


def test_selection_rejects_zero_sample_size():
    with pytest.raises(
        ValueError,
        match="max_contexts",
    ):
        select_shadow_synthesis_contexts(
            contexts=[],
            max_contexts=0,
        )
