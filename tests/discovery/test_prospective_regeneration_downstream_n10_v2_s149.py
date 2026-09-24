from pipeline_core.discovery.prospective_regeneration_downstream_n10_v2 import (
    certification_status_from_production_gate,
    unresolved_dimensions_from_gate_row,
)


def _gate(selection_class: str, positive: bool, allowed: bool):
    return {
        "schema_version": "scientific-novelty-fallback-gate-v2",
        "production_authority": True,
        "authority_scope": "alpha6_post_generation_candidate",
        "authority_source": "n10_role_aware_nonobviousness_v2",
        "positive_authority_requires": "ELIGIBLE_AND_ROLE_AWARE_POSITIVE_NONOBVIOUSNESS",
        "conditional_is_positive": False,
        "absence_is_novelty": False,
        "candidate_semantics_preserved": True,
        "gates": [
            {
                "hypothesis_id": "hypothesis:regen",
                "selection_class": selection_class,
                "fallback_allowed": allowed,
                "positive_nonobviousness_authority": positive,
                "action": "KEEP" if allowed else "RESOLVE",
                "blocking_claim_ids": [],
                "unresolved_claim_ids": [],
                "resolution_requirements": [],
                "reason_codes": [],
            }
        ],
    }


def test_regeneration_certification_uses_frozen_class_mapping():
    status, _ = certification_status_from_production_gate(
        regenerated_hypothesis_id="hypothesis:regen",
        production_gate=_gate("ELIGIBLE", True, True),
    )
    assert status == "NOVELTY_CERTIFIED"

    status, _ = certification_status_from_production_gate(
        regenerated_hypothesis_id="hypothesis:regen",
        production_gate=_gate("CONDITIONAL", False, False),
    )
    assert status == "NOVELTY_UNRESOLVED"

    status, _ = certification_status_from_production_gate(
        regenerated_hypothesis_id="hypothesis:regen",
        production_gate=_gate("INELIGIBLE", False, False),
    )
    assert status == "NOVELTY_REJECTED"


def test_unresolved_dimension_mapping_matches_alpha6_helper_semantics():
    row = {
        "resolution_requirements": [
            {
                "novelty_selection_role": "NOVELTY_BEARING",
                "action": "RESOLVE_NOVELTY_BEARING_EVIDENCE",
                "nonobviousness_outcome": "INSUFFICIENT_FOR_JUDGMENT",
                "reason_codes": ["candidate_not_ready_for_adjudication"],
            },
            {
                "novelty_selection_role": "REQUIRED_ENABLING_RELATION",
                "action": "RESOLVE",
                "nonobviousness_outcome": "CONDITIONAL",
                "reason_codes": [],
            },
        ]
    }
    assert unresolved_dimensions_from_gate_row(row) == [
        "EVIDENCE_CLOSURE",
        "REQUIRED_ENABLING_RELATION",
    ]
