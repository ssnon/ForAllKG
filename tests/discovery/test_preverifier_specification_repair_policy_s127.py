from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.preverifier_specification_diagnostics import (
    PreVerifierClaimDiagnostic,
    PreVerifierClaimSnapshot,
)
from pipeline_core.discovery.preverifier_specification_repair_policy import (
    SpecificationClaimRepairPlan,
    SpecificationRepairPolicyV1,
    _completion_claim_plan,
    _lexical_claim_plan,
)


def _claim(
    *,
    reason_codes: list[str],
    action: str,
    abstention_reason: str | None = None,
) -> PreVerifierClaimDiagnostic:
    return PreVerifierClaimDiagnostic(
        claim_id="claim:1",
        diagnostic_classes=[
            (
                "ENDPOINT_LITERAL_ALIGNMENT_FAILURE"
                if action == "LEXICAL_ALIGNMENT_REPAIR"
                else "CONTRACT_COMPLETENESS_FAILURE"
            )
        ],
        proposed_repair_actions=[action],
        scientific_delta_policy="ZERO_SCIENTIFIC_DELTA_REQUIRED",
        abstention_reason=abstention_reason,
        source_claim=PreVerifierClaimSnapshot(
            claim_id="claim:1",
            novelty_selection_role="NOVELTY_BEARING",
            binding_status=(
                "READY_FOR_LITERAL_ENDPOINT_BINDING"
                if action == "LEXICAL_ALIGNMENT_REPAIR"
                else "INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION"
            ),
            reason_codes=reason_codes,
            claim_text="Particle spacing changes Raman intensity.",
            required_bridge=(
                "Spacing reduction changes Raman intensity."
                if action == "LEXICAL_ALIGNMENT_REPAIR"
                else ""
            ),
            predicted_observation=(
                "Particle spacing changes Raman intensity."
                if "missing_predicted_observation" not in reason_codes
                else ""
            ),
            falsification_condition=(
                "Particle spacing does not change Raman intensity."
                if "missing_falsification_condition" not in reason_codes
                else ""
            ),
            prior_art_identity_terms=["substrate composition"],
            relation_nucleus_terms=["particle spacing", "Raman intensity"],
        ),
    )


def test_policy_v1_forbids_scientific_content_changes() -> None:
    policy = SpecificationRepairPolicyV1()
    assert policy.claim_text_mutation_allowed is False
    assert policy.new_scientific_concepts_allowed is False
    assert policy.new_mechanism_allowed is False
    assert policy.new_scope_allowed is False
    assert policy.relation_direction_change_allowed is False
    assert policy.literature_retrieval_during_repair_allowed is False


def test_lexical_alignment_may_edit_required_bridge_only() -> None:
    plan = _lexical_claim_plan(
        _claim(
            reason_codes=[],
            action="LEXICAL_ALIGNMENT_REPAIR",
            abstention_reason="literal wording mismatch without paraphrase",
        )
    )
    assert plan.editable_fields == ["required_bridge"]
    assert plan.fill_only_fields == []
    assert "claim_text" in plan.immutable_fields
    assert "predicted_observation" in plan.immutable_fields
    assert "falsification_condition" in plan.immutable_fields


def test_contract_completion_edits_only_missing_fields() -> None:
    plan = _completion_claim_plan(
        _claim(
            reason_codes=[
                "missing_predicted_observation",
                "missing_falsification_condition",
            ],
            action="CONTRACT_COMPLETION_REPAIR",
        )
    )
    assert plan.editable_fields == [
        "predicted_observation",
        "falsification_condition",
    ]
    assert plan.fill_only_fields == plan.editable_fields
    assert "required_bridge" in plan.immutable_fields
    assert "claim_text" in plan.immutable_fields


def test_contract_completion_rejects_missing_claim_text() -> None:
    with pytest.raises(ValueError, match="cannot automatically repair"):
        _completion_claim_plan(
            _claim(
                reason_codes=["missing_claim_text"],
                action="CONTRACT_COMPLETION_REPAIR",
            )
        )


def test_lexical_plan_schema_rejects_broader_edit_surface() -> None:
    source = _lexical_claim_plan(
        _claim(
            reason_codes=[],
            action="LEXICAL_ALIGNMENT_REPAIR",
            abstention_reason="literal mismatch",
        )
    ).model_dump(mode="json")
    source["editable_fields"] = [
        "required_bridge",
        "predicted_observation",
    ]
    with pytest.raises(
        ValidationError,
        match="required_bridge only",
    ):
        SpecificationClaimRepairPlan.model_validate(source)


def test_contract_completion_schema_requires_fill_only_fields() -> None:
    source = _completion_claim_plan(
        _claim(
            reason_codes=["missing_required_bridge"],
            action="CONTRACT_COMPLETION_REPAIR",
        )
    ).model_dump(mode="json")
    source["fill_only_fields"] = []
    with pytest.raises(
        ValidationError,
        match="originally empty fields",
    ):
        SpecificationClaimRepairPlan.model_validate(source)
