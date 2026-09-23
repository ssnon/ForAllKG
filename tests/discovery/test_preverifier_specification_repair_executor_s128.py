from __future__ import annotations

from pipeline_core.discovery.preverifier_specification_repair_executor import (
    SpecificationRepairAuditDraft,
    SpecificationRepairDraft,
    semantic_audit_passes,
    validate_repair_draft,
)
from pipeline_core.discovery.preverifier_specification_repair_policy import (
    SpecificationClaimRepairPlan,
)


def _lexical_plan() -> SpecificationClaimRepairPlan:
    return SpecificationClaimRepairPlan(
        claim_id="claim:lex",
        repair_action="LEXICAL_ALIGNMENT_REPAIR",
        source_claim_snapshot_sha256="a" * 64,
        editable_fields=["required_bridge"],
        fill_only_fields=[],
        immutable_fields=[
            "claim_text",
            "predicted_observation",
            "falsification_condition",
        ],
        source_claim_text=(
            "For substrate composition, decreasing interparticle gap "
            "increases Raman intensity."
        ),
        source_required_bridge=(
            "For substrate composition, gap reduction increases Raman intensity."
        ),
        source_predicted_observation=(
            "For substrate composition, decreasing interparticle gap "
            "increases Raman intensity."
        ),
        source_falsification_condition=(
            "For substrate composition, decreasing interparticle gap "
            "does not increase Raman intensity."
        ),
        source_prior_art_identity_terms=["substrate composition"],
        source_relation_nucleus_terms=[
            "interparticle gap",
            "Raman intensity",
        ],
        source_reason_codes=[],
        source_abstention_reason="literal mismatch without paraphrase",
    )


def _completion_plan() -> SpecificationClaimRepairPlan:
    return SpecificationClaimRepairPlan(
        claim_id="claim:complete",
        repair_action="CONTRACT_COMPLETION_REPAIR",
        source_claim_snapshot_sha256="b" * 64,
        editable_fields=[
            "predicted_observation",
            "falsification_condition",
        ],
        fill_only_fields=[
            "predicted_observation",
            "falsification_condition",
        ],
        immutable_fields=["claim_text", "required_bridge"],
        source_claim_text=(
            "For gold substrate composition, increasing laser power "
            "decreases spectral stability."
        ),
        source_required_bridge=(
            "For gold substrate composition, increasing laser power "
            "decreases spectral stability."
        ),
        source_predicted_observation="",
        source_falsification_condition="",
        source_prior_art_identity_terms=["gold substrate composition"],
        source_relation_nucleus_terms=[
            "laser power",
            "spectral stability",
        ],
        source_reason_codes=[
            "missing_predicted_observation",
            "missing_falsification_condition",
        ],
    )


def test_lexical_repair_accepts_zero_delta_bridge_alignment() -> None:
    plan = _lexical_plan()
    draft = SpecificationRepairDraft(
        claim_id=plan.claim_id,
        required_bridge=(
            "For substrate composition, decreasing interparticle gap "
            "increases Raman intensity."
        ),
        predicted_observation=plan.source_predicted_observation,
        falsification_condition=plan.source_falsification_condition,
    )
    assert validate_repair_draft(plan=plan, draft=draft) == []


def test_lexical_repair_rejects_new_scientific_surface_token() -> None:
    plan = _lexical_plan()
    draft = SpecificationRepairDraft(
        claim_id=plan.claim_id,
        required_bridge=(
            "For substrate composition, decreasing interparticle gap "
            "increases Raman intensity through plasmon hybridization."
        ),
        predicted_observation=plan.source_predicted_observation,
        falsification_condition=plan.source_falsification_condition,
    )
    reasons = validate_repair_draft(plan=plan, draft=draft)
    assert any(
        reason.startswith(
            "new_surface_content_tokens:required_bridge:"
        )
        for reason in reasons
    )


def test_completion_repair_rejects_mutation_of_existing_bridge() -> None:
    plan = _completion_plan()
    draft = SpecificationRepairDraft(
        claim_id=plan.claim_id,
        required_bridge=(
            "For gold substrate composition, laser power changes stability."
        ),
        predicted_observation=(
            "For gold substrate composition, increasing laser power "
            "decreases spectral stability."
        ),
        falsification_condition=(
            "For gold substrate composition, increasing laser power "
            "does not decrease spectral stability."
        ),
    )
    reasons = validate_repair_draft(plan=plan, draft=draft)
    assert "immutable_field_changed:required_bridge" in reasons


def test_completion_repair_accepts_claim_supported_fill_only_fields() -> None:
    plan = _completion_plan()
    draft = SpecificationRepairDraft(
        claim_id=plan.claim_id,
        required_bridge=plan.source_required_bridge,
        predicted_observation=(
            "For gold substrate composition, increasing laser power "
            "decreases spectral stability."
        ),
        falsification_condition=(
            "For gold substrate composition, increasing laser power "
            "does not decrease spectral stability."
        ),
    )
    assert validate_repair_draft(plan=plan, draft=draft) == []


def test_semantic_audit_rejects_any_nonzero_delta_signal() -> None:
    audit = SpecificationRepairAuditDraft(
        claim_id="claim:1",
        zero_scientific_delta=False,
        added_scientific_concepts=["plasmon hybridization"],
        rationale="A mechanism was added.",
    )
    passed, reasons = semantic_audit_passes(
        audit,
        claim_id="claim:1",
    )
    assert passed is False
    assert "audit_nonzero_scientific_delta" in reasons
    assert "audit_added_scientific_concepts" in reasons


def test_semantic_audit_accepts_clean_zero_delta() -> None:
    audit = SpecificationRepairAuditDraft(
        claim_id="claim:1",
        zero_scientific_delta=True,
        rationale="Only literal wording was aligned.",
    )
    passed, reasons = semantic_audit_passes(
        audit,
        claim_id="claim:1",
    )
    assert passed is True
    assert reasons == []
