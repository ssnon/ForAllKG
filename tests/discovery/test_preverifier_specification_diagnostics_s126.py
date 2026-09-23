from __future__ import annotations

from pipeline_core.discovery.preverifier_specification_diagnostics import (
    _classify_main_e2e_failure,
    _primary_class,
    classify_binding_reason,
    classify_endpoint_abstention,
    repair_action_for_class,
)


def test_endpoint_paraphrase_abstention_is_literal_alignment_failure() -> None:
    reason = (
        'The gap-related endpoint is phrased as "decreasing interparticle gap" '
        'while the bridge uses "gap reduction"; two distinct literal endpoints '
        'cannot be selected without paraphrase.'
    )
    assert classify_endpoint_abstention(reason) == "ENDPOINT_LITERAL_ALIGNMENT_FAILURE"
    action, policy, eligible = repair_action_for_class("ENDPOINT_LITERAL_ALIGNMENT_FAILURE")
    assert action == "LEXICAL_ALIGNMENT_REPAIR"
    assert policy == "ZERO_SCIENTIFIC_DELTA_REQUIRED"
    assert eligible is True


def test_endpoint_inference_abstention_requires_escalation() -> None:
    reason = "A second endpoint is only implied and would require inference not explicitly stated in both frozen fields."
    assert classify_endpoint_abstention(reason) == "ENDPOINT_UNGROUNDED_OR_INFERENTIAL_FAILURE"
    action, policy, eligible = repair_action_for_class("ENDPOINT_UNGROUNDED_OR_INFERENTIAL_FAILURE")
    assert action == "MANUAL_DIAGNOSIS_REQUIRED"
    assert policy == "ESCALATE_BEFORE_REPAIR"
    assert eligible is False


def test_binding_reason_taxonomy_separates_contract_failures() -> None:
    assert classify_binding_reason("missing_required_bridge") == "CONTRACT_COMPLETENESS_FAILURE"
    assert classify_binding_reason("identity_not_literal_in_required_bridge:0") == "CONTRACT_LITERAL_IDENTITY_ALIGNMENT_FAILURE"
    assert classify_binding_reason("missing_novelty_selection_role") == "CONTRACT_ROLE_OR_IDENTITY_METADATA_FAILURE"


def test_main_e2e_network_failure_is_not_scientific_repair() -> None:
    manifest = {
        "status": "failed",
        "failure": {"type": "RateLimitError", "message": "HTTP 429 rate limit exceeded", "traceback": "provider request failed"},
        "stages": [{"name": "[10/13] External novelty alpha5.2", "status": "failed"}],
    }
    diagnostic, stage, failure_type, message = _classify_main_e2e_failure(manifest)
    assert diagnostic == "MAIN_E2E_INFRASTRUCTURE_FAILURE"
    assert stage == "[10/13] External novelty alpha5.2"
    assert failure_type == "RateLimitError"
    assert "429" in str(message)


def test_main_e2e_nonnetwork_stage_failure_requires_manual_diagnosis() -> None:
    manifest = {
        "status": "failed",
        "failure": {"type": "CalledProcessError", "message": "stage command exited with status 1"},
        "stages": [{"name": "[8/13] Discovery-axis hypothesis synthesis", "status": "failed"}],
    }
    diagnostic, stage, _, _ = _classify_main_e2e_failure(manifest)
    assert diagnostic == "MAIN_E2E_UPSTREAM_STAGE_FAILURE"
    assert stage == "[8/13] Discovery-axis hypothesis synthesis"


def test_primary_diagnostic_prefers_escalation_over_lexical_repair() -> None:
    assert _primary_class([
        "ENDPOINT_LITERAL_ALIGNMENT_FAILURE",
        "ENDPOINT_UNGROUNDED_OR_INFERENTIAL_FAILURE",
    ]) == "ENDPOINT_UNGROUNDED_OR_INFERENTIAL_FAILURE"
