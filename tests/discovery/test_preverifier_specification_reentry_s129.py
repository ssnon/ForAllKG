from __future__ import annotations

from collections import Counter

from pipeline_core.discovery.preverifier_specification_reentry import (
    materialize_repaired_binding_plan,
)
from pipeline_core.discovery.preverifier_specification_repair_executor import (
    SpecificationRepairAuditDraft,
    SpecificationRepairCaseResult,
    SpecificationRepairClaimResult,
)
from pipeline_core.discovery.preverifier_specification_repair_policy import (
    SpecificationCaseRepairPlan,
    SpecificationClaimRepairPlan,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingClaimPlan,
    RelationalAtomicBindingHypothesisPlan,
    RelationalAtomicBindingPlan,
)


def _claim(
    *,
    claim_id: str,
    bridge: str,
    prediction: str,
    falsifier: str,
    ready: bool,
) -> RelationalAtomicBindingClaimPlan:
    return RelationalAtomicBindingClaimPlan(
        candidate_hypothesis_id="hypothesis:candidate",
        final_hypothesis_id="hypothesis:final",
        claim_id=claim_id,
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        claim_text=(
            "For gold substrate composition, increasing laser power "
            "decreases spectral stability."
        ),
        rationale="test",
        prior_art_identity_terms=["gold substrate composition"],
        relation_nucleus_terms=["laser power", "spectral stability"],
        required_bridge=bridge,
        predicted_observation=prediction,
        falsification_condition=falsifier,
        search_concepts=[],
        search_queries=[],
        source_claim_sha256="a" * 64,
        binding_status=(
            "READY_FOR_LITERAL_ENDPOINT_BINDING"
            if ready
            else "INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION"
        ),
        reason_codes=(
            []
            if ready
            else [
                "missing_predicted_observation",
                "missing_falsification_condition",
            ]
        ),
    )


def _source_plan() -> RelationalAtomicBindingPlan:
    claim = _claim(
        claim_id="claim:1",
        bridge=(
            "For gold substrate composition, increasing laser power "
            "decreases spectral stability."
        ),
        prediction="",
        falsifier="",
        ready=False,
    )
    hypothesis = RelationalAtomicBindingHypothesisPlan(
        original_hypothesis_id="hypothesis:original",
        candidate_hypothesis_id="hypothesis:candidate",
        final_hypothesis_id="hypothesis:final",
        alpha6_decision="accepted",
        certification_status="UNRESOLVED",
        n10_selection_class="CONDITIONAL",
        source_candidate_portfolio="/tmp/portfolio.json",
        source_candidate_portfolio_sha256="b" * 64,
        source_query_plan="/tmp/query.json",
        source_query_plan_sha256="c" * 64,
        claim_count=1,
        binding_ready_claim_count=0,
        novelty_bearing_binding_ready_claim_count=0,
        binding_status="NO_BINDABLE_CLAIMS",
        claims=[claim],
    )
    body = {
        "schema_version": "relational-atomic-binding-plan-v1",
        "run_dir": "/tmp/run",
        "source_alpha6_candidate_portfolio": "/tmp/final.json",
        "source_alpha6_candidate_portfolio_sha256": "d" * 64,
        "source_certification_report": "/tmp/cert.json",
        "source_certification_report_sha256": "e" * 64,
        "hypotheses": [hypothesis.model_dump(mode="json")],
        "hypothesis_count": 1,
        "ready_hypothesis_count": 0,
        "not_ready_hypothesis_count": 1,
        "claim_count": 1,
        "binding_ready_claim_count": 0,
        "novelty_bearing_binding_ready_claim_count": 0,
        "hypothesis_status_counts": {"NO_BINDABLE_CLAIMS": 1},
        "claim_status_counts": {
            "INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION": 1
        },
        "certification_only_source_required": True,
        "candidate_final_authority_equivalence_required": True,
        "endpoint_binding_performed": False,
        "verifier_result_observed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    import hashlib, json
    digest = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return RelationalAtomicBindingPlan(
        **body,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )


def _case_plan() -> SpecificationCaseRepairPlan:
    return SpecificationCaseRepairPlan(
        case_id="P09",
        source_case_result_id="case:result",
        source_case_result_sha256="f" * 64,
        source_primary_diagnostic_class="CONTRACT_COMPLETENESS_FAILURE",
        source_original_disposition="NO_BINDING_READY_HYPOTHESIS",
        status="PLANNED_AUTOMATIC_REPAIR",
        claim_repairs=[
            SpecificationClaimRepairPlan(
                claim_id="claim:1",
                repair_action="CONTRACT_COMPLETION_REPAIR",
                source_claim_snapshot_sha256="1" * 64,
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
                source_prior_art_identity_terms=[
                    "gold substrate composition"
                ],
                source_relation_nucleus_terms=[
                    "laser power",
                    "spectral stability",
                ],
                source_reason_codes=[
                    "missing_predicted_observation",
                    "missing_falsification_condition",
                ],
            )
        ],
        repair_output_dir="/tmp/r1/P09",
    )


def _case_result(
    *,
    status: str = "MATERIALIZED_R1",
) -> SpecificationRepairCaseResult:
    accepted = status == "MATERIALIZED_R1"
    row = SpecificationRepairClaimResult(
        claim_id="claim:1",
        source_claim_snapshot_sha256="1" * 64,
        repair_action="CONTRACT_COMPLETION_REPAIR",
        status=status,
        proposed_required_bridge=(
            "For gold substrate composition, increasing laser power "
            "decreases spectral stability."
        ),
        proposed_predicted_observation=(
            "For gold substrate composition, increasing laser power "
            "decreases spectral stability."
        ),
        proposed_falsification_condition=(
            "For gold substrate composition, increasing laser power "
            "does not decrease spectral stability."
        ),
        deterministic_policy_passed=accepted,
        deterministic_reason_codes=(
            [] if accepted else ["synthetic_reject"]
        ),
        semantic_audit_performed=accepted,
        semantic_audit_passed=(True if accepted else None),
        semantic_audit=(
            SpecificationRepairAuditDraft(
                claim_id="claim:1",
                zero_scientific_delta=True,
                rationale="same proposition",
            )
            if accepted
            else None
        ),
        generation_llm_calls=1,
        audit_llm_calls=(1 if accepted else 0),
        accepted_required_bridge=(
            "For gold substrate composition, increasing laser power "
            "decreases spectral stability."
            if accepted
            else None
        ),
        accepted_predicted_observation=(
            "For gold substrate composition, increasing laser power "
            "decreases spectral stability."
            if accepted
            else None
        ),
        accepted_falsification_condition=(
            "For gold substrate composition, increasing laser power "
            "does not decrease spectral stability."
            if accepted
            else None
        ),
    )
    counts = Counter([status])
    return SpecificationRepairCaseResult(
        case_id="P09",
        source_case_result_id="case:result",
        source_case_result_sha256="f" * 64,
        source_case_plan_status="PLANNED_AUTOMATIC_REPAIR",
        claim_results=[row],
        planned_claim_count=1,
        materialized_claim_count=(1 if accepted else 0),
        status_counts=dict(counts),
    )


def test_materialized_completion_can_recover_binding_readiness() -> None:
    repaired, lineage = materialize_repaired_binding_plan(
        source_plan=_source_plan(),
        case_plan=_case_plan(),
        case_result=_case_result(),
        repair_execution_report_id="repair:report",
    )
    assert repaired.ready_hypothesis_count == 1
    assert repaired.binding_ready_claim_count == 1
    assert repaired.novelty_bearing_binding_ready_claim_count == 1
    assert repaired.hypotheses[0].binding_status == (
        "READY_FOR_LITERAL_ENDPOINT_BINDING"
    )
    assert repaired.hypotheses[0].claims[0].reason_codes == []
    assert lineage.applied_repair_count == 1


def test_materialization_changes_claim_fingerprint() -> None:
    source = _source_plan()
    repaired, lineage = materialize_repaired_binding_plan(
        source_plan=source,
        case_plan=_case_plan(),
        case_result=_case_result(),
        repair_execution_report_id="repair:report",
    )
    assert (
        repaired.hypotheses[0].claims[0].source_claim_sha256
        != source.hypotheses[0].claims[0].source_claim_sha256
    )
    assert (
        lineage.claim_lineages[0].repaired_source_claim_sha256
        == repaired.hypotheses[0].claims[0].source_claim_sha256
    )


def test_materialization_preserves_original_plan_object() -> None:
    source = _source_plan()
    before = source.model_dump(mode="json")
    materialize_repaired_binding_plan(
        source_plan=source,
        case_plan=_case_plan(),
        case_result=_case_result(),
        repair_execution_report_id="repair:report",
    )
    assert source.model_dump(mode="json") == before


def test_materialization_rejects_when_no_claim_was_accepted() -> None:
    try:
        materialize_repaired_binding_plan(
            source_plan=_source_plan(),
            case_plan=_case_plan(),
            case_result=_case_result(
                status="REJECTED_DETERMINISTIC_POLICY"
            ),
            repair_execution_report_id="repair:report",
        )
    except ValueError as exc:
        assert "no MATERIALIZED_R1 claims" in str(exc)
    else:
        raise AssertionError("expected no-materialized repair rejection")


def test_repaired_falsifier_must_still_preserve_identity_literal() -> None:
    case_result = _case_result()
    row = case_result.claim_results[0]
    payload = row.model_dump(mode="json")
    payload["accepted_falsification_condition"] = (
        "Increasing laser power does not decrease spectral stability."
    )
    payload["proposed_falsification_condition"] = (
        payload["accepted_falsification_condition"]
    )
    mutated = SpecificationRepairClaimResult.model_validate(payload)
    case_result = SpecificationRepairCaseResult(
        case_id="P09",
        source_case_result_id="case:result",
        source_case_result_sha256="f" * 64,
        source_case_plan_status="PLANNED_AUTOMATIC_REPAIR",
        claim_results=[mutated],
        planned_claim_count=1,
        materialized_claim_count=1,
        status_counts={"MATERIALIZED_R1": 1},
    )

    repaired, _ = materialize_repaired_binding_plan(
        source_plan=_source_plan(),
        case_plan=_case_plan(),
        case_result=case_result,
        repair_execution_report_id="repair:report",
    )
    claim = repaired.hypotheses[0].claims[0]
    assert claim.binding_status == (
        "INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION"
    )
    assert any(
        reason.startswith(
            "identity_not_literal_in_falsification_condition"
        )
        for reason in claim.reason_codes
    )


def test_repaired_plan_hash_roundtrips() -> None:
    repaired, _ = materialize_repaired_binding_plan(
        source_plan=_source_plan(),
        case_plan=_case_plan(),
        case_result=_case_result(),
        repair_execution_report_id="repair:report",
    )
    roundtrip = RelationalAtomicBindingPlan.model_validate(
        repaired.model_dump(mode="json")
    )
    assert roundtrip.plan_id == repaired.plan_id
    assert roundtrip.plan_sha256 == repaired.plan_sha256
