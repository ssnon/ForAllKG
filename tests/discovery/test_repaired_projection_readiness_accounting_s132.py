from __future__ import annotations

from pipeline_core.discovery.relational_atomic_projection import (
    RelationalAtomicProjectionReport,
    RelationalAtomicProjectionRow,
)
from pipeline_core.discovery.repaired_projection_readiness_accounting import (
    classify_projection_report,
)


def _report(
    rows: list[RelationalAtomicProjectionRow],
) -> RelationalAtomicProjectionReport:
    body = {
        "schema_version": "relational-atomic-projection-report-v1",
        "source_binding_plan_id": "plan:1",
        "source_binding_plan_sha256": "a" * 64,
        "source_endpoint_binding_report_id": "endpoint:1",
        "rows": [row.model_dump(mode="json") for row in rows],
        "selected_claim_count": len(rows),
        "projected_claim_count": sum(
            row.projection_status == "PROJECTED" for row in rows
        ),
        "skipped_endpoint_abstention_count": sum(
            row.projection_status == "SKIPPED_ENDPOINT_ABSTENTION"
            for row in rows
        ),
        "source_binding_abstention_count": sum(
            row.projection_status == "ABSTAINED_SOURCE_BINDING"
            for row in rows
        ),
        "source_population_frozen_before_endpoint_binding": True,
        "candidate_final_authority_equivalence_required": True,
        "exact_prediction_source_binding_required": True,
        "exact_falsifier_source_binding_required": True,
        "shared_observable_identity_required": True,
        "scientific_content_added": False,
        "novelty_assessment_performed": False,
        "verifier_result_observed": False,
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
    return RelationalAtomicProjectionReport(
        **body,
        report_id="relational_atomic_projection:" + digest[:20],
        report_sha256=digest,
    )


def _row(status: str, reasons: list[str]):
    return RelationalAtomicProjectionRow(
        candidate_hypothesis_id="candidate:1",
        final_hypothesis_id="final:1",
        claim_id="claim:1",
        source_claim_sha256="b" * 64,
        endpoint_binding_outcome=(
            "ABSTAINED_UNBINDABLE"
            if status == "SKIPPED_ENDPOINT_ABSTENTION"
            else "BOUND_LITERAL_ENDPOINTS"
        ),
        projection_status=status,
        reason_codes=reasons,
    )


def test_source_binding_abstention_is_not_full_verifier_ready() -> None:
    report = _report(
        [
            _row(
                "ABSTAINED_SOURCE_BINDING",
                [
                    "prediction_exact_source_binding_cardinality:0",
                    "falsifier_exact_source_binding_cardinality:0",
                ],
            )
        ]
    )
    disposition, reasons = classify_projection_report(report)
    assert disposition == "SOURCE_BINDING_ABSTAINED_AFTER_R1"
    assert "prediction_exact_source_binding_cardinality:0" in reasons


def test_endpoint_abstention_is_distinct_from_source_binding() -> None:
    report = _report(
        [
            _row(
                "SKIPPED_ENDPOINT_ABSTENTION",
                ["literal_endpoint_binding_abstained"],
            )
        ]
    )
    disposition, _ = classify_projection_report(report)
    assert disposition == "ENDPOINT_BINDING_ABSTAINED_AFTER_R1"


def test_any_projected_claim_is_full_verifier_ready() -> None:
    # A projected row needs a specification under the production model.
    # This focused classifier test uses model_construct to isolate the
    # readiness accounting rule from specification construction.
    row = RelationalAtomicProjectionRow.model_construct(
        candidate_hypothesis_id="candidate:1",
        final_hypothesis_id="final:1",
        claim_id="claim:1",
        source_claim_sha256="b" * 64,
        endpoint_binding_outcome="BOUND_LITERAL_ENDPOINTS",
        projection_status="PROJECTED",
        reason_codes=[],
        specification=object(),
        candidate_final_authority_equivalence_inherited=True,
        source_claim_content_preserved=True,
        relation_endpoints_from_literal_binding_only=True,
        observable_from_exact_prediction_falsifier_source_binding_only=True,
        scientific_content_added=False,
        novelty_authority=False,
        production_selection_authority=False,
    )
    report = RelationalAtomicProjectionReport.model_construct(
        schema_version="relational-atomic-projection-report-v1",
        report_id="projection:test",
        report_sha256="a" * 64,
        source_binding_plan_id="plan:1",
        source_binding_plan_sha256="b" * 64,
        source_endpoint_binding_report_id="endpoint:1",
        rows=[row],
        selected_claim_count=1,
        projected_claim_count=1,
        skipped_endpoint_abstention_count=0,
        source_binding_abstention_count=0,
        source_population_frozen_before_endpoint_binding=True,
        candidate_final_authority_equivalence_required=True,
        exact_prediction_source_binding_required=True,
        exact_falsifier_source_binding_required=True,
        shared_observable_identity_required=True,
        scientific_content_added=False,
        novelty_assessment_performed=False,
        verifier_result_observed=False,
        production_selection_changed=False,
        canonical_graph_mutated=False,
    )
    disposition, reasons = classify_projection_report(report)
    assert disposition == "FULL_VERIFIER_READY_AFTER_R1"
    assert reasons == []


def test_source_binding_reasons_are_deduplicated_in_order() -> None:
    report = _report(
        [
            _row(
                "ABSTAINED_SOURCE_BINDING",
                [
                    "unsupported_atomic_claim_kind:composite",
                    "prediction_exact_source_binding_cardinality:0",
                ],
            ),
            RelationalAtomicProjectionRow(
                candidate_hypothesis_id="candidate:1",
                final_hypothesis_id="final:1",
                claim_id="claim:2",
                source_claim_sha256="c" * 64,
                endpoint_binding_outcome="BOUND_LITERAL_ENDPOINTS",
                projection_status="ABSTAINED_SOURCE_BINDING",
                reason_codes=[
                    "prediction_exact_source_binding_cardinality:0",
                    "falsifier_exact_source_binding_cardinality:0",
                ],
            ),
        ]
    )
    _, reasons = classify_projection_report(report)
    assert reasons == [
        "unsupported_atomic_claim_kind:composite",
        "prediction_exact_source_binding_cardinality:0",
        "falsifier_exact_source_binding_cardinality:0",
    ]


def test_empty_projection_accounting_is_rejected() -> None:
    report = _report([])
    try:
        classify_projection_report(report)
    except ValueError as exc:
        assert "no projected or abstained claim accounting" in str(exc)
    else:
        raise AssertionError("expected empty projection rejection")
