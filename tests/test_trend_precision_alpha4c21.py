from __future__ import annotations

from dac_her.trend_precision import (
    PaperLocalTrendResult,
    TrendEvidenceAnnotation,
    TrendPrecisionAdapter,
    audit_trend_precision,
)


def test_precision_contract_requires_exact_evidence_membership():
    annotation = TrendEvidenceAnnotation(
        trend_id="t1", paper_id="P", precision_semantics_id="p1",
        evidence_kind="reported_claim", classification_basis="claim_contract",
        control_family="structural", observable_semantics="measured_signal_intensity",
    )
    result = PaperLocalTrendResult(
        result_id="r1", paper_id="P", domain_profile_id="d",
        trend_semantics_id="tsem", precision_semantics_id="p1",
        result_lane="claim", independent_variable_key="x",
        dependent_observable_key="y", direction="positive", shape="monotonic",
        control_family="structural", observable_semantics="measured_signal_intensity",
        member_trend_ids=("t1",), evidence_kinds=("reported_claim",),
    )
    adapter = TrendPrecisionAdapter(
        adapter_id="d", domain_profile_id="d", trend_semantics_id="tsem",
        precision_semantics_id="p1", annotate_fn=lambda row, graph: annotation,
        consolidate_fn=lambda rows, anns, graphs: [result],
    )
    audit = audit_trend_precision(
        evidence_rows=[{
            "trend_id":"t1", "paper_id":"P",
            "evidence_basis":"reported_directional_claim",
            "independent_variable_key":"x", "dependent_observable_key":"y",
        }],
        annotations=[annotation], results=[result], adapter=adapter,
    )
    assert audit.structural_gate is True
    assert audit.local_result_count == 1
