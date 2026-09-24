import hashlib
import json

import pytest

from pipeline_core.discovery.prospective_regeneration_binding_gate_v2 import (
    RegenerationReachabilityComparisonV2,
)


def _sha(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_semantic_terminal_remains_in_regeneration_denominator():
    body = {
        "schema_version":
            "regeneration-preverifier-reachability-comparison-v2",
        "case_id": "P18",
        "source_semantic_report_id": "semantic:1",
        "source_semantic_report_sha256": "a" * 64,
        "source_external_n10_report_id": "n10:1",
        "source_binding_plan_id": "plan:1",
        "source_gate_report_id": "gate:1",
        "lineage_count": 2,
        "initial_gate_ready_hypothesis_count": 0,
        "primary_gate_ready_hypothesis_count": 0,
        "regenerated_gate_ready_hypothesis_count": 1,
        "gate_evaluated_lineage_count": 1,
        "semantic_terminal_lineage_count": 1,
        "n10_certified_count": 0,
        "n10_unresolved_count": 1,
        "n10_rejected_count": 0,
        "n10_not_evaluated_due_semantic_terminal_count": 1,
        "lineages": [
            {
                "source_final_hypothesis_id": "hypothesis:old-a",
                "regenerated_hypothesis_id": "hypothesis:new-a",
                "semantic_status": "HARD_GATE_FAILED",
                "external_n10_evaluated": False,
                "gate_evaluated": False,
                "n10_certification_status":
                    "NOT_EVALUATED_SEMANTIC_TERMINAL",
                "n10_selection_class": "NOT_EVALUATED",
                "gate_ready": False,
                "novelty_bearing_gate_ready_claim_ids": [],
                "terminal_reason": "HARD_GATE_FAILED",
            },
            {
                "source_final_hypothesis_id": "hypothesis:old-b",
                "regenerated_hypothesis_id": "hypothesis:new-b",
                "semantic_status": "SEMANTIC_ACCEPTED",
                "external_n10_evaluated": True,
                "gate_evaluated": True,
                "n10_certification_status": "NOVELTY_UNRESOLVED",
                "n10_selection_class": "CONDITIONAL",
                "gate_ready": True,
                "novelty_bearing_gate_ready_claim_ids": ["claim:1"],
                "terminal_reason": None,
            },
        ],
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "second_regeneration_performed": False,
        "scientific_content_mutated": False,
        "production_selection_changed": False,
    }
    digest = _sha(body)
    report = RegenerationReachabilityComparisonV2(
        **body,
        report_id=(
            "regeneration_preverifier_reachability_comparison_v2:"
            + digest[:20]
        ),
        report_sha256=digest,
    )
    assert report.lineage_count == 2
    assert report.gate_evaluated_lineage_count == 1
    assert report.semantic_terminal_lineage_count == 1
    assert report.regenerated_gate_ready_hypothesis_count == 1
    assert report.n10_not_evaluated_due_semantic_terminal_count == 1


def test_semantic_terminal_cannot_be_gate_ready():
    with pytest.raises(ValueError, match="non-evaluated lineage"):
        body = {
            "schema_version":
                "regeneration-preverifier-reachability-comparison-v2",
            "case_id": "P18",
            "source_semantic_report_id": "semantic:1",
            "source_semantic_report_sha256": "a" * 64,
            "source_external_n10_report_id": "n10:1",
            "source_binding_plan_id": "plan:1",
            "source_gate_report_id": "gate:1",
            "lineage_count": 1,
            "initial_gate_ready_hypothesis_count": 0,
            "primary_gate_ready_hypothesis_count": 0,
            "regenerated_gate_ready_hypothesis_count": 1,
            "gate_evaluated_lineage_count": 0,
            "semantic_terminal_lineage_count": 1,
            "n10_certified_count": 0,
            "n10_unresolved_count": 0,
            "n10_rejected_count": 0,
            "n10_not_evaluated_due_semantic_terminal_count": 1,
            "lineages": [{
                "source_final_hypothesis_id": "old",
                "regenerated_hypothesis_id": "new",
                "semantic_status": "HARD_GATE_FAILED",
                "external_n10_evaluated": False,
                "gate_evaluated": False,
                "n10_certification_status":
                    "NOT_EVALUATED_SEMANTIC_TERMINAL",
                "n10_selection_class": "NOT_EVALUATED",
                "gate_ready": True,
                "novelty_bearing_gate_ready_claim_ids": [],
                "terminal_reason": "HARD_GATE_FAILED",
            }],
            "endpoint_binding_performed": False,
            "verifier_performed": False,
            "second_regeneration_performed": False,
            "scientific_content_mutated": False,
            "production_selection_changed": False,
        }
        digest = _sha(body)
        RegenerationReachabilityComparisonV2(
            **body,
            report_id=(
                "regeneration_preverifier_reachability_comparison_v2:"
                + digest[:20]
            ),
            report_sha256=digest,
        )
