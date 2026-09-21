from __future__ import annotations

from pipeline_core.discovery.reframing.evidence_tension import (
    extract_evidence_level_tensions,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ConditionEvidenceExample,
    ReframeEvidenceStatement,
    ScientificReframeEvidencePacket,
)
from pipeline_core.discovery.reframing.trigger_detection import (
    detect_scientific_reframe_triggers,
)


def _evidence(
    *,
    text_a: str = "Under low loading the response increases.",
    text_b: str = "Under high loading no significant effect is detected.",
    claim_a: str = "observation",
    claim_b: str = "observation",
    papers=("P1", "P2"),
) -> ScientificReframeEvidencePacket:
    return ScientificReframeEvidencePacket(
        task_id="task:1",
        question="When does the response law change?",
        source_context_id="context:1",
        source_context_sha256="context-sha",
        premise_statements=[
            ReframeEvidenceStatement(
                statement_id="s1",
                text=text_a,
                epistemic_role="reported",
                claim_kind=claim_a,
                paper_ids=[papers[0]],
                scientific_support_node_ids=["n1"],
            ),
            ReframeEvidenceStatement(
                statement_id="s2",
                text=text_b,
                epistemic_role="reported",
                claim_kind=claim_b,
                paper_ids=[papers[1]],
                scientific_support_node_ids=["n2"],
            ),
        ],
        gap_statements=[],
        condition_examples=[
            ConditionEvidenceExample(
                object_kind="measurement",
                paper_id=papers[0],
                chunk_id="c1",
                object_id="m1",
                label="response",
                conditions=[{"name": "loading", "value_numeric": 1}],
                direct_grounded_object=True,
            ),
            ConditionEvidenceExample(
                object_kind="measurement",
                paper_id=papers[1],
                chunk_id="c2",
                object_id="m2",
                label="response",
                conditions=[{"name": "loading", "value_numeric": 10}],
                direct_grounded_object=True,
            ),
        ],
        grounded_source_chunk_count=2,
        unresolved_grounded_node_count=0,
    )


def _report(tension_type="context_dependency"):
    return {
        "schema_version": "exploration-report-v1",
        "report_id": "report:1",
        "task_id": "task:1",
        "evidence_tensions": [
            {
                "tension_id": "t1",
                "statement_id": "s1",
                "side_a_statement_ids": ["s1"],
                "side_b_statement_ids": ["s2"],
                "tension_type": tension_type,
                "paper_ids": ["P1", "P2"],
            }
        ],
    }


def _packet(type_a="Measurement", type_b="Measurement"):
    return {
        "task": {"task_id": "task:1"},
        "evidence_catalog": {
            "nodes": {
                "n1": {"node_type": type_a},
                "n2": {"node_type": type_b},
            }
        },
    }


def _assessment(result, operator_id):
    return next(row for row in result.assessments if row.operator_id == operator_id)


def test_extracts_null_context_and_measurement_without_lexical_pseudo_boundary():
    result = extract_evidence_level_tensions(
        explorer_report=_report(),
        evidence=_evidence(),
        explorer_packet=_packet(),
    )
    witness = result.witnesses[0]
    assert result.extraction_mode == "packet_assisted"
    assert set(witness.tension_types) >= {
        "CONTEXT",
        "NULL_EFFECT",
        "MEASUREMENT",
    }
    assert "BOUNDARY" not in witness.tension_types
    assert witness.paired_grounded_sides is True
    assert witness.paired_response_signal is True
    assert witness.relevant_condition_signature_count == 2
    assert result.scientific_conflict_authority is False
    assert result.regime_boundary_authority is False


def test_opposite_directional_language_is_detected():
    result = extract_evidence_level_tensions(
        explorer_report=_report("qualitative_difference"),
        evidence=_evidence(
            text_a="The response increases with loading.",
            text_b="The response decreases with loading.",
        ),
        explorer_packet=_packet(),
    )
    assert "DIRECTIONAL" in result.witnesses[0].tension_types


def test_quantitative_difference_maps_to_magnitude():
    result = extract_evidence_level_tensions(
        explorer_report=_report("quantitative_difference"),
        evidence=_evidence(
            text_a="The response was 2 fold larger.",
            text_b="The response was 1.2 fold larger.",
        ),
        explorer_packet=_packet(),
    )
    assert "MAGNITUDE" in result.witnesses[0].tension_types


def test_mechanistic_type_uses_claim_kind_or_packet_node_type():
    result = extract_evidence_level_tensions(
        explorer_report=_report("qualitative_difference"),
        evidence=_evidence(claim_b="mechanism"),
        explorer_packet=_packet(type_b="MechanismClaim"),
    )
    assert "MECHANISTIC" in result.witnesses[0].tension_types


def test_packet_assisted_refined_trigger_uses_pair_local_boundary_evidence():
    evidence = _evidence()
    tension = extract_evidence_level_tensions(
        explorer_report=_report(),
        evidence=evidence,
        explorer_packet=_packet(),
    )
    result = detect_scientific_reframe_triggers(
        explorer_report=_report(),
        evidence=evidence,
        explorer_packet=_packet(),
        evidence_tension_report=tension,
    )
    assert result.trigger_resolution_mode == "evidence_level_refined"
    assert result.source_evidence_tension_report_id == tension.report_id
    assert _assessment(result, "LATENT_VARIABLE").decision == "triggered"
    assert _assessment(result, "REGIME_BOUNDARY").decision == "triggered"


def test_refined_regime_does_not_trigger_from_magnitude_plus_task_diversity_alone():
    evidence = _evidence(
        text_a="The response was 2 fold larger.",
        text_b="The response was 1.2 fold larger.",
    )
    report = _report("quantitative_difference")
    tension = extract_evidence_level_tensions(
        explorer_report=report,
        evidence=evidence,
        explorer_packet=_packet(),
    )
    assert "BOUNDARY" not in tension.witnesses[0].tension_types
    result = detect_scientific_reframe_triggers(
        explorer_report=report,
        evidence=evidence,
        explorer_packet=_packet(),
        evidence_tension_report=tension,
    )
    assert _assessment(result, "REGIME_BOUNDARY").decision == "insufficient_trigger_evidence"


def test_latent_refined_trigger_requires_independent_family_signal():
    evidence = _evidence(papers=("P1", "P1"))
    report = _report("potential_conflict")
    report["evidence_tensions"][0]["paper_ids"] = ["P1"]
    tension = extract_evidence_level_tensions(
        explorer_report=report,
        evidence=evidence,
        explorer_packet=_packet(),
    )
    assert tension.witnesses[0].independent_family_signal is False
    result = detect_scientific_reframe_triggers(
        explorer_report=report,
        evidence=evidence,
        explorer_packet=_packet(),
        evidence_tension_report=tension,
    )
    assert _assessment(result, "LATENT_VARIABLE").decision == "insufficient_trigger_evidence"


def test_explorer_only_mode_preserves_legacy_trigger_contract():
    evidence = _evidence()
    tension = extract_evidence_level_tensions(
        explorer_report=_report(),
        evidence=evidence,
        explorer_packet=None,
    )
    result = detect_scientific_reframe_triggers(
        explorer_report=_report(),
        evidence=evidence,
        evidence_tension_report=tension,
    )
    assert tension.extraction_mode == "explorer_only"
    assert result.trigger_resolution_mode == "legacy_explorer_tension"
    assert _assessment(result, "REGIME_BOUNDARY").decision == "triggered"
