from __future__ import annotations

from pipeline_core.discovery.reframing.cross_domain_diagnostics import (
    build_cross_domain_task_trigger_diagnostic,
    build_cross_domain_trigger_diagnostic_pack,
)
from pipeline_core.discovery.reframing.evidence_tension import (
    EvidenceLevelTensionWitness,
    ScientificEvidenceTensionReport,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ReframeEvidenceStatement,
    ScientificReframeEvidencePacket,
)
from pipeline_core.discovery.reframing.trigger_contracts import (
    ConditionDiversitySignal,
    DirectScientificTriggerSignal,
    ReframeOperatorTriggerAssessment,
    ScientificReframeTriggerReport,
    ScientificTensionWitness,
)


def _statement(statement_id: str, text: str, claim_kind: str = "observation") -> ReframeEvidenceStatement:
    return ReframeEvidenceStatement(
        statement_id=statement_id,
        text=text,
        epistemic_role="reported",
        claim_kind=claim_kind,
        paper_ids=["P1"],
    )


def _evidence(*, premise_texts: list[str], gap_texts: list[str] | None = None) -> ScientificReframeEvidencePacket:
    return ScientificReframeEvidencePacket(
        task_id="task:cross",
        question="How does the response change?",
        source_context_id="ctx:1",
        source_context_sha256="a" * 64,
        premise_statements=[
            _statement(f"p{index}", text)
            for index, text in enumerate(premise_texts, start=1)
        ],
        gap_statements=[
            _statement(f"g{index}", text, "scope_limit")
            for index, text in enumerate(gap_texts or [], start=1)
        ],
        grounded_source_chunk_count=3,
        unresolved_grounded_node_count=0,
    )


def _tensions(witnesses: list[EvidenceLevelTensionWitness] | None = None) -> ScientificEvidenceTensionReport:
    return ScientificEvidenceTensionReport(
        report_id="tension-report:1",
        source_task_id="task:cross",
        source_context_id="ctx:1",
        source_context_sha256="a" * 64,
        source_explorer_report_sha256="b" * 64,
        source_evidence_sha256="c" * 64,
        extraction_mode="packet_assisted",
        witnesses=witnesses or [],
        tension_type_counts={},
    )


def _trigger(
    *,
    witnesses: list[ScientificTensionWitness] | None = None,
    direct: list[DirectScientificTriggerSignal] | None = None,
) -> ScientificReframeTriggerReport:
    return ScientificReframeTriggerReport(
        report_id="trigger-report:1",
        source_task_id="task:cross",
        source_context_id="ctx:1",
        source_context_sha256="a" * 64,
        source_explorer_report_sha256="b" * 64,
        source_evidence_sha256="c" * 64,
        source_evidence_tension_report_id="tension-report:1",
        trigger_resolution_mode="evidence_level_refined",
        tension_witnesses=witnesses or [],
        direct_trigger_signals=direct or [],
        condition_diversity=ConditionDiversitySignal(
            example_count=0,
            paper_count=0,
            distinct_condition_signature_count=0,
        ),
        assessments=[
            ReframeOperatorTriggerAssessment(
                operator_id="LATENT_VARIABLE",
                decision="not_triggered",
                scientific_trigger_signal=False,
            ),
            ReframeOperatorTriggerAssessment(
                operator_id="REGIME_BOUNDARY",
                decision="not_triggered",
                scientific_trigger_signal=False,
            ),
        ],
    )


def _diagnostic(evidence, tensions=None, trigger=None, explorer=None):
    return build_cross_domain_task_trigger_diagnostic(
        task_key="cross/default",
        case_key="cross",
        canonical_dir="/tmp/cross",
        trigger_pattern="none",
        evidence=evidence,
        tensions=tensions or _tensions(),
        trigger=trigger or _trigger(),
        explorer_report=explorer or {"evidence_tensions": []},
    )


def test_sers_like_response_cue_is_visible_but_not_scientific_authority():
    row = _diagnostic(_evidence(premise_texts=["SERS signal increased with gap narrowing."]))
    trace = row.statement_traces[0]
    assert trace.frozen_trigger_response_cue.matched is True
    assert "signal" in [value.casefold() for value in trace.frozen_trigger_response_cue.matches]
    assert trace.scientific_response_authority is False


def test_dac_her_overpotential_is_not_silently_promoted_into_frozen_response_cue():
    row = _diagnostic(
        _evidence(premise_texts=["HER overpotential decreased as current density increased."])
    )
    trace = row.statement_traces[0]
    assert trace.frozen_trigger_response_cue.matched is False
    assert row.summary.premise_response_cue_match_count == 0
    assert "NO_PREMISE_MATCHES_FROZEN_TRIGGER_RESPONSE_CUE" in row.summary.diagnostic_flags


def test_zero_explorer_tensions_is_distinguished_from_grounding_loss():
    row = _diagnostic(_evidence(premise_texts=["The catalytic activity changed with coordination."]))
    assert row.summary.explorer_raw_tension_count == 0
    assert "NO_UPSTREAM_EXPLORER_TENSIONS" in row.summary.diagnostic_flags
    assert "EXPLORER_TENSIONS_DROPPED_BEFORE_GROUNDED_WITNESS" not in row.summary.diagnostic_flags


def test_raw_explorer_tension_without_grounded_reference_is_traced():
    row = _diagnostic(
        _evidence(premise_texts=["The response increased."]),
        explorer={
            "evidence_tensions": [
                {
                    "tension_id": "t1",
                    "tension_type": "context_dependency",
                    "side_a_statement_ids": ["missing-a"],
                    "side_b_statement_ids": ["missing-b"],
                }
            ]
        },
    )
    assert row.summary.explorer_raw_tension_count == 1
    assert row.summary.grounded_evidence_tension_witness_count == 0
    assert "EXPLORER_TENSIONS_DROPPED_BEFORE_GROUNDED_WITNESS" in row.summary.diagnostic_flags
    assert row.explorer_tension_traces[0].grounded_reference_present is False


def test_strong_mediation_gap_without_actual_signal_is_explicit():
    row = _diagnostic(
        _evidence(
            premise_texts=["The signal changed with catalyst structure."],
            gap_texts=["It is unresolved whether adsorption mediates the activity relation."],
        )
    )
    gap = [trace for trace in row.statement_traces if trace.role == "gap"][0]
    assert gap.strong_mediation_gap_cue.matched is True
    assert "MEDIATION_GAP_CUE_PRESENT_WITHOUT_ACTUAL_DIRECT_SIGNAL" in row.summary.diagnostic_flags


def test_regime_statement_cues_are_reconstructed_without_changing_trigger():
    row = _diagnostic(
        _evidence(
            premise_texts=[
                "SERS response increased with gap size but decreased at the smallest gap."
            ]
        )
    )
    trace = row.statement_traces[0]
    assert "NON_MONOTONIC_RESPONSE" in trace.potential_direct_signal_kinds
    assert "STATEMENT_LEVEL_REGIME_CUE_PRESENT_WITHOUT_ACTUAL_DIRECT_SIGNAL" in row.summary.diagnostic_flags
    assert row.trigger_pattern == "none"


def test_actual_direct_signal_is_carried_without_reclassification():
    signal = DirectScientificTriggerSignal(
        signal_id="direct:1",
        kind="FINITE_OPTIMUM",
        target_operator_id="REGIME_BOUNDARY",
        strength="supporting",
        statement_ids=["p1"],
        reasons=["diagnostic"],
    )
    trigger = _trigger(direct=[signal])
    row = _diagnostic(
        _evidence(premise_texts=["SERS response was best at a 20 nm gap."]),
        trigger=trigger,
    )
    assert row.summary.actual_direct_trigger_signal_count == 1
    assert row.actual_direct_trigger_signals[0]["kind"] == "FINITE_OPTIMUM"
    assert "NO_ACTUAL_DIRECT_TRIGGER_SIGNALS" not in row.summary.diagnostic_flags


def test_pack_rejects_semantics_drift_and_remains_non_authoritative():
    task = _diagnostic(_evidence(premise_texts=["HER overpotential decreased."]))
    try:
        build_cross_domain_trigger_diagnostic_pack(
            source_validation_id="validation:1",
            validation_domain_label="dac_her",
            frozen_semantics_fingerprint="a" * 64,
            current_semantics_fingerprint="b" * 64,
            tasks=[task],
        )
    except ValueError as exc:
        assert "frozen semantics changed" in str(exc)
    else:
        raise AssertionError("semantic drift must be rejected")

    pack = build_cross_domain_trigger_diagnostic_pack(
        source_validation_id="validation:1",
        validation_domain_label="dac_her",
        frozen_semantics_fingerprint="a" * 64,
        current_semantics_fingerprint="a" * 64,
        tasks=[task],
    )
    assert pack.task_count == 1
    assert pack.llm_calls_performed == 0
    assert pack.trigger_semantics_modified is False
    assert pack.trigger_quality_evaluated is False
    assert pack.generalization_claim_established is False
