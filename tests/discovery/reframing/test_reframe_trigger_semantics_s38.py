from __future__ import annotations

from pipeline_core.discovery.reframing.evidence_tension import (
    extract_evidence_level_tensions,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ReframeEvidenceStatement,
    ScientificReframeEvidencePacket,
)
from pipeline_core.discovery.reframing.trigger_detection import (
    detect_scientific_reframe_triggers,
)


def _statement(
    statement_id: str,
    text: str,
    *,
    claim_kind: str = "observation",
    paper_id: str = "P1",
) -> ReframeEvidenceStatement:
    return ReframeEvidenceStatement(
        statement_id=statement_id,
        text=text,
        epistemic_role="reported",
        claim_kind=claim_kind,
        paper_ids=[paper_id],
    )


def _evidence(
    premises: list[ReframeEvidenceStatement],
    gaps: list[ReframeEvidenceStatement] | None = None,
) -> ScientificReframeEvidencePacket:
    return ScientificReframeEvidencePacket(
        task_id="task:semantic-trigger",
        question="Which response model is scientifically worth one shadow attempt?",
        source_context_id="context:semantic-trigger",
        source_context_sha256="context-sha",
        premise_statements=premises,
        gap_statements=gaps or [],
        condition_examples=[],
        grounded_source_chunk_count=max(1, len(premises)),
        unresolved_grounded_node_count=0,
    )


def _report(
    *,
    side_a: list[str] | None = None,
    side_b: list[str] | None = None,
    tension_type: str = "context_dependency",
) -> dict:
    rows = []
    if side_a is not None or side_b is not None:
        rows.append(
            {
                "tension_id": "tension:1",
                "side_a_statement_ids": side_a or [],
                "side_b_statement_ids": side_b or [],
                "tension_type": tension_type,
                "paper_ids": ["P1", "P2"],
            }
        )
    return {
        "schema_version": "exploration-report-v1",
        "report_id": "report:semantic-trigger",
        "task_id": "task:semantic-trigger",
        "evidence_tensions": rows,
    }


def _packet() -> dict:
    return {
        "task": {"task_id": "task:semantic-trigger"},
        "evidence_catalog": {"nodes": {}},
    }


def _detect(evidence: ScientificReframeEvidencePacket, report: dict):
    tensions = extract_evidence_level_tensions(
        explorer_report=report,
        evidence=evidence,
        explorer_packet=_packet(),
    )
    triggers = detect_scientific_reframe_triggers(
        explorer_report=report,
        evidence=evidence,
        explorer_packet=_packet(),
        evidence_tension_report=tensions,
    )
    return tensions, triggers


def _assessment(result, operator_id: str):
    return next(row for row in result.assessments if row.operator_id == operator_id)


def test_latent_rejects_one_sided_nominal_tension():
    evidence = _evidence(
        [
            _statement(
                "s1",
                "Compact aggregates produced strong SERS enhancement and dense hotspots.",
                claim_kind="mechanism",
                paper_id="P1",
            ),
            _statement(
                "s2",
                "Another aggregate geometry also produced strong Raman signal.",
                paper_id="P2",
            ),
        ]
    )
    tensions, triggers = _detect(evidence, _report(side_a=["s1", "s2"], side_b=[]))
    witness = tensions.witnesses[0]
    assert witness.paired_grounded_sides is False
    assert witness.paired_response_signal is False
    latent = _assessment(triggers, "LATENT_VARIABLE")
    assert latent.decision == "insufficient_trigger_evidence"
    assert latent.scientific_trigger_signal is False


def test_cross_variable_parameter_change_is_not_directional_or_regime_boundary():
    evidence = _evidence(
        [
            _statement(
                "s1",
                "The average interparticle gap decreased from 23 nm to 4.1 nm.",
                paper_id="P1",
            ),
            _statement(
                "s2",
                "Electric field intensity increased when the interparticle distance was below 2 nm.",
                claim_kind="mechanism",
                paper_id="P2",
            ),
        ]
    )
    tensions, triggers = _detect(
        evidence,
        _report(side_a=["s1"], side_b=["s2"], tension_type="context_dependency"),
    )
    witness = tensions.witnesses[0]
    assert witness.paired_response_signal is False
    assert "DIRECTIONAL" not in witness.tension_types
    assert "BOUNDARY" not in witness.tension_types
    assert _assessment(triggers, "LATENT_VARIABLE").scientific_trigger_signal is False
    assert _assessment(triggers, "REGIME_BOUNDARY").scientific_trigger_signal is False


def test_mediation_gap_can_trigger_latent_without_explicit_tension():
    evidence = _evidence(
        [
            _statement(
                "s1",
                "A capture layer increased analyte delivery near the SERS substrate.",
                claim_kind="association",
                paper_id="P1",
            ),
            _statement(
                "s2",
                "High-density electromagnetic hotspots increased Raman enhancement.",
                claim_kind="mechanism",
                paper_id="P2",
            ),
            _statement(
                "s3",
                "More uniform hotspot sampling improved SERS reproducibility.",
                claim_kind="mechanism",
                paper_id="P3",
            ),
        ],
        [
            _statement(
                "g1",
                "The packet does not test whether improved analyte delivery can compensate for spatially heterogeneous electromagnetic enhancement.",
                claim_kind="scope_limit",
                paper_id="P1",
            )
        ],
    )
    _, triggers = _detect(evidence, _report())
    latent = _assessment(triggers, "LATENT_VARIABLE")
    assert latent.decision == "triggered"
    assert latent.scientific_trigger_signal is True
    signal = next(row for row in triggers.direct_trigger_signals if row.kind == "MEDIATION_GAP")
    assert signal.target_operator_id == "LATENT_VARIABLE"
    assert signal.strength == "sufficient"
    assert signal.scientific_authority is False


def test_direct_non_monotonic_response_triggers_regime():
    evidence = _evidence(
        [
            _statement(
                "s1",
                "SERS enhancement generally increases as the gap decreases, but the Raman signal decreases again at the smallest gaps studied.",
                paper_id="P1",
            ),
            _statement(
                "s2",
                "A 20 nm gap produced the best SERS enhancement in a separate structure.",
                paper_id="P2",
            ),
        ]
    )
    _, triggers = _detect(evidence, _report())
    regime = _assessment(triggers, "REGIME_BOUNDARY")
    assert regime.decision == "triggered"
    assert regime.scientific_trigger_signal is True
    assert any(
        row.kind == "NON_MONOTONIC_RESPONSE" and row.strength == "sufficient"
        for row in triggers.direct_trigger_signals
    )


def test_finite_optimum_alone_is_supporting_not_sufficient_for_regime():
    evidence = _evidence(
        [
            _statement(
                "s1",
                "A 20 nm gap produced the best SERS enhancement among the tested structures.",
                paper_id="P1",
            )
        ]
    )
    _, triggers = _detect(evidence, _report())
    regime = _assessment(triggers, "REGIME_BOUNDARY")
    assert regime.decision == "insufficient_trigger_evidence"
    assert regime.scientific_trigger_signal is False
    signal = next(row for row in triggers.direct_trigger_signals if row.kind == "FINITE_OPTIMUM")
    assert signal.strength == "supporting"


def test_explicit_saturation_or_threshold_is_sufficient_for_regime():
    evidence = _evidence(
        [
            _statement(
                "s1",
                "The Raman response reached a plateau and showed no further increase above the threshold concentration.",
                paper_id="P1",
            )
        ]
    )
    _, triggers = _detect(evidence, _report())
    regime = _assessment(triggers, "REGIME_BOUNDARY")
    kinds = {row.kind for row in triggers.direct_trigger_signals}
    assert regime.scientific_trigger_signal is True
    assert "SATURATION_OR_PLATEAU" in kinds
    assert "EXPLICIT_THRESHOLD" in kinds


def test_paired_response_tension_still_triggers_latent():
    evidence = _evidence(
        [
            _statement(
                "s1",
                "One substrate showed higher SERS response when LSPR matched excitation.",
                claim_kind="mechanism",
                paper_id="P1",
            ),
            _statement(
                "s2",
                "Another substrate showed maximum Raman enhancement at an offset resonance.",
                claim_kind="mechanism",
                paper_id="P2",
            ),
        ]
    )
    tensions, triggers = _detect(evidence, _report(side_a=["s1"], side_b=["s2"]))
    witness = tensions.witnesses[0]
    assert witness.paired_grounded_sides is True
    assert witness.paired_response_signal is True
    assert _assessment(triggers, "LATENT_VARIABLE").scientific_trigger_signal is True


def test_direct_signal_ids_are_stable_and_non_authoritative():
    evidence = _evidence(
        [
            _statement(
                "s1",
                "The SERS response increased initially but decreased at the smallest gap.",
                paper_id="P1",
            )
        ]
    )
    first = _detect(evidence, _report())[1]
    second = _detect(evidence, _report())[1]
    assert [row.signal_id for row in first.direct_trigger_signals] == [
        row.signal_id for row in second.direct_trigger_signals
    ]
    assert all(row.diagnostic_only for row in first.direct_trigger_signals)
    assert all(row.scientific_authority is False for row in first.direct_trigger_signals)
    assert first.production_selection_changed is False
    assert first.novelty_authority is False
