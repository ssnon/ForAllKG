from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline_core.discovery.reframing.reframe_contracts import (
    DifferentialPredictionDraft,
    DiscriminatingTestDraft,
    ReframeFalsifierDraft,
    ScientificModelDraft,
    ScientificReframeBatchDraft,
    ScientificReframeDraft,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ConditionEvidenceExample,
    ReframeEvidenceStatement,
    ScientificReframeEvidencePacket,
)
from pipeline_core.discovery.reframing.reframe_llm import ReframeDraftGeneration
from pipeline_core.discovery.reframing.reframe_runtime import (
    ScientificReframingShadowRuntime,
)
from pipeline_core.discovery.reframing.trigger_contracts import (
    ReframeOperatorTriggerAssessment,
    ScientificReframeTriggerReport,
)
from pipeline_core.discovery.reframing.trigger_detection import (
    detect_scientific_reframe_triggers,
)


def _evidence(*, condition_diversity: bool = True) -> ScientificReframeEvidencePacket:
    examples = [
        ConditionEvidenceExample(
            object_kind="measurement",
            paper_id="P1",
            chunk_id="c1",
            object_id="m1",
            label="SERS intensity",
            conditions=[{"name": "wavelength", "value_numeric": 532}],
            direct_grounded_object=True,
        )
    ]
    if condition_diversity:
        examples.append(
            ConditionEvidenceExample(
                object_kind="measurement",
                paper_id="P2",
                chunk_id="c2",
                object_id="m2",
                label="SERS intensity",
                conditions=[{"name": "wavelength", "value_numeric": 633}],
                direct_grounded_object=True,
            )
        )
    return ScientificReframeEvidencePacket(
        task_id="task:1",
        question="When does spectral matching govern SERS?",
        source_context_id="context:1",
        source_context_sha256="context-sha",
        premise_statements=[
            ReframeEvidenceStatement(
                statement_id="s1",
                text="Matching is associated with stronger response in one context.",
                epistemic_role="reported",
                claim_kind="association",
                paper_ids=["P1"],
            ),
            ReframeEvidenceStatement(
                statement_id="s2",
                text="An offset maximum is reported in another context.",
                epistemic_role="reported",
                claim_kind="observation",
                paper_ids=["P2"],
            ),
        ],
        gap_statements=[],
        condition_examples=examples,
        grounded_source_chunk_count=len(examples),
        unresolved_grounded_node_count=0,
    )


def _report(tension_type: str = "context_dependency", *, paper_ids=None) -> dict:
    return {
        "schema_version": "exploration-report-v1",
        "report_id": "explorer-report:1",
        "task_id": "task:1",
        "evidence_tensions": [
            {
                "tension_id": "t1",
                "statement_id": "s1",
                "side_a_statement_ids": ["s1"],
                "side_b_statement_ids": ["s2"],
                "tension_type": tension_type,
                "paper_ids": paper_ids if paper_ids is not None else ["P1", "P2"],
            }
        ],
        "unresolved_connections": [],
    }


def _assessment(result, operator_id):
    return next(row for row in result.assessments if row.operator_id == operator_id)


def test_context_tension_triggers_latent_and_regime_when_conditions_diverse():
    result = detect_scientific_reframe_triggers(
        explorer_report=_report(),
        evidence=_evidence(),
    )
    assert _assessment(result, "LATENT_VARIABLE").decision == "triggered"
    assert _assessment(result, "REGIME_BOUNDARY").decision == "triggered"
    assert result.llm_calls_performed == 0
    assert result.scientific_conflict_authority is False


def test_potential_conflict_does_not_create_conflict_authority_or_regime_trigger():
    result = detect_scientific_reframe_triggers(
        explorer_report=_report("potential_conflict"),
        evidence=_evidence(),
    )
    witness = result.tension_witnesses[0]
    assert witness.normalized_tension_type == "POTENTIAL_CONFLICT"
    assert witness.scientific_conflict_authority is False
    assert _assessment(result, "LATENT_VARIABLE").decision == "triggered"
    assert _assessment(result, "REGIME_BOUNDARY").decision == "not_triggered"


def test_regime_requires_condition_diversity_not_just_context_tension():
    result = detect_scientific_reframe_triggers(
        explorer_report=_report(),
        evidence=_evidence(condition_diversity=False),
    )
    assert _assessment(result, "REGIME_BOUNDARY").decision == "insufficient_trigger_evidence"
    assert result.condition_diversity.diversity_present is False


def test_no_tension_means_no_operator_trigger():
    report = _report()
    report["evidence_tensions"] = []
    result = detect_scientific_reframe_triggers(
        explorer_report=report,
        evidence=_evidence(),
    )
    assert _assessment(result, "LATENT_VARIABLE").decision == "not_triggered"
    assert _assessment(result, "REGIME_BOUNDARY").decision == "not_triggered"


def test_report_id_is_deterministic():
    left = detect_scientific_reframe_triggers(
        explorer_report=_report(), evidence=_evidence()
    )
    right = detect_scientific_reframe_triggers(
        explorer_report=_report(), evidence=_evidence()
    )
    assert left.report_id == right.report_id


def test_task_mismatch_fails_closed():
    report = _report()
    report["task_id"] = "task:other"
    with pytest.raises(ValueError, match="task_id"):
        detect_scientific_reframe_triggers(
            explorer_report=report,
            evidence=_evidence(),
        )


def _latent_draft() -> ScientificReframeDraft:
    return ScientificReframeDraft(
        local_id="l1",
        operator_id="LATENT_VARIABLE",
        title="Latent local coupling",
        premise_statement_ids=["s1", "s2"],
        gap_statement_ids=[],
        baseline_model=ScientificModelDraft(
            summary="one direct matching rule",
            explained_statement_ids=["s1", "s2"],
            expected_observations=["matching tracks response"],
        ),
        alternative_model=ScientificModelDraft(
            summary="a latent local coupling construct mediates both observations",
            explained_statement_ids=["s1", "s2"],
            expected_observations=["local coupling predicts both matching and offset cases"],
        ),
        challenged_assumption="far-field matching is sufficient",
        proposed_constructs=["local coupling"],
        latent_constructs=["local coupling"],
        differential_predictions=[
            DifferentialPredictionDraft(
                local_id="d1",
                observable="SERS action spectrum",
                baseline_expectation="tracks far-field resonance",
                alternative_expectation="can remain systematically offset",
                discriminating_outcome="local spectrum predicts the offset",
            )
        ],
        falsifiers=[
            ReframeFalsifierDraft(
                local_id="f1",
                falsifying_outcome="far-field resonance fully predicts response",
            )
        ],
        discriminating_test=DiscriminatingTestDraft(
            test_design="compare far-field and local response",
            primary_observables=["far-field spectrum", "local spectrum", "SERS"],
            baseline_favoring_outcome="far-field alone predicts SERS",
            alternative_favoring_outcome="local coupling explains residual offsets",
        ),
    )


class CountingBackend:
    backend_name = "fake"
    model_name = "fake-model"

    def __init__(self):
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt.operator_id)
        return ReframeDraftGeneration(
            draft=ScientificReframeBatchDraft(
                operator_id=prompt.operator_id,
                candidates=[_latent_draft()],
            )
        )


def _readiness():
    return SimpleNamespace(
        scope_id="task:1",
        assessments=[
            SimpleNamespace(
                operator_id="LATENT_VARIABLE",
                status="ready_now",
                reasons=["ready"],
            ),
            SimpleNamespace(
                operator_id="REGIME_BOUNDARY",
                status="ready_with_optional_enrichment",
                reasons=["ready"],
            ),
        ],
    )


def test_runtime_trigger_gating_skips_untriggered_operator_without_llm_call():
    detected = detect_scientific_reframe_triggers(
        explorer_report=_report("potential_conflict"),
        evidence=_evidence(),
    )
    assert _assessment(detected, "LATENT_VARIABLE").decision == "triggered"
    assert _assessment(detected, "REGIME_BOUNDARY").decision == "not_triggered"

    backend = CountingBackend()
    outcome = ScientificReframingShadowRuntime(backend).run(
        evidence=_evidence(),
        readiness=_readiness(),
        trigger_report=detected,
    )
    assert backend.calls == ["LATENT_VARIABLE"]
    by_operator = {row.operator_id: row for row in outcome.report.runs}
    assert by_operator["LATENT_VARIABLE"].decision == "generated"
    assert by_operator["REGIME_BOUNDARY"].decision == "skipped_not_triggered"
    assert outcome.report.llm_calls_performed == 1
    assert outcome.report.scientific_trigger_evaluated is True
    assert outcome.report.source_trigger_report_id == detected.report_id


def test_runtime_rejects_trigger_lineage_mismatch():
    detected = detect_scientific_reframe_triggers(
        explorer_report=_report(), evidence=_evidence()
    )
    payload = detected.model_dump(mode="json")
    payload["source_context_sha256"] = "wrong"
    bad = ScientificReframeTriggerReport.model_validate(payload)
    with pytest.raises(ValueError, match="context_sha256"):
        ScientificReframingShadowRuntime(CountingBackend()).run(
            evidence=_evidence(),
            readiness=_readiness(),
            trigger_report=bad,
        )
