from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.discovery_axis_contracts import (
    AxisAttemptRecord,
    DiscoveryHypothesisLineage,
    DiscoveryAxisSynthesisReport,
)


def test_historical_lineage_defaults_to_not_assessed() -> None:
    row = DiscoveryHypothesisLineage(
        hypothesis_id="h:1",
        axis_id="axis:1",
        inspiration_id="insp:1",
        candidate_unit_id="unit:1",
        axis_fidelity_status="pass",
        internal_novelty_status="corpus_distinct_candidate",
    )

    assert row.inference_status == "not_assessed"
    assert row.inference_repaired is False


def test_inference_attempt_contract_is_supported() -> None:
    row = AxisAttemptRecord(
        axis_id="axis:1",
        stage="inference_repair",
        generation_index=2,
        decision="inference_rejected",
        hypothesis_id="h:1",
        title="test",
        fidelity_status="pass",
        inference_status="reframe_required",
        repair_reason="unsupported specificity",
    )

    assert row.stage == "inference_repair"
    assert row.decision == "inference_rejected"
    assert row.inference_status == "reframe_required"


def test_historical_synthesis_policy_v1_remains_parseable() -> None:
    report = DiscoveryAxisSynthesisReport(
        report_id="report:1",
        report_sha256="sha",
        source_dual_context_id="dual:1",
        source_dual_context_sha256="dualsha",
        axis_plan_id="plan:1",
        axis_plan_sha256="plansha",
        final_portfolio_id="portfolio:1",
        final_portfolio_sha256="portsha",
        attempted_axis_count=0,
        accepted_hypothesis_count=0,
        lineages=[],
        attempts=[],
        policy_version="discovery-axis-synthesis-policy-v1",
    )

    assert (
        report.policy_version
        == "discovery-axis-synthesis-policy-v1"
    )


def test_new_synthesis_policy_defaults_to_v2() -> None:
    report = DiscoveryAxisSynthesisReport(
        report_id="report:2",
        report_sha256="sha",
        source_dual_context_id="dual:1",
        source_dual_context_sha256="dualsha",
        axis_plan_id="plan:1",
        axis_plan_sha256="plansha",
        final_portfolio_id="portfolio:1",
        final_portfolio_sha256="portsha",
        attempted_axis_count=0,
        accepted_hypothesis_count=0,
        lineages=[],
        attempts=[],
    )

    assert (
        report.policy_version
        == "discovery-axis-synthesis-policy-v2"
    )


def test_runtime_orders_inference_before_internal_novelty() -> None:
    text = Path(
        "pipeline_core/discovery/discovery_axis_runtime.py"
    ).read_text(encoding="utf-8")

    inference_marker = (
        "# Discovery-axis inference-strength gate."
    )
    novelty_marker = (
        "novelty = self._novelty_card(dual, portfolio)"
    )

    assert inference_marker in text
    assert novelty_marker in text

    assert (
        text.index(inference_marker)
        < text.index(novelty_marker)
    )


def test_novelty_repair_rechecks_inference() -> None:
    text = Path(
        "pipeline_core/discovery/discovery_axis_runtime.py"
    ).read_text(encoding="utf-8")

    # Verify ordering rather than searching for the runtime-concatenated
    # representation of adjacent Python string literals.
    novelty_repair_fidelity = text.index(
        "novelty repair lost assigned-axis fidelity"
    )

    inference_recheck = text.index(
        "inference_outcome = self.inference_critic.review(",
        novelty_repair_fidelity,
    )

    novelty_reassessment = text.index(
        "novelty = self._novelty_card(dual, portfolio)",
        inference_recheck,
    )

    assert (
        novelty_repair_fidelity
        < inference_recheck
        < novelty_reassessment
    )

    # Preserve an explicit diagnostic assertion without depending on
    # Python's adjacent-string-literal runtime concatenation.
    assert '"novelty repair introduced or retained "' in text
    assert '"unsupported inference specificity"' in text


def test_production_runner_supplies_inference_critic() -> None:
    text = Path(
        "scripts/discovery/run_discovery_axis_hypothesis_maker.py"
    ).read_text(encoding="utf-8")

    assert (
        "InstructorOpenAICompatibleAxisInferenceBackend"
        in text
    )
    assert (
        "inference_critic=inference_critic"
        in text
    )
    assert (
        '".inference.json"'
        in text
    )


def test_e2e_requires_inference_artifact() -> None:
    text = Path(
        "scripts/discovery/run_dac_discovery_e2e.py"
    ).read_text(encoding="utf-8")

    assert (
        'axis_inference = run / "hypothesis_axis_a4.inference.json"'
        in text
    )
    assert (
        '"--inference-critic-model"'
        in text
    )
