from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.prospective_routed_initial_intake_v2 import (
    build_initial_intake_v2,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_ready_requires_both_n10_inputs(tmp_path: Path) -> None:
    _write(
        tmp_path / "e2e_runner.manifest.json",
        {"status": "complete"},
    )
    _write(
        tmp_path / "hypothesis_axis_a4.portfolio.json",
        {"hypotheses": [{"id": "h1"}]},
    )
    _write(
        tmp_path / "novelty_refinement_a6.n10.candidate.portfolio.json",
        {},
    )
    _write(
        tmp_path / "novelty_refinement_a6.n10.certification.json",
        {},
    )

    report = build_initial_intake_v2(
        case_id="P16",
        run_dir=tmp_path,
    )
    assert report.disposition == "READY_FOR_RELATIONAL_PREVERIFIER"
    assert report.binding_plan_should_run is True
    assert report.gate_v2_should_run is True
    assert report.routed_dispatch_should_run is True


def test_alpha4_zero_is_upstream_no_binding_inputs(tmp_path: Path) -> None:
    _write(
        tmp_path / "e2e_runner.manifest.json",
        {"status": "complete"},
    )
    _write(
        tmp_path / "hypothesis_axis_a4.portfolio.json",
        {"hypotheses": []},
    )

    report = build_initial_intake_v2(
        case_id="P16",
        run_dir=tmp_path,
    )
    assert report.disposition == "UPSTREAM_NO_RELATIONAL_BINDING_INPUTS"
    assert "alpha4_no_surviving_hypotheses" in report.reason_codes
    assert report.binding_plan_should_run is False


def test_missing_candidate_only_is_fail_closed(tmp_path: Path) -> None:
    _write(
        tmp_path / "hypothesis_axis_a4.portfolio.json",
        {"hypotheses": [{"id": "h1"}]},
    )
    _write(
        tmp_path / "novelty_refinement_a6.n10.certification.json",
        {},
    )

    report = build_initial_intake_v2(
        case_id="P17",
        run_dir=tmp_path,
    )
    assert report.disposition == "UPSTREAM_NO_RELATIONAL_BINDING_INPUTS"
    assert "missing_n10_candidate_portfolio" in report.reason_codes
    assert report.gate_v2_should_run is False


def test_missing_certification_only_is_fail_closed(tmp_path: Path) -> None:
    _write(
        tmp_path / "hypothesis_axis_a4.portfolio.json",
        {"hypotheses": [{"id": "h1"}]},
    )
    _write(
        tmp_path / "novelty_refinement_a6.n10.candidate.portfolio.json",
        {},
    )

    report = build_initial_intake_v2(
        case_id="P18",
        run_dir=tmp_path,
    )
    assert report.disposition == "UPSTREAM_NO_RELATIONAL_BINDING_INPUTS"
    assert "missing_n10_certification" in report.reason_codes
    assert report.routed_dispatch_should_run is False


def test_guard_performs_no_scientific_actions(tmp_path: Path) -> None:
    report = build_initial_intake_v2(
        case_id="P19",
        run_dir=tmp_path,
    )
    assert report.llm_calls_performed == 0
    assert report.scientific_mutation_performed is False
    assert report.retrieval_performed is False
    assert report.endpoint_binding_performed is False
    assert report.verifier_result_observed is False
