from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import dac_her.alpha4c5h2_postmortem as postmortem
from dac_her.alpha4c5h2_postmortem import (
    ALPHA4C5H2_POSTMORTEM_SEMANTICS_ID,
    EXPECTED_FAILURE_ERROR_CODES,
    build_postmortem_manifest,
    verify_postmortem_manifest,
)
from dac_her.alpha4c5h1_reserve_b import (
    ALPHA4C5H1_PROTOCOL_SEMANTICS_ID,
    EXPECTED_5H_FREEZE_ID,
)
from dac_her.hypothesis_trend_directional_run_record import (
    HYPOTHESIS_TREND_DIRECTIONAL_RUNTIME_SEMANTICS_ID,
)
from dac_her.hypothesis_trend_directional_validator import (
    HYPOTHESIS_TREND_DIRECTIONAL_VALIDATOR_SEMANTICS_ID,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _paper_ids() -> list[str]:
    return [f"SYNTH_{index:02d}" for index in range(25)]


def _fake_protocol():
    return SimpleNamespace(
        protocol_id="synthetic_h1_protocol",
        protocol_sha256="a" * 64,
        semantics_id=ALPHA4C5H1_PROTOCOL_SEMANTICS_ID,
        campaign_id="sers_alpha4c5h1_reserve_b_v1",
        reserve_partition="reserve_b",
        domain_profile_id="sers_au_ag",
        evaluation_root="evaluation/sers_alpha4c5h1/reserve_b_v1",
        reserve_paper_ids=_paper_ids(),
        five_h_freeze_id=EXPECTED_5H_FREEZE_ID,
        trend_semantics_id="sers_au_ag_trend_v6r2_alpha4c5g2r2",
    )


def _seed_closed_failure(
    root: Path,
    *,
    error_codes: tuple[str, ...] = EXPECTED_FAILURE_ERROR_CODES,
    repair_attempts: int = 1,
) -> Path:
    protocol_path = (
        root
        / "evaluation/sers_alpha4c5h1/reserve_b_v1/control/"
        "execution_protocol.json"
    )
    _write_json(protocol_path, {"synthetic": True})

    eval_root = root / "evaluation/sers_alpha4c5h1/reserve_b_v1"
    hypothesis = eval_root / "hypothesis"
    protocol = _fake_protocol()

    _write_json(
        eval_root / "consumption_started.json",
        {
            "campaign_id": protocol.campaign_id,
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.protocol_sha256,
            "reserve_partition": "reserve_b",
            "paper_ids": protocol.reserve_paper_ids,
            "trend_semantics_id": protocol.trend_semantics_id,
            "reserve_consumed": True,
        },
    )
    failure = {
        "campaign_id": protocol.campaign_id,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "reserve_consumed": True,
        "accepted": False,
        "rerun_allowed": False,
        "reserve_b_failure_authorizes_tuning": False,
        "automatic_scientific_output_rollback": False,
    }
    _write_json(eval_root / "CAMPAIGN_FAIL.json", failure)
    _write_json(
        eval_root / "campaign_manifest.json",
        {**failure, "state": "fail"},
    )
    (eval_root / "command_log.jsonl").write_text(
        '{"stage":"direction_aware_hypothesis_maker_llm","returncode":2}\n',
        encoding="utf-8",
    )

    issues = [
        {
            "severity": "error",
            "code": code,
            "location": "synthetic",
            "message": "synthetic",
        }
        for code in error_codes
    ]
    _write_json(
        hypothesis / "reserve_maker.validation.json",
        {
            "semantics_id":
                HYPOTHESIS_TREND_DIRECTIONAL_VALIDATOR_SEMANTICS_ID,
            "passes": False,
            "errors": len(issues),
            "warnings": 0,
            "issues": issues,
        },
    )
    _write_json(
        hypothesis / "reserve_maker.run.json",
        {
            "runtime_semantics_id":
                HYPOTHESIS_TREND_DIRECTIONAL_RUNTIME_SEMANTICS_ID,
            "generation_attempts": 1,
            "repair_attempts": repair_attempts,
            "final_validation_passed": False,
            "failure_stage": "validation",
            "validation_errors": len(issues),
            "validation_warnings": 0,
            "max_repairs": 1,
        },
    )
    _write_json(
        hypothesis / "reserve_maker.draft.json",
        {"synthetic": "draft0"},
    )
    for index in range(1, repair_attempts + 1):
        _write_json(
            hypothesis / f"reserve_maker.repair{index}.draft.json",
            {"synthetic": f"repair{index}"},
        )
    _write_json(
        hypothesis / "reserve_maker.rejected_portfolio.json",
        {"synthetic": "rejected"},
    )
    _write_json(
        hypothesis / "reserve_maker.directional_exposure.json",
        {"synthetic": "exposure"},
    )
    return protocol_path


@pytest.fixture
def closed_failure(tmp_path: Path, monkeypatch):
    protocol_path = _seed_closed_failure(tmp_path)
    protocol = _fake_protocol()
    monkeypatch.setattr(
        postmortem,
        "load_h1_protocol",
        lambda path: protocol,
    )
    monkeypatch.setattr(
        postmortem,
        "verify_h1_protocol",
        lambda *, root, protocol: [],
    )
    return tmp_path, protocol_path


def test_build_postmortem_freezes_exact_incident_signature(
    closed_failure,
):
    root, protocol_path = closed_failure
    manifest = build_postmortem_manifest(
        root=root,
        protocol_path=protocol_path,
    )

    assert manifest.semantics_id == (
        ALPHA4C5H2_POSTMORTEM_SEMANTICS_ID
    )
    assert manifest.reserve_consumed is True
    assert manifest.campaign_terminal_state == "fail"
    assert manifest.campaign_closed is True
    assert manifest.failure_stage == "validation"
    assert manifest.observed_error_codes == sorted(
        EXPECTED_FAILURE_ERROR_CODES
    )
    assert manifest.maker_repair_attempts == 1
    assert len(manifest.maker_draft_artifacts) == 2
    assert manifest.rerun_allowed is False
    assert manifest.reserve_reuse_allowed is False
    assert manifest.reserve_b_failure_authorizes_tuning is False
    assert manifest.scientific_transformation_performed is False
    assert manifest.llm_calls == 0


def test_refuses_campaign_with_pass_marker(
    closed_failure,
):
    root, protocol_path = closed_failure
    _write_json(
        root
        / "evaluation/sers_alpha4c5h1/reserve_b_v1/"
        "CAMPAIGN_PASS.json",
        {"accepted": True},
    )
    with pytest.raises(ValueError, match="CAMPAIGN_PASS"):
        build_postmortem_manifest(
            root=root,
            protocol_path=protocol_path,
        )


def test_refuses_wrong_validation_error_signature(
    tmp_path: Path,
    monkeypatch,
):
    protocol_path = _seed_closed_failure(
        tmp_path,
        error_codes=("SOME_OTHER_ERROR",),
    )
    protocol = _fake_protocol()
    monkeypatch.setattr(
        postmortem,
        "load_h1_protocol",
        lambda path: protocol,
    )
    monkeypatch.setattr(
        postmortem,
        "verify_h1_protocol",
        lambda *, root, protocol: [],
    )

    with pytest.raises(ValueError, match="error signature mismatch"):
        build_postmortem_manifest(
            root=tmp_path,
            protocol_path=protocol_path,
        )


def test_refuses_downstream_evaluation_after_maker_failure(
    closed_failure,
):
    root, protocol_path = closed_failure
    _write_json(
        root
        / "evaluation/sers_alpha4c5h1/reserve_b_v1/"
        "hypothesis/reserve_evaluation.json",
        {"accepted": False},
    )
    with pytest.raises(ValueError, match="Downstream 5e evaluation"):
        build_postmortem_manifest(
            root=root,
            protocol_path=protocol_path,
        )


def test_frozen_binding_detects_later_artifact_drift(
    closed_failure,
):
    root, protocol_path = closed_failure
    manifest = build_postmortem_manifest(
        root=root,
        protocol_path=protocol_path,
    )
    assert verify_postmortem_manifest(
        root=root,
        manifest=manifest,
    ) == []

    marker = (
        root
        / "evaluation/sers_alpha4c5h1/reserve_b_v1/"
        "consumption_started.json"
    )
    marker.write_text("{}\n", encoding="utf-8")
    issues = verify_postmortem_manifest(
        root=root,
        manifest=manifest,
    )
    assert any(
        "frozen artifact SHA drift" in issue
        for issue in issues
    )
