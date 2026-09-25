from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline_core.discovery.prospective_routed_route_compiler import (
    _dominant_hint,
    _editable_fields_from_binding_reasons,
    _replace_run_dir,
)
from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    PreVerifierContractGateV2Row,
)


def _row(hint: str) -> PreVerifierContractGateV2Row:
    binding_status = "READY_FOR_LITERAL_ENDPOINT_BINDING"
    binding_reasons: list[str] = []
    source_reasons: list[str] = []
    claim_kind = "moderator_interaction"
    atomic_kind_supported = True
    prediction_count = 1
    falsifier_count = 1
    shared_identity: bool | None = True

    if hint == "SPECIFICATION_REPAIR_REVIEW":
        binding_status = "INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION"
        binding_reasons = ["missing_required_bridge"]
    elif hint == "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW":
        source_reasons = [
            "prediction_exact_source_binding_cardinality:0"
        ]
        prediction_count = 0
        shared_identity = None
    elif hint == "DECOMPOSE_OR_REGENERATE_REVIEW":
        source_reasons = ["unsupported_atomic_claim_kind:composite"]
        claim_kind = "composite"
        atomic_kind_supported = False
    elif hint != "PROCEED_TO_LITERAL_ENDPOINT_BINDING":
        raise AssertionError("unsupported fixture router hint: " + hint)

    return PreVerifierContractGateV2Row(
        candidate_hypothesis_id="candidate:1",
        final_hypothesis_id="final:1",
        claim_id="claim:" + hint,
        novelty_selection_role="NOVELTY_BEARING",
        claim_kind=claim_kind,
        binding_plan_status=binding_status,
        binding_contract_reason_codes=binding_reasons,
        source_contract_reason_codes=source_reasons,
        gate_status=(
            "READY_FOR_LITERAL_ENDPOINT_BINDING"
            if hint == "PROCEED_TO_LITERAL_ENDPOINT_BINDING"
            else "NOT_READY_FOR_LITERAL_ENDPOINT_BINDING"
        ),
        router_hint=hint,
        atomic_kind_supported=atomic_kind_supported,
        prediction_exact_source_binding_count=prediction_count,
        falsifier_exact_source_binding_count=falsifier_count,
        shared_observable_identity_satisfied=shared_identity,
    )


def test_dominant_hint_uses_frozen_precedence() -> None:
    rows = [
        _row("PROCEED_TO_LITERAL_ENDPOINT_BINDING"),
        _row("SPECIFICATION_REPAIR_REVIEW"),
        _row("SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"),
        _row("DECOMPOSE_OR_REGENERATE_REVIEW"),
    ]
    assert _dominant_hint(rows) == "DECOMPOSE_OR_REGENERATE_REVIEW"


def test_source_contract_outranks_specification_repair() -> None:
    rows = [
        _row("SPECIFICATION_REPAIR_REVIEW"),
        _row("SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"),
    ]
    assert _dominant_hint(rows) == (
        "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"
    )


def test_editable_fields_cover_missing_contract_fields() -> None:
    assert _editable_fields_from_binding_reasons(
        [
            "missing_required_bridge",
            "missing_predicted_observation",
            "missing_falsification_condition",
        ]
    ) == [
        "required_bridge",
        "predicted_observation",
        "falsification_condition",
    ]


def test_editable_fields_cover_identity_failures_without_claim_text() -> None:
    fields = _editable_fields_from_binding_reasons(
        [
            "identity_not_literal_in_required_bridge:0",
            "identity_not_literal_in_predicted_observation:0",
            "identity_not_literal_in_falsification_condition:0",
            "identity_not_literal_in_claim_text:0",
        ]
    )
    assert fields == [
        "required_bridge",
        "predicted_observation",
        "falsification_condition",
    ]


def test_regeneration_replaces_only_run_dir() -> None:
    original = [
        "python",
        "-m",
        "scripts.discovery.run_dac_discovery_e2e",
        "--run-dir",
        "/tmp/original",
        "--source",
        "shape",
        "--target",
        "polarization",
    ]
    replaced = _replace_run_dir(
        original,
        new_run_dir="/tmp/regenerated",
    )
    assert replaced[4] == "/tmp/regenerated"
    assert original[4] == "/tmp/original"
    assert replaced[:4] == original[:4]
    assert replaced[5:] == original[5:]


def test_regeneration_forbids_overwrite_run() -> None:
    with pytest.raises(ValueError, match="must not use --overwrite-run"):
        _replace_run_dir(
            [
                "python",
                "-m",
                "scripts.discovery.run_dac_discovery_e2e",
                "--run-dir",
                "/tmp/original",
                "--overwrite-run",
            ],
            new_run_dir="/tmp/new",
        )


def test_regeneration_requires_run_dir_flag() -> None:
    with pytest.raises(ValueError, match="lacks --run-dir"):
        _replace_run_dir(
            ["python", "-m", "scripts.discovery.run_dac_discovery_e2e"],
            new_run_dir="/tmp/new",
        )
