from __future__ import annotations

from pathlib import Path

import pytest

from scripts.discovery.run_dac_discovery_e2e import (
    _open_world_discovery_output_contract,
    _stage8_axis_plan_input,
    _validate_open_world_discovery_parent_manifest,
)


def _valid_manifest() -> dict:
    return {
        "provider_plan_sha256": "a" * 64,
        "external_literature_authority":
            "INSPIRATION_ONLY",
        "positive_premise_authority_changed":
            False,
        "novelty_authority_created":
            False,
        "hypothesis_generation_performed":
            False,
        "canonical_fallback_authorized":
            False,
        "external_axis_count": 3,
    }


def test_output_contract_uses_one_open_world_prefix(
    tmp_path: Path,
) -> None:
    contract = _open_world_discovery_output_contract(
        tmp_path
    )

    assert (
        contract["external_axis_plan"]
        == tmp_path
        / "hypothesis_axis_open_world.external_axis_plan.json"
    )
    assert (
        contract["manifest"]
        == tmp_path
        / "hypothesis_axis_open_world.manifest.json"
    )
    assert (
        contract["prompt"]
        == tmp_path
        / "hypothesis_axis_open_world.axis_synthesis_prompt.json"
    )


def test_default_stage8_plan_remains_control_plan() -> None:
    control = Path("control.axis_plan.json")
    external = Path("external.axis_plan.json")

    assert (
        _stage8_axis_plan_input(
            control_plan=control,
            open_world_plan=external,
            open_world_enabled=False,
        )
        == control
    )


def test_opt_in_stage8_plan_uses_external_plan() -> None:
    control = Path("control.axis_plan.json")
    external = Path("external.axis_plan.json")

    assert (
        _stage8_axis_plan_input(
            control_plan=control,
            open_world_plan=external,
            open_world_enabled=True,
        )
        == external
    )


def test_opt_in_without_external_plan_fails_closed() -> None:
    with pytest.raises(
        RuntimeError,
        match="no external axis plan",
    ):
        _stage8_axis_plan_input(
            control_plan=Path("control.json"),
            open_world_plan=None,
            open_world_enabled=True,
        )


def test_parent_manifest_accepts_inspiration_only_contract() -> None:
    _validate_open_world_discovery_parent_manifest(
        _valid_manifest(),
        expected_provider_plan_sha256="a" * 64,
    )


def test_parent_manifest_rejects_provider_plan_drift() -> None:
    payload = _valid_manifest()
    payload["provider_plan_sha256"] = "b" * 64

    with pytest.raises(
        RuntimeError,
        match="provider plan drifted",
    ):
        _validate_open_world_discovery_parent_manifest(
            payload,
            expected_provider_plan_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "field",
    [
        "positive_premise_authority_changed",
        "novelty_authority_created",
        "hypothesis_generation_performed",
        "canonical_fallback_authorized",
    ],
)
def test_parent_manifest_rejects_authority_promotion(
    field: str,
) -> None:
    payload = _valid_manifest()
    payload[field] = True

    with pytest.raises(
        RuntimeError,
        match="authority contract violated",
    ):
        _validate_open_world_discovery_parent_manifest(
            payload,
            expected_provider_plan_sha256="a" * 64,
        )


def test_parent_manifest_rejects_zero_external_axes() -> None:
    payload = _valid_manifest()
    payload["external_axis_count"] = 0

    with pytest.raises(
        RuntimeError,
        match="zero validated external axes",
    ):
        _validate_open_world_discovery_parent_manifest(
            payload,
            expected_provider_plan_sha256="a" * 64,
        )
