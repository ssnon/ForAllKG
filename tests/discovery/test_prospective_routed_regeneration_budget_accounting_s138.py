from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.prospective_routed_regeneration_budget_accounting import (
    _is_full_e2e_argv,
)


def test_full_e2e_module_is_detected() -> None:
    assert _is_full_e2e_argv(
        [
            "python",
            "-m",
            "scripts.discovery.run_dac_discovery_e2e",
            "--run-dir",
            "/tmp/x",
        ]
    ) is True


def test_non_e2e_module_is_not_detected() -> None:
    assert _is_full_e2e_argv(
        [
            "python",
            "-m",
            "scripts.discovery.run_preverifier_contract_gate_v2",
        ]
    ) is False


def test_empty_argv_is_not_full_e2e() -> None:
    assert _is_full_e2e_argv(None) is False
    assert _is_full_e2e_argv([]) is False


def test_module_name_must_match_exactly() -> None:
    assert _is_full_e2e_argv(
        [
            "python",
            "-m",
            "scripts.discovery.run_dac_discovery_e2e_extra",
        ]
    ) is False


def test_full_e2e_detection_is_independent_of_run_dir_position() -> None:
    assert _is_full_e2e_argv(
        [
            "python",
            "--some-python-flag",
            "-m",
            "scripts.discovery.run_dac_discovery_e2e",
            "--source",
            "x",
        ]
    ) is True


def test_plain_script_path_is_not_silently_treated_as_module() -> None:
    assert _is_full_e2e_argv(
        [
            "python",
            "scripts/discovery/run_dac_discovery_e2e.py",
        ]
    ) is False
