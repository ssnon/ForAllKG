from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

from domains.feasibility_registry import (
    get_feasibility_adapter,
)
import scripts.discovery.run_feasibility_e2e as feasibility_runner


def test_feasibility_registry_requires_explicit_profile_argument():
    parameter = inspect.signature(
        get_feasibility_adapter
    ).parameters["profile_id"]

    assert parameter.default is inspect.Parameter.empty


def test_feasibility_registry_rejects_empty_profile_identity():
    with pytest.raises(
        ValueError,
        match="profile_id must be explicit",
    ):
        get_feasibility_adapter("")  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="profile_id must be explicit",
    ):
        get_feasibility_adapter(None)  # type: ignore[arg-type]


def test_generic_feasibility_cli_requires_domain_profile(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_feasibility_e2e",
            "--context",
            "context.json",
            "--portfolio",
            "portfolio.json",
            "--semantic-review",
            "semantic.json",
            "--output-dir",
            "out",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        feasibility_runner.main()

    assert exc_info.value.code == 2


def test_generic_feasibility_runner_has_no_dac_domain_default():
    source = Path(
        "scripts/discovery/run_feasibility_e2e.py"
    ).read_text(encoding="utf-8")

    assert 'default="dac_her"' not in source
