from __future__ import annotations

import sys

import pytest

from scripts.discovery import run_external_novelty


def _argv(*extra: str) -> list[str]:
    return [
        "run_external_novelty",
        "--portfolio",
        "unused.portfolio.json",
        "--domain-profile",
        "unused-domain",
        "--model",
        "unused-model",
        "--output-prefix",
        "unused-output",
        *extra,
    ]


def test_downstream_gate_requires_pre_review_shadow(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("--downstream-gate-shadow"),
    )

    with pytest.raises(
        ValueError,
        match=(
            "--downstream-gate-shadow requires "
            "--pre-review-coverage-shadow"
        ),
    ):
        run_external_novelty.main()


def test_downstream_gate_rejects_coverage_only_mode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            "--pre-review-coverage-shadow",
            "--pre-review-coverage-only",
            "--downstream-gate-shadow",
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "--downstream-gate-shadow cannot be combined"
        ),
    ):
        run_external_novelty.main()


def test_parse_args_accepts_shadow_gate_contract(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            "--pre-review-coverage-shadow",
            "--downstream-gate-shadow",
        ),
    )

    args = run_external_novelty.parse_args()

    assert args.pre_review_coverage_shadow is True
    assert args.pre_review_coverage_only is False
    assert args.downstream_gate_shadow is True
