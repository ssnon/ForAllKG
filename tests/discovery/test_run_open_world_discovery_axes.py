from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.discovery.run_open_world_discovery_axes import (
    _provider_request,
    _stage_manifest,
    build_parser,
)


def test_provider_request_auto_and_explicit() -> None:
    assert _provider_request("auto") is None
    assert _provider_request("") is None
    assert _provider_request(
        "openalex,crossref"
    ) == [
        "openalex",
        "crossref",
    ]


def test_provider_request_rejects_empty_explicit_surface() -> None:
    with pytest.raises(
        argparse.ArgumentTypeError,
    ):
        _provider_request(" , ")


def test_cli_defaults_pin_validated_s17_bounds() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dual-context",
            "dual.json",
            "--control-axis-plan",
            "control.json",
            "--output-prefix",
            "out/run",
        ]
    )

    assert args.providers == "auto"
    assert args.results_per_query == 20
    assert args.max_selected_works == 12
    assert args.max_control_similarity == pytest.approx(
        0.85
    )
    assert args.max_source_span_words == 40


def test_manifest_keeps_stage_non_authoritative() -> None:
    dual = SimpleNamespace(
        dual_context_id="dual:1",
        dual_context_sha256="a" * 64,
    )
    control = SimpleNamespace(
        plan_id="plan:control",
        plan_sha256="b" * 64,
    )
    provider_plan = SimpleNamespace(
        plan_id="providers:1",
        plan_sha256="c" * 64,
        mode="STANDARD_2_PROVIDER",
        active_providers=[
            "openalex",
            "crossref",
        ],
    )
    external_plan = SimpleNamespace(
        plan_id="plan:external",
        plan_sha256="d" * 64,
        axes=[
            SimpleNamespace(axis_id="axis:ext")
        ],
    )
    outcome = SimpleNamespace(
        retrieval=SimpleNamespace(
            complete=True,
            seeds=[1, 2],
            executions=[1, 2, 3, 4],
        ),
        selected_works=(1, 2, 3),
        generation=SimpleNamespace(
            draft=SimpleNamespace(
                axes=[1, 2]
            )
        ),
        validation=SimpleNamespace(
            accepted_axes=[1],
            rejected_axes=[2],
        ),
        axis_plan=SimpleNamespace(
            plan=external_plan,
            bundle=SimpleNamespace(
                bundle_id="bundle:external",
                bundle_sha256="e" * 64,
            ),
            rejected_axes=[1],
        ),
    )

    manifest = _stage_manifest(
        dual=dual,
        control_plan=control,
        provider_plan=provider_plan,
        outcome=outcome,
        index_dir=Path("/tmp/index"),
    )

    assert manifest[
        "external_literature_authority"
    ] == "INSPIRATION_ONLY"
    assert (
        manifest[
            "positive_premise_authority_changed"
        ]
        is False
    )
    assert (
        manifest["novelty_authority_created"]
        is False
    )
    assert (
        manifest[
            "hypothesis_generation_performed"
        ]
        is False
    )
    assert (
        manifest[
            "canonical_fallback_authorized"
        ]
        is False
    )
    assert manifest["external_axis_count"] == 1
