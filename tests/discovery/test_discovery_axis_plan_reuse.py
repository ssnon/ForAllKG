from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.discovery.run_discovery_axis_hypothesis_maker import (
    _resolve_axis_plan,
)


def _plan_payload(
    *,
    dual_id: str = "dual:test",
    dual_sha: str = "a" * 64,
):
    return {
        "schema_version":
            "discovery-axis-plan-v1",

        "plan_id":
            "discovery_axis_plan:test",

        "plan_sha256":
            "b" * 64,

        "source_dual_context_id":
            dual_id,

        "source_dual_context_sha256":
            dual_sha,

        "source_bundle_id":
            "discovery_bundle:test",

        "source_bundle_sha256":
            "c" * 64,

        "corpus_id":
            "test-corpus",

        "axes":
            [],

        "excluded_inspiration_ids":
            [],

        "policy":
            {
                "policy_version":
                    "discovery-axis-planner-v1",

                "max_axes":
                    5,

                "require_candidate_unit":
                    True,

                "min_exploration_score":
                    0.05,

                "min_candidate_unit_score":
                    0.30,

                "max_reaction_domain_switch_penalty":
                    0.50,
            },
    }


def _args(axis_plan_input):
    return SimpleNamespace(
        axis_plan_input=axis_plan_input,

        max_axes=5,

        allow_non_candidate_axes=False,

        min_exploration_score=0.05,

        min_candidate_unit_score=0.30,

        max_reaction_switch_penalty=0.50,
    )


def _dual(
    *,
    dual_id="dual:test",
    dual_sha="a" * 64,
):
    return SimpleNamespace(
        dual_context_id=dual_id,
        dual_context_sha256=dual_sha,
    )


def test_reuses_frozen_plan_verbatim(
    tmp_path,
):
    path = (
        tmp_path
        / "axis_plan.json"
    )

    payload = _plan_payload()

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    plan, mode = _resolve_axis_plan(
        args=_args(path),
        dual=_dual(),
    )

    assert mode == (
        "reused_frozen_plan"
    )

    assert plan.plan_id == (
        payload["plan_id"]
    )

    assert (
        plan.source_dual_context_id
        == "dual:test"
    )

    assert (
        plan.source_dual_context_sha256
        == "a" * 64
    )


def test_rejects_frozen_plan_from_other_dual_id(
    tmp_path,
):
    path = (
        tmp_path
        / "axis_plan.json"
    )

    path.write_text(
        json.dumps(
            _plan_payload(
                dual_id="dual:other"
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="dual_context_id mismatch",
    ):
        _resolve_axis_plan(
            args=_args(path),
            dual=_dual(),
        )


def test_rejects_frozen_plan_from_other_dual_sha(
    tmp_path,
):
    path = (
        tmp_path
        / "axis_plan.json"
    )

    path.write_text(
        json.dumps(
            _plan_payload(
                dual_sha="d" * 64
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="dual_context_sha256 mismatch",
    ):
        _resolve_axis_plan(
            args=_args(path),
            dual=_dual(),
        )
