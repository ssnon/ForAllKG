from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.discovery.run_discovery_axis_hypothesis_maker import (
    _require_axis_source_availability,
)


def _dual(*, inspiration_count: int):
    return SimpleNamespace(
        discovery_bundle=SimpleNamespace(
            inspirations=[
                object()
                for _ in range(inspiration_count)
            ]
        )
    )


def test_explicit_frozen_axis_plan_does_not_require_kg_inspiration() -> None:
    args = SimpleNamespace(
        axis_plan_input="external_axis_plan.json"
    )

    _require_axis_source_availability(
        args=args,
        dual=_dual(inspiration_count=0),
    )


def test_planning_from_dual_still_refuses_empty_inspiration_bundle() -> None:
    args = SimpleNamespace(
        axis_plan_input=None
    )

    with pytest.raises(
        SystemExit,
        match="refuses to collapse back to canonical synthesis",
    ):
        _require_axis_source_availability(
            args=args,
            dual=_dual(inspiration_count=0),
        )


def test_planning_from_dual_accepts_existing_inspiration() -> None:
    args = SimpleNamespace(
        axis_plan_input=None
    )

    _require_axis_source_availability(
        args=args,
        dual=_dual(inspiration_count=1),
    )


def test_explicit_plan_seam_does_not_mutate_discovery_bundle() -> None:
    args = SimpleNamespace(
        axis_plan_input="external_axis_plan.json"
    )
    dual = _dual(inspiration_count=0)

    before = list(
        dual.discovery_bundle.inspirations
    )
    _require_axis_source_availability(
        args=args,
        dual=dual,
    )
    assert dual.discovery_bundle.inspirations == before
