from __future__ import annotations

from types import SimpleNamespace

import pytest

from domains.sers.context_compiler import (
    SERSContextCompilationError,
)
from domains.sers.context_review_adapter import (
    SERSDiscoveryAxisContextReviewer,
)
from pipeline_core.discovery.discovery_axis_context_runtime import (
    AxisContextReviewUnavailableError,
)


class _FailingGroundedCompiler:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def compile_grounded_statement(self, statement):
        self.calls.append(statement.statement_id)
        raise SERSContextCompilationError(
            "claim-local closure produced no SERS context facts for "
            + statement.statement_id
        )


class _AxisCompiler:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def compile_axis_inspiration(self, inspiration):
        self.calls.append(inspiration.inspiration_id)
        return SimpleNamespace(signature_id="axis:test")


class _ForbiddenInterpreter:
    def interpret(self, **_kwargs):
        raise AssertionError(
            "interpreter must not run after unavailable grounded context"
        )


class _ForbiddenComparator:
    def compare(self, **_kwargs):
        raise AssertionError(
            "comparator must not run after unavailable grounded context"
        )


def test_grounded_context_compilation_failure_is_axis_local_unavailable() -> None:
    grounded = _FailingGroundedCompiler()
    axis = _AxisCompiler()

    reviewer = SERSDiscoveryAxisContextReviewer(
        grounded_compiler=grounded,
        axis_compiler=axis,
        interpreter=_ForbiddenInterpreter(),
        comparator=_ForbiddenComparator(),
    )

    dual = SimpleNamespace(
        grounded_context=SimpleNamespace(
            domain_profile_id="sers_au_ag",
            evidence_statements=[
                SimpleNamespace(statement_id="stmt:no_context"),
            ],
        ),
        discovery_bundle=SimpleNamespace(
            inspirations=[
                SimpleNamespace(inspiration_id="insp:selected"),
            ],
        ),
        task_lane_inspirations=[],
    )
    card = SimpleNamespace(
        hypothesis_id="hypothesis:h1",
        premise_statement_ids=["stmt:no_context"],
    )
    discovery_axis = SimpleNamespace(
        axis_id="axis:1",
        inspiration_id="insp:selected",
    )

    with pytest.raises(
        AxisContextReviewUnavailableError,
        match=(
            "selected grounded premise cannot produce "
            "claim-local SERS scientific context"
        ),
    ) as exc:
        reviewer.review(
            dual=dual,
            axis=discovery_axis,
            card=card,
        )

    assert "stmt:no_context" in str(exc.value)
    assert grounded.calls == ["stmt:no_context"]
    assert axis.calls == []
