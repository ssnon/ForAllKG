from __future__ import annotations

import inspect

from domains.sers.context_compiler import (
    SERSContextCompilationError,
)
from domains.sers.context_review_adapter import (
    SERSDiscoveryAxisContextReviewer,
)
from pipeline_core.discovery.discovery_axis_context_runtime import (
    AxisContextReviewUnavailableError,
)
from pipeline_core.discovery.discovery_axis_contracts import (
    AxisAttemptRecord,
)
from pipeline_core.discovery.discovery_axis_runtime import (
    DiscoveryAxisSynthesisRuntime,
)


def test_context_rejected_is_valid_axis_attempt_decision() -> None:
    row = AxisAttemptRecord(
        axis_id="axis:test",
        stage="initial",
        generation_index=1,
        decision="context_rejected",
        hypothesis_id="hypothesis:test",
        title="test",
        repair_reason="claim-local context unavailable",
    )

    assert row.decision == "context_rejected"


def test_generic_context_failure_is_domain_neutral() -> None:
    exc = AxisContextReviewUnavailableError(
        "context unavailable"
    )

    assert isinstance(exc, RuntimeError)
    assert "SERS" not in type(exc).__name__


def test_sers_reviewer_translates_strict_compiler_failure() -> None:
    source = inspect.getsource(
        SERSDiscoveryAxisContextReviewer.review
    )

    assert "except SERSContextCompilationError as exc:" in source
    assert "raise AxisContextReviewUnavailableError(" in source

    # The strict domain compiler exception still exists and remains the
    # producer-side fail-closed signal.
    assert issubclass(
        SERSContextCompilationError,
        ValueError,
    )


def test_runtime_isolates_context_failure_at_both_review_sites() -> None:
    source = inspect.getsource(
        DiscoveryAxisSynthesisRuntime.run
    )

    assert (
        source.count(
            "except AxisContextReviewUnavailableError as exc:"
        )
        == 2
    )

    assert (
        source.count(
            'decision="context_rejected"'
        )
        == 2
    )

    assert (
        source.count(
            "continue"
        )
        >= 2
    )


def test_runtime_does_not_catch_all_context_review_errors() -> None:
    source = inspect.getsource(
        DiscoveryAxisSynthesisRuntime.run
    )

    assert "except Exception" not in source
    assert "except BaseException" not in source



# ----------------------------------------------------------------------
# Behavioral regression:
# a context-compilation failure on the middle axis must not terminate
# the per-axis synthesis loop.
# ----------------------------------------------------------------------

from dataclasses import dataclass
from types import SimpleNamespace

import pipeline_core.discovery.discovery_axis_runtime as axis_runtime_module


@dataclass(frozen=True)
class _BehaviorAxis:
    axis_id: str
    axis_rank: int
    inspiration_id: str
    candidate_unit_id: str


class _BehaviorPromptAssembler:
    def __init__(
        self,
        axis,
        *,
        family_hierarchy=None,
    ):
        self.axis = axis
        self.family_hierarchy = family_hierarchy


@dataclass(frozen=True)
class _BehaviorPromptRecord:
    axis_id: str
    axis_rank: int
    prompt: object


class _BehaviorMakerRuntime:
    def __init__(
        self,
        draft_backend,
        *,
        prompt_assembler,
        compiler,
        validator,
        max_repairs,
    ):
        self.prompt_assembler = prompt_assembler

    def run(self, context):
        axis = self.prompt_assembler.axis

        proposal = SimpleNamespace(
            hypothesis_id=(
                f"hypothesis:{axis.axis_id}"
            ),
            title=f"title:{axis.axis_id}",
        )

        draft = SimpleNamespace(
            hypotheses=[proposal],
        )

        return SimpleNamespace(
            prompt=SimpleNamespace(
                axis_id=axis.axis_id,
            ),
            final_draft=draft,
            draft_history=(draft,),
            compile_issues=(),
        )


class _BehaviorCompiler:
    def compile(
        self,
        context,
        draft,
    ):
        # Aggregate finalization reaches this with zero accepted
        # hypotheses. Per-axis compilation is overridden below.
        assert list(draft.hypotheses) == []

        return SimpleNamespace(
            portfolio_id="portfolio:final-empty",
            hypotheses=[],
        )


class _BehaviorValidator:
    def validate(
        self,
        context,
        portfolio,
    ):
        return SimpleNamespace(
            passes=True,
            issues=[],
        )


class _BehaviorFidelityCritic:
    def review(
        self,
        axis,
        card,
        encoder,
    ):
        return SimpleNamespace(
            status="pass",
        )


class _BehaviorNoveltyAssessor:
    def assess(
        self,
        dual,
        portfolio,
        mapper,
    ):
        cards = []

        for card in portfolio.hypotheses:
            cards.append(
                SimpleNamespace(
                    hypothesis_id=card.hypothesis_id,
                    status=(
                        "reconstructs_existing_corpus_claim"
                    ),
                    interpretation=(
                        "forced novelty rejection for "
                        "axis-isolation regression"
                    ),
                )
            )

        return SimpleNamespace(
            cards=cards,
        )


class _BehaviorContextReviewer:
    def __init__(self):
        self.calls = []

    def review(
        self,
        *,
        dual,
        axis,
        card,
    ):
        self.calls.append(
            axis.axis_id
        )

        if axis.axis_id == "axis:B":
            raise AxisContextReviewUnavailableError(
                "synthetic claim-local context unavailable"
            )

        return SimpleNamespace(
            review_id=(
                f"context-review:{axis.axis_id}"
            ),
            hypothesis_id=card.hypothesis_id,
            status="pass",
        )


class _BehaviorMapper:
    encoder = object()


class _BehaviorRuntime(
    DiscoveryAxisSynthesisRuntime
):
    def _compile_validate(
        self,
        context,
        draft,
    ):
        assert len(
            draft.hypotheses
        ) == 1

        proposal = draft.hypotheses[0]

        card = SimpleNamespace(
            hypothesis_id=proposal.hypothesis_id,
            title=proposal.title,
        )

        portfolio = SimpleNamespace(
            portfolio_id=(
                f"portfolio:{proposal.hypothesis_id}"
            ),
            hypotheses=[card],
        )

        return (
            portfolio,
            [],
            [],
        )


def test_context_failure_is_axis_local_and_later_axes_continue(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        axis_runtime_module,
        "DiscoveryAxisHypothesisPromptAssembler",
        _BehaviorPromptAssembler,
    )

    monkeypatch.setattr(
        axis_runtime_module,
        "HypothesisMakerAgentRuntime",
        _BehaviorMakerRuntime,
    )

    monkeypatch.setattr(
        axis_runtime_module,
        "AxisPromptRecord",
        _BehaviorPromptRecord,
    )

    # The hash helper normally serializes Pydantic models.
    # This harness intentionally uses minimal runtime-only objects.
    monkeypatch.setattr(
        axis_runtime_module,
        "_sha256_json",
        lambda value: "a" * 64,
    )

    axes = [
        _BehaviorAxis(
            axis_id="axis:A",
            axis_rank=1,
            inspiration_id="inspiration:A",
            candidate_unit_id="candidate:A",
        ),
        _BehaviorAxis(
            axis_id="axis:B",
            axis_rank=2,
            inspiration_id="inspiration:B",
            candidate_unit_id="candidate:B",
        ),
        _BehaviorAxis(
            axis_id="axis:C",
            axis_rank=3,
            inspiration_id="inspiration:C",
            candidate_unit_id="candidate:C",
        ),
    ]

    dual = SimpleNamespace(
        dual_context_id="dual:test",
        dual_context_sha256="b" * 64,
        grounded_context=SimpleNamespace(),
    )

    plan = SimpleNamespace(
        plan_id="plan:test",
        plan_sha256="c" * 64,
        source_dual_context_id=(
            dual.dual_context_id
        ),
        source_dual_context_sha256=(
            dual.dual_context_sha256
        ),
        axes=axes,
    )

    reviewer = _BehaviorContextReviewer()

    runtime = _BehaviorRuntime(
        draft_backend=SimpleNamespace(),
        mapper=_BehaviorMapper(),
        compiler=_BehaviorCompiler(),
        validator=_BehaviorValidator(),
        fidelity_critic=(
            _BehaviorFidelityCritic()
        ),
        inference_critic=None,
        context_reviewer=reviewer,
        novelty_assessor=(
            _BehaviorNoveltyAssessor()
        ),
        max_compile_repairs=0,
        max_fidelity_repairs=0,
        max_inference_repairs=0,
        max_novelty_repairs=0,
    )

    outcome = runtime.run(
        dual,
        plan,
    )

    # Strongest behavioral assertion:
    # the reviewer was reached for C after B failed.
    assert reviewer.calls == [
        "axis:A",
        "axis:B",
        "axis:C",
    ]

    decisions = {
        (
            row.axis_id,
            row.decision,
        )
        for row in outcome.report.attempts
    }

    assert (
        "axis:B",
        "context_rejected",
    ) in decisions

    assert (
        "axis:A",
        "novelty_rejected",
    ) in decisions

    assert (
        "axis:C",
        "novelty_rejected",
    ) in decisions

    assert (
        outcome.report.attempted_axis_count
        == 3
    )

    assert (
        outcome.report.accepted_hypothesis_count
        == 0
    )

    # Successful reviews only; failed B review never enters history.
    assert [
        row.axis_id
        for row
        in outcome.context_review_history
    ] == [
        "axis:A",
        "axis:C",
    ]
