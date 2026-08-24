from types import SimpleNamespace

import pytest

from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyRefinementReport,
    RefinementAttempt,
)
from pipeline_core.discovery.novelty_refinement_runtime import (
    TargetedNoveltyRefinementRuntime,
)


def _attempt(
    *,
    original: str,
    candidate: str | None,
    decision: str,
    action: str,
) -> RefinementAttempt:
    return RefinementAttempt(
        original_hypothesis_id=original,
        candidate_hypothesis_id=candidate,
        gap_id=f"gap:{original}",
        action=action,
        decision=decision,
        original_external_status=(
            "NEW_COMBINATION_OF_KNOWN_EFFECTS"
        ),
        grounding_preserved=True,
        interpretation="test",
    )


def test_final_ids_bind_only_after_final_portfolio_compile():
    attempts = [
        _attempt(
            original="hypothesis:original-h1",
            candidate="hypothesis:original-h1",
            decision="kept_original",
            action="keep",
        ),
        _attempt(
            original="hypothesis:original-rejected",
            candidate="hypothesis:candidate-rejected",
            decision="external_novelty_rejected",
            action="targeted_search_then_refine",
        ),
        _attempt(
            original="hypothesis:original-h2",
            candidate="hypothesis:candidate-refined-h2",
            decision="accepted_refinement",
            action="targeted_search_then_refine",
        ),
    ]

    final_portfolio = SimpleNamespace(
        hypotheses=[
            SimpleNamespace(
                hypothesis_id="hypothesis:final-h1"
            ),
            SimpleNamespace(
                hypothesis_id="hypothesis:final-h2"
            ),
        ]
    )

    bound = (
        TargetedNoveltyRefinementRuntime
        ._bind_final_hypothesis_ids(
            attempts,
            final_portfolio,
        )
    )

    assert (
        bound[0].candidate_hypothesis_id
        == "hypothesis:original-h1"
    )
    assert (
        bound[0].final_hypothesis_id
        == "hypothesis:final-h1"
    )

    assert (
        bound[1].candidate_hypothesis_id
        == "hypothesis:candidate-rejected"
    )
    assert (
        bound[1].final_hypothesis_id
        is None
    )

    assert (
        bound[2].candidate_hypothesis_id
        == "hypothesis:candidate-refined-h2"
    )
    assert (
        bound[2].final_hypothesis_id
        == "hypothesis:final-h2"
    )


def test_final_id_binding_rejects_survivor_cardinality_drift():
    attempts = [
        _attempt(
            original="hypothesis:h1",
            candidate="hypothesis:h1",
            decision="kept_original",
            action="keep",
        ),
    ]

    final_portfolio = SimpleNamespace(
        hypotheses=[]
    )

    with pytest.raises(
        RuntimeError,
        match="cardinality mismatch",
    ):
        (
            TargetedNoveltyRefinementRuntime
            ._bind_final_hypothesis_ids(
                attempts,
                final_portfolio,
            )
        )


def test_v2_report_requires_true_final_membership_ids():
    attempts = (
        TargetedNoveltyRefinementRuntime
        ._bind_final_hypothesis_ids(
            [
                _attempt(
                    original="hypothesis:h1",
                    candidate="hypothesis:h1",
                    decision="kept_original",
                    action="keep",
                ),
            ],
            SimpleNamespace(
                hypotheses=[
                    SimpleNamespace(
                        hypothesis_id="hypothesis:final-h1"
                    ),
                ]
            ),
        )
    )

    report = NoveltyRefinementReport(
        report_id="report:test",
        report_sha256="sha:test",
        source_portfolio_id="portfolio:source",
        source_external_report_id="external:test",
        source_gap_plan_id="gap-plan:test",
        final_portfolio_id="portfolio:final",
        attempts=attempts,
        accepted_refinement_count=0,
        kept_original_count=1,
        rejected_count=0,
    )

    assert (
        report.schema_version
        == "novelty-refinement-report-v2"
    )
    assert (
        report.attempts[0].final_hypothesis_id
        == "hypothesis:final-h1"
    )


def test_v1_historical_report_remains_parseable_without_reinterpretation():
    legacy = RefinementAttempt(
        original_hypothesis_id="hypothesis:legacy",
        # Historical v1 used this field for the attempt-stage identity.
        final_hypothesis_id="hypothesis:legacy",
        gap_id="gap:legacy",
        action="keep",
        decision="kept_original",
        original_external_status=(
            "NEW_COMBINATION_OF_KNOWN_EFFECTS"
        ),
        grounding_preserved=True,
        interpretation="historical",
    )

    report = NoveltyRefinementReport(
        schema_version="novelty-refinement-report-v1",
        report_id="report:legacy",
        report_sha256="sha:legacy",
        source_portfolio_id="portfolio:legacy-source",
        source_external_report_id="external:legacy",
        source_gap_plan_id="gap-plan:legacy",
        final_portfolio_id="portfolio:legacy-final",
        attempts=[legacy],
        accepted_refinement_count=0,
        kept_original_count=1,
        rejected_count=0,
    )

    assert (
        report.schema_version
        == "novelty-refinement-report-v1"
    )
    assert (
        report.attempts[0].candidate_hypothesis_id
        is None
    )
    assert (
        report.attempts[0].final_hypothesis_id
        == "hypothesis:legacy"
    )


def _portfolio_shell(
    portfolio_id: str,
    hypothesis_ids: list[str],
):
    from pipeline_core.discovery.hypothesis_contracts import (
        HypothesisPortfolio,
    )

    return HypothesisPortfolio.model_construct(
        portfolio_id=portfolio_id,
        hypotheses=[
            SimpleNamespace(
                hypothesis_id=value
            )
            for value in hypothesis_ids
        ],
    )


def test_v2_artifact_binds_report_ids_to_actual_final_portfolio():
    from pipeline_core.discovery.novelty_refinement_contracts import (
        NoveltyRefinementArtifact,
    )

    attempts = (
        TargetedNoveltyRefinementRuntime
        ._bind_final_hypothesis_ids(
            [
                _attempt(
                    original="hypothesis:original",
                    candidate="hypothesis:original",
                    decision="kept_original",
                    action="keep",
                ),
            ],
            SimpleNamespace(
                hypotheses=[
                    SimpleNamespace(
                        hypothesis_id="hypothesis:final"
                    ),
                ]
            ),
        )
    )

    report = NoveltyRefinementReport(
        report_id="report:artifact-test",
        report_sha256="sha:artifact-test",
        source_portfolio_id="portfolio:source",
        source_external_report_id="external:test",
        source_gap_plan_id="gap-plan:test",
        final_portfolio_id="portfolio:final",
        attempts=attempts,
        accepted_refinement_count=0,
        kept_original_count=1,
        rejected_count=0,
    )

    artifact = NoveltyRefinementArtifact(
        portfolio=_portfolio_shell(
            "portfolio:final",
            ["hypothesis:final"],
        ),
        report=report,
    )

    assert (
        artifact.report.attempts[0]
        .final_hypothesis_id
        == "hypothesis:final"
    )


def test_v2_artifact_rejects_report_final_id_not_in_portfolio():
    from pydantic import ValidationError

    from pipeline_core.discovery.novelty_refinement_contracts import (
        NoveltyRefinementArtifact,
    )

    attempts = (
        TargetedNoveltyRefinementRuntime
        ._bind_final_hypothesis_ids(
            [
                _attempt(
                    original="hypothesis:original",
                    candidate="hypothesis:original",
                    decision="kept_original",
                    action="keep",
                ),
            ],
            SimpleNamespace(
                hypotheses=[
                    SimpleNamespace(
                        hypothesis_id="hypothesis:reported-final"
                    ),
                ]
            ),
        )
    )

    report = NoveltyRefinementReport(
        report_id="report:mismatch",
        report_sha256="sha:mismatch",
        source_portfolio_id="portfolio:source",
        source_external_report_id="external:test",
        source_gap_plan_id="gap-plan:test",
        final_portfolio_id="portfolio:final",
        attempts=attempts,
        accepted_refinement_count=0,
        kept_original_count=1,
        rejected_count=0,
    )

    with pytest.raises(
        ValidationError,
        match=(
            "final_hypothesis_id set does not equal "
            "final portfolio"
        ),
    ):
        NoveltyRefinementArtifact(
            portfolio=_portfolio_shell(
                "portfolio:final",
                ["hypothesis:actual-final"],
            ),
            report=report,
        )
