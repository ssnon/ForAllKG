from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.hypothesis_semantic_contracts import (
    HypothesisSemanticDimensionDraft,
    HypothesisSemanticReviewDraft,
    SEMANTIC_DIMENSIONS,
)
from pipeline_core.discovery.hypothesis_semantic_llm import (
    HypothesisSemanticGeneration,
)
from pipeline_core.discovery.hypothesis_semantic_runtime import (
    HypothesisSemanticCriticRuntime,
)
from pipeline_core.discovery.pre_n10_regeneration_reentry_v2 import (
    execute_pre_n10_regeneration_reentry_v2,
)
from tests.discovery.test_pre_n10_regeneration_reentry_v1_s155 import (
    _DecompositionBackend,
    _SemanticBackend,
    _regenerate,
)


def _portfolio(regeneration) -> HypothesisPortfolio:
    path = Path(
        regeneration.lineages[0].regenerated_portfolio_path
    )
    return HypothesisPortfolio.model_validate_json(
        path.read_text(encoding="utf-8")
    )


class _WarningSemanticBackend:
    backend_name = "semantic-warning"
    model_name = "semantic-warning"

    def __init__(self, portfolio: HypothesisPortfolio):
        self.portfolio = portfolio
        self.calls = 0

    def review(self, prompt):
        self.calls += 1
        hid = self.portfolio.hypotheses[0].hypothesis_id
        rows = []
        for dimension in SEMANTIC_DIMENSIONS:
            if dimension == "hypothesis_distinctness":
                verdict = "not_applicable"
            elif dimension == "causal_strengthening":
                verdict = "warning"
            else:
                verdict = "pass"
            rows.append(
                HypothesisSemanticDimensionDraft(
                    dimension=dimension,
                    verdict=verdict,
                    rationale="fixture",
                    hypothesis_ids=(
                        []
                        if dimension == "abstention_appropriateness"
                        else [hid]
                    ),
                    statement_ids=(
                        ["statement:1"]
                        if dimension == "premise_fidelity"
                        else []
                    ),
                )
            )
        return HypothesisSemanticGeneration(
            draft=HypothesisSemanticReviewDraft(
                dimensions=rows,
                overall_summary="warning fixture",
            ),
            input_tokens=1,
            output_tokens=1,
        )


def test_reentry_v2_semantic_pass_reaches_pre_n10_contract(
    tmp_path: Path,
):
    context, regeneration = _regenerate(tmp_path)
    portfolio = _portfolio(regeneration)
    semantic_backend = _SemanticBackend(portfolio, accept=True)
    decomposition_backend = _DecompositionBackend(malformed=False)

    report, _ = execute_pre_n10_regeneration_reentry_v2(
        context=context,
        regeneration_report=regeneration,
        semantic_runner_factory=lambda _source, _dir: (
            HypothesisSemanticCriticRuntime(semantic_backend)
        ),
        decomposition_backend_factory=lambda _source, _dir: (
            decomposition_backend
        ),
        output_root=tmp_path / "reentry-v2",
    )

    assert semantic_backend.calls == 1
    assert decomposition_backend.calls == 1
    assert report.semantic_review_valid_count == 1
    assert report.semantic_admissible_count == 1
    assert report.semantic_intervention_required_count == 0
    assert report.claim_decomposition_request_count == 1
    assert report.ready_for_n10_count == 1
    row = report.lineages[0]
    assert row.semantic_status == "SEMANTIC_ADMISSIBLE"
    assert row.semantic_disposition == "PASS"
    assert row.semantic_admissible_for_pre_n10 is True
    assert row.semantic_failed_dimensions == []
    assert row.final_status == "PRE_N10_READY"
    assert row.semantic_disposition_path is not None
    assert Path(row.semantic_disposition_path).is_file()
    assert report.external_novelty_performed is False
    assert report.n10_performed is False


def test_reentry_v2_semantic_fail_is_valid_review_but_blocks_pre_n10(
    tmp_path: Path,
):
    context, regeneration = _regenerate(tmp_path)
    portfolio = _portfolio(regeneration)
    semantic_backend = _SemanticBackend(portfolio, accept=False)

    class _ShouldNotDecompose:
        def decompose(self, hypothesis, *, max_claims):
            raise AssertionError(
                "semantic fail must block claim decomposition"
            )

    report, _ = execute_pre_n10_regeneration_reentry_v2(
        context=context,
        regeneration_report=regeneration,
        semantic_runner_factory=lambda _source, _dir: (
            HypothesisSemanticCriticRuntime(semantic_backend)
        ),
        decomposition_backend_factory=lambda _source, _dir: (
            _ShouldNotDecompose()
        ),
        output_root=tmp_path / "reentry-v2",
    )

    assert semantic_backend.calls == 1
    assert report.semantic_review_valid_count == 1
    assert report.semantic_admissible_count == 0
    assert report.semantic_intervention_required_count == 1
    assert report.claim_decomposition_request_count == 0
    assert report.ready_for_n10_count == 0
    row = report.lineages[0]
    assert row.semantic_review_valid is True
    assert row.semantic_admissible_for_pre_n10 is False
    assert row.semantic_disposition == "REQUIRES_SEMANTIC_INTERVENTION"
    assert "premise_fidelity" in row.semantic_failed_dimensions
    assert row.final_status == "SEMANTIC_INTERVENTION_REQUIRED"
    assert row.ready_for_n10 is False
    assert report.semantic_disposition_is_not_final_rejection_authority is True
    assert report.semantic_disposition_is_not_novelty_authority is True
    assert report.n10_performed is False


def test_reentry_v2_semantic_warning_does_not_block_pre_n10(
    tmp_path: Path,
):
    context, regeneration = _regenerate(tmp_path)
    portfolio = _portfolio(regeneration)
    semantic_backend = _WarningSemanticBackend(portfolio)
    decomposition_backend = _DecompositionBackend(malformed=False)

    report, _ = execute_pre_n10_regeneration_reentry_v2(
        context=context,
        regeneration_report=regeneration,
        semantic_runner_factory=lambda _source, _dir: (
            HypothesisSemanticCriticRuntime(semantic_backend)
        ),
        decomposition_backend_factory=lambda _source, _dir: (
            decomposition_backend
        ),
        output_root=tmp_path / "reentry-v2",
    )

    assert semantic_backend.calls == 1
    assert decomposition_backend.calls == 1
    assert report.semantic_review_valid_count == 1
    assert report.semantic_admissible_count == 1
    assert report.semantic_intervention_required_count == 0
    row = report.lineages[0]
    assert row.semantic_disposition == "PASS"
    assert row.semantic_failed_dimensions == []
    assert row.semantic_warning_dimensions == ["causal_strengthening"]
    assert row.final_status == "PRE_N10_READY"
    assert report.semantic_warning_blocks_pre_n10 is False
