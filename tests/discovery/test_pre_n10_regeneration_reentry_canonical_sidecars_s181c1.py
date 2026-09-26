from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.atomic_scientific_source_provenance import (
    AtomicScientificSourceBindingBundle,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.hypothesis_semantic_runtime import (
    HypothesisSemanticCriticRuntime,
)
from pipeline_core.discovery.pre_n10_canonical_source_reference_v1 import (
    PreN10CanonicalSourceReferenceReportV1,
)
from pipeline_core.discovery.pre_n10_regeneration_reentry_v2 import (
    execute_pre_n10_regeneration_reentry_v2,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v2 import (
    PreN10ScientificContractReportV2,
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


def test_reentry_fresh_decomposition_materializes_new_canonical_chain(
    tmp_path: Path,
) -> None:
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

    row = report.lineages[0]
    assert row.final_status == "PRE_N10_READY"

    bundle_path = Path(str(row.source_binding_bundle_path))
    canonical_path = Path(str(row.canonical_source_reference_path))
    v2_path = Path(str(row.contract_v2_report_path))

    assert bundle_path.is_file()
    assert canonical_path.is_file()
    assert v2_path.is_file()

    bundle = AtomicScientificSourceBindingBundle.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    canonical = PreN10CanonicalSourceReferenceReportV1.model_validate_json(
        canonical_path.read_text(encoding="utf-8")
    )
    contract_v2 = PreN10ScientificContractReportV2.model_validate_json(
        v2_path.read_text(encoding="utf-8")
    )

    assert bundle.source_portfolio_id == portfolio.portfolio_id
    assert bundle.claim_count == 1
    assert bundle.prediction_source_id_present_count == 1
    assert bundle.falsifier_source_id_present_count == 1
    assert row.source_binding_bundle_id == bundle.bundle_id
    assert row.source_binding_bundle_sha256 == bundle.bundle_sha256

    assert canonical.source_binding_bundle_id == bundle.bundle_id
    assert row.canonical_source_reference_report_id == canonical.report_id
    assert (
        row.canonical_source_reference_report_sha256
        == canonical.report_sha256
    )
    assert canonical.stable_ready_count == 1

    assert contract_v2.source_binding_bundle_id == bundle.bundle_id
    assert (
        contract_v2.canonical_source_reference_report_id
        == canonical.report_id
    )
    assert row.contract_v2_report_id == contract_v2.report_id
    assert row.contract_v2_report_sha256 == contract_v2.report_sha256
    assert row.canonical_pre_n10_disposition == "READY_FOR_N10"
    assert contract_v2.disposition == "READY_FOR_N10"

    # Existing handoff compatibility is deliberately unchanged in S181c-1.
    assert row.contract_report_path is not None
    assert row.contract_report_id is not None
    assert row.pre_n10_disposition == "READY_FOR_N10"


def test_reentry_semantic_terminal_has_no_canonical_contract_artifacts(
    tmp_path: Path,
) -> None:
    context, regeneration = _regenerate(tmp_path)
    portfolio = _portfolio(regeneration)
    semantic_backend = _SemanticBackend(portfolio, accept=False)

    class _ShouldNotDecompose:
        def decompose(self, hypothesis, *, max_claims):
            raise AssertionError(
                "semantic terminal lineage must not decompose claims"
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

    row = report.lineages[0]
    assert row.final_status == "SEMANTIC_INTERVENTION_REQUIRED"
    assert row.source_binding_bundle_path is None
    assert row.canonical_source_reference_path is None
    assert row.contract_v2_report_path is None
    assert row.canonical_pre_n10_disposition is None


def test_reentry_malformed_contract_still_has_stable_source_provenance(
    tmp_path: Path,
) -> None:
    context, regeneration = _regenerate(tmp_path)
    portfolio = _portfolio(regeneration)
    semantic_backend = _SemanticBackend(portfolio, accept=True)
    decomposition_backend = _DecompositionBackend(malformed=True)

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

    row = report.lineages[0]
    assert row.final_status == "PRE_N10_INTERVENTION_REQUIRED"

    bundle = AtomicScientificSourceBindingBundle.model_validate_json(
        Path(str(row.source_binding_bundle_path)).read_text(
            encoding="utf-8"
        )
    )
    canonical = PreN10CanonicalSourceReferenceReportV1.model_validate_json(
        Path(str(row.canonical_source_reference_path)).read_text(
            encoding="utf-8"
        )
    )
    contract_v2 = PreN10ScientificContractReportV2.model_validate_json(
        Path(str(row.contract_v2_report_path)).read_text(
            encoding="utf-8"
        )
    )

    assert bundle.prediction_source_id_present_count == 1
    assert bundle.falsifier_source_id_present_count == 1
    assert canonical.stable_ready_count == 1
    assert contract_v2.disposition == "INTERVENTION_REQUIRED"
    assert row.canonical_pre_n10_disposition == (
        "INTERVENTION_REQUIRED"
    )
