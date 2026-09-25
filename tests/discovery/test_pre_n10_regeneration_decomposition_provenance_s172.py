from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
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


def _lineage_dir(root: Path, source_hypothesis_id: str) -> Path:
    return root / "lineage" / (
        source_hypothesis_id.replace(":", "_").replace("/", "_")
    )


def _run_pass(tmp_path: Path, root: Path):
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
        output_root=root,
    )
    return report


def test_regeneration_reentry_preserves_decomposition_sanitization_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reentry-v2"
    report = _run_pass(tmp_path, root)
    row = report.lineages[0]

    audit_path = (
        _lineage_dir(root, row.source_hypothesis_id)
        / "claim_decomposition.sanitization_audit.json"
    )
    assert audit_path.is_file()

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["schema_version"] == (
        "pre-n10-regeneration-decomposition-"
        "sanitization-audit-v1"
    )
    assert audit["source_hypothesis_id"] == row.source_hypothesis_id
    assert audit["record_count"] == 1
    assert audit["prediction_source_id_count"] == 1
    assert audit["falsifier_source_id_count"] == 1
    assert audit["prediction_and_falsifier_source_id_count"] == 1

    assert audit["diagnostic_only"] is True
    assert audit["production_authority"] is False
    assert audit["scientific_content_mutated"] is False
    assert audit["query_plan_mutated_by_audit"] is False
    assert audit["contract_mutated_by_audit"] is False
    assert audit["retrieval_performed"] is False
    assert audit["external_novelty_performed"] is False
    assert audit["n9_performed"] is False
    assert audit["n10_performed"] is False

    fidelity = audit["records"][0]["semantic_fidelity_shadow"]
    assert fidelity["prediction_observation_id"]
    assert fidelity["falsification_criterion_id"]

    assert Path(audit["query_plan_path"]) == Path(row.query_plan_path)
    assert Path(audit["contract_report_path"]) == Path(
        row.contract_report_path
    )
    assert audit["query_plan_id"] == row.query_plan_id
    assert audit["contract_report_id"] == row.contract_report_id


def test_regeneration_reentry_semantic_terminal_writes_no_decomposition_audit(
    tmp_path: Path,
) -> None:
    context, regeneration = _regenerate(tmp_path)
    portfolio = _portfolio(regeneration)
    semantic_backend = _SemanticBackend(portfolio, accept=False)
    root = tmp_path / "reentry-v2"

    class _ShouldNotDecompose:
        def decompose(self, hypothesis, *, max_claims):
            raise AssertionError(
                "semantic intervention must block decomposition"
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
        output_root=root,
    )

    row = report.lineages[0]
    assert row.final_status == "SEMANTIC_INTERVENTION_REQUIRED"
    assert not (
        _lineage_dir(root, row.source_hypothesis_id)
        / "claim_decomposition.sanitization_audit.json"
    ).exists()


def test_regeneration_decomposition_audit_is_write_once_replay_exact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reentry-v2"

    first = _run_pass(tmp_path, root)
    row = first.lineages[0]
    audit_path = (
        _lineage_dir(root, row.source_hypothesis_id)
        / "claim_decomposition.sanitization_audit.json"
    )
    before = audit_path.read_bytes()

    second = _run_pass(tmp_path, root)
    after = audit_path.read_bytes()

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert before == after
