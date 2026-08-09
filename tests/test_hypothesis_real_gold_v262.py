from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dac_her.hypothesis_real_gold import (
    build_real_gold_suite,
    to_semantic_gold_suite,
    validate_real_gold_lineage,
)
from dac_her.hypothesis_real_gold_contracts import HypothesisRealGoldSpec
from dac_her.hypothesis_semantic_contracts import SEMANTIC_DIMENSIONS


def _expectations():
    rows = []
    for dimension in SEMANTIC_DIMENSIONS:
        verdict = (
            "not_applicable"
            if dimension in {"candidate_calibration", "hypothesis_distinctness"}
            else "pass"
        )
        rows.append(
            {
                "dimension": dimension,
                "allowed_verdicts": [verdict],
                "critical": True,
                "note": "fixture",
            }
        )
    return rows


def _write_fixture(repo: Path) -> tuple[Path, Path]:
    context = {
        "schema_version": "hypothesis-context-v1",
        "context_id": "ctx:real",
        "context_sha256": "ctxsha",
        "source_packet_id": "packet:real",
        "source_packet_sha256": "packetsha",
        "source_report_id": "report:real",
        "source_report_sha256": "reportsha",
        "task_id": "task:real",
        "question": "Is there enough evidence?",
        "corpus_id": "fixture",
        "evidence_statements": [],
        "mechanism_routes": [],
        "mechanistic_motifs": [],
        "reported_design_levers": [],
        "research_gaps": [],
        "partial_absence_blocked_paper_ids": [],
        "policy": {
            "generated_hypotheses_allowed": True,
            "external_novelty_claims_allowed": False,
            "experiment_protocols_allowed": False,
            "unsupported_numeric_predictions_allowed": False,
            "alignment_can_be_scientific_premise": False,
            "unresolved_can_be_positive_premise": False,
            "navigation_note_can_be_positive_premise": False,
            "candidate_evidence_must_propagate": True,
            "falsifiable_prediction_required": True,
            "falsification_condition_required": True,
            "source_report_must_validate": True,
        },
    }
    portfolio = {
        "schema_version": "hypothesis-portfolio-v1",
        "portfolio_id": "portfolio:real",
        "source_context_id": "ctx:real",
        "source_context_sha256": "ctxsha",
        "source_report_id": "report:real",
        "source_report_sha256": "reportsha",
        "hypotheses": [],
        "abstention_reason": "No eligible positive premise is supplied.",
    }
    context_path = repo / "data" / "context.json"
    portfolio_path = repo / "data" / "portfolio.json"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(json.dumps(context), encoding="utf-8")
    portfolio_path.write_text(json.dumps(portfolio), encoding="utf-8")
    return context_path, portfolio_path


def _spec() -> HypothesisRealGoldSpec:
    return HypothesisRealGoldSpec.model_validate(
        {
            "suite_id": "real-gold-test",
            "cases": [
                {
                    "case_id": "real_case",
                    "description": "real-output fixture",
                    "context_path": "data/context.json",
                    "portfolio_path": "data/portfolio.json",
                    "generator_version": "test",
                    "expectations": _expectations(),
                }
            ],
        }
    )


def test_real_gold_requires_all_semantic_dimensions():
    payload = {
        "suite_id": "bad",
        "cases": [
            {
                "case_id": "bad",
                "description": "missing dimensions",
                "context_path": "context.json",
                "portfolio_path": "portfolio.json",
                "expectations": _expectations()[:-1],
            }
        ],
    }
    with pytest.raises(ValidationError):
        HypothesisRealGoldSpec.model_validate(payload)


def test_real_gold_build_and_lineage_preflight(tmp_path: Path):
    repo = tmp_path / "repo"
    _write_fixture(repo)
    output = repo / "benchmarks" / "real" / "gold.json"
    suite = build_real_gold_suite(
        _spec(),
        repo_root=repo,
        output_path=output,
    )
    assert not Path(suite.cases[0].context_path).is_absolute()
    assert not Path(suite.cases[0].portfolio_path).is_absolute()
    assert len(suite.cases[0].lineage.context_file_sha256) == 64
    assert len(suite.cases[0].lineage.portfolio_file_sha256) == 64

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(suite.model_dump_json(indent=2), encoding="utf-8")
    preflight = validate_real_gold_lineage(suite, suite_path=output)
    assert preflight.passed
    assert preflight.failed_cases == 0


def test_real_gold_preflight_detects_file_mutation(tmp_path: Path):
    repo = tmp_path / "repo"
    _, portfolio_path = _write_fixture(repo)
    output = repo / "benchmarks" / "real" / "gold.json"
    suite = build_real_gold_suite(
        _spec(),
        repo_root=repo,
        output_path=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(suite.model_dump_json(indent=2), encoding="utf-8")

    portfolio_path.write_text(
        portfolio_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    preflight = validate_real_gold_lineage(suite, suite_path=output)
    assert not preflight.passed
    codes = {
        issue.code
        for row in preflight.case_results
        for issue in row.issues
    }
    assert "PORTFOLIO_FILE_SHA_MISMATCH" in codes


def test_real_gold_projection_preserves_full_expectations(tmp_path: Path):
    repo = tmp_path / "repo"
    _write_fixture(repo)
    output = repo / "benchmarks" / "real" / "gold.json"
    suite = build_real_gold_suite(
        _spec(),
        repo_root=repo,
        output_path=output,
    )
    semantic = to_semantic_gold_suite(suite)
    assert semantic.suite_id == suite.suite_id
    assert len(semantic.cases[0].expectations) == len(SEMANTIC_DIMENSIONS)
