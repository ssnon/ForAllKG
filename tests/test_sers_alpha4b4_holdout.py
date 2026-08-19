from __future__ import annotations

import json
from pathlib import Path

import pytest

from campaigns.sers_alpha4_epoch.holdout.cli.run_sers_alpha4b4_holdout import (
    FrozenContractViolation,
    build_holdout_report,
    validate_protocol,
)


def _protocol() -> dict:
    return {
        "protocol_version": "alpha4b4-v1",
        "holdout_scope": "evidence_substrate_from_frozen_existing_strict_extraction",
        "domain_profile": "sers_au_ag",
        "data_root": "data_sers",
        "mode": "exploratory",
        "calibration_papers": ["A"],
        "holdout_papers": ["B"],
        "frozen_semantics": {
            "corpus": "corpus-v1",
            "reproducibility": "repro-v1",
            "metric_definition": "metric-v1",
            "comparison": "comparison-v1",
            "method": "method-v1",
            "quality_gate": "quality-v1",
        },
        "acceptance_policy": {
            "minimum_numeric_ranking_allowed": None,
            "minimum_same_protocol_pairs": None,
            "maximum_unknown_contexts": None,
            "maximum_different_protocol_pairs": None,
            "minimum_metric_definition_known": None,
        },
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _outputs(root: Path, *, bad_global=False, structural=True) -> dict[str, str]:
    ids = {
        "corpus": "holdout_corpus",
        "reproducibility": "holdout_repro",
        "metric_definition": "holdout_metric",
        "comparison": "holdout_comparison",
    }
    croot = root / "data_sers" / "corpus" / ids["corpus"] / "exploratory"
    _write(croot / "manifest.json", {
        "paper_ids": ["B"],
        "corpus_semantics_id": "corpus-v1",
        "passes_structural_gate": structural,
        "destructive_cross_paper_merges": 0,
    })
    _write(croot / "audit.json", {"passes_structural_gate": structural})
    _write(croot / "reproducibility" / ids["reproducibility"] / "summary.json", {
        "reproducibility_semantics_id": "repro-v1",
        "structural_gate": structural,
        "evidence_count": 0,
        "evidence_kind_counts": {},
        "possible_duplicate_result_pair_count": 0,
    })
    _write(croot / "reproducibility" / ids["reproducibility"] / "audit.json", {"structural_gate": structural})
    _write(croot / "metric_definition" / ids["metric_definition"] / "summary.json", {
        "metric_definition_semantics_id": "metric-v1",
        "structural_gate": structural,
        "context_count": 9,
        "definition_status_counts": {"known": 1, "unknown": 8},
    })
    _write(croot / "metric_definition" / ids["metric_definition"] / "audit.json", {"structural_gate": structural})
    _write(croot / "comparison" / ids["comparison"] / "summary.json", {
        "comparison_semantics_id": "comparison-v1",
        "method_semantics_id": "method-v1",
        "quality_gate_semantics_id": "quality-v1",
        "passes_structural_gate": structural,
        "global_entity_concentration_consumed": bad_global,
        "missing_context_is_not_quarantine": True,
        "context_count": 20,
        "assessment_count": 40,
        "compatibility_counts": {"unknown": 40},
        "protocol_comparability_counts": {"different_protocol": 12, "unknown": 28},
        "method_dimension_status_counts": {"analyte": {"known": 2, "unknown": 18}},
        "metric_definition_compatibility_counts": {"unknown": 30, "not_applicable": 10},
        "metric_definition_ranking_relevant_assessment_count": 30,
        "metric_definition_ranking_relevant_gate_pass_count": 0,
        "numeric_ranking_allowed_count": 0,
        "observable_family_counts": {"sers_performance": 40},
        "unregistered_observable_assessment_count": 3,
    })
    _write(croot / "comparison" / ids["comparison"] / "audit.json", {"passes_structural_gate": structural})
    return ids


def test_protocol_requires_disjoint_sets_and_no_result_targets():
    protocol = _protocol()
    validate_protocol(protocol)
    protocol["acceptance_policy"]["minimum_numeric_ranking_allowed"] = 1
    with pytest.raises(ValueError, match="optimization targets"):
        validate_protocol(protocol)
    protocol = _protocol()
    protocol["holdout_papers"] = ["A"]
    with pytest.raises(ValueError, match="overlap"):
        validate_protocol(protocol)


def test_heterogeneity_and_zero_rankable_do_not_fail_holdout(tmp_path: Path):
    protocol = _protocol()
    ids = _outputs(tmp_path)
    report = build_holdout_report(tmp_path, protocol, ids)
    assert report["verdict"] == "pass"
    assert report["count_thresholds_used_for_acceptance"] is False
    assert report["distribution_observations"]["numeric_ranking_allowed_count"] == 0
    codes = {item["code"] for item in report["nonfatal_classifications"]}
    assert "different_protocol_pairs" in codes
    assert "unknown_protocol_pairs" in codes
    assert "unknown_metric_definitions" in codes
    assert "unregistered_observable_assessments" in codes


def test_global_entity_concentration_leak_is_spec_violation(tmp_path: Path):
    protocol = _protocol()
    ids = _outputs(tmp_path, bad_global=True)
    report = build_holdout_report(tmp_path, protocol, ids)
    assert report["verdict"] == "fail"
    assert any(item["code"] == "global_entity_concentration_consumed" for item in report["violations"])


def test_structural_gate_failure_is_spec_violation(tmp_path: Path):
    protocol = _protocol()
    ids = _outputs(tmp_path, structural=False)
    report = build_holdout_report(tmp_path, protocol, ids)
    assert report["verdict"] == "fail"
    assert any("structural_gate" in item["code"] for item in report["violations"])
