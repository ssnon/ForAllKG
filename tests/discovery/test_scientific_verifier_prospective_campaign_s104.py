from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_core.discovery.scientific_verifier_prospective_campaign import (
    build_scientific_verifier_prospective_campaign_report,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _abstention(root: Path, case_id: str) -> None:
    _write(root / case_id / "scientific_atomic_verifier_prospective_e2e_manifest.json", {
        "status": "abstained_upstream_before_candidate_contract",
        "upstream_pre_n10_status": "abstained_no_reframing_candidates",
        "verifier_result_consumed_by_production": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    })


def _complete(root: Path, case_id: str) -> None:
    run = root / case_id
    _write(run / "scientific_atomic_verifier_prospective_e2e_manifest.json", {
        "status": "complete",
        "verifier_result_consumed_by_production": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    })
    _write(run / "scientific_pre_n10_candidate_portfolio.json", {"candidate_count": 4})
    _write(run / "scientific_atomic_verifier_prospective_cohort.json", {
        "hypothesis_count": 2,
        "claim_count": 2,
        "frozen_before_old_n10": True,
        "frozen_before_new_verifier": True,
        "verifier_results_observed_before_freeze": False,
        "old_n10_results_observed_before_freeze": False,
        "production_selection_authority": False,
        "canonical_graph_mutated": False,
    })
    _write(run / "scientific_atomic_verifier_prospective_comparison.json", {
        "hypothesis_count": 2,
        "agreement_count": 2,
        "disagreement_count": 0,
        "comparison_cells": {"OLD_UNRESOLVED__NEW_UNRESOLVED": 2},
        "cohort_changed_after_freeze": False,
        "old_n10_mutated_by_verifier": False,
        "production_selection_changed": False,
        "comparison_is_diagnostic_only": True,
    })


def test_five_case_campaign_matches_frozen_p01_p05_shape(tmp_path: Path) -> None:
    for case_id in ("P01", "P02", "P03", "P05"):
        _abstention(tmp_path, case_id)
    _complete(tmp_path, "P04")
    report = build_scientific_verifier_prospective_campaign_report(
        root=tmp_path,
        case_ids=["P01", "P02", "P03", "P04", "P05"],
    )
    assert report.case_count == 5
    assert report.disposition_counts == {"UPSTREAM_ABSTENTION": 4, "VERIFIER_REACHED_COMPLETE": 1}
    assert report.verifier_reached_case_count == 1
    assert report.verifier_not_reached_case_count == 4
    assert report.frozen_hypothesis_count == 2
    assert report.frozen_claim_count == 2
    assert report.old_new_agreement_count == 2
    assert report.old_new_disagreement_count == 0
    assert report.comparison_cells == {"OLD_UNRESOLVED__NEW_UNRESOLVED": 2}


def test_campaign_rejects_post_freeze_cohort_mutation(tmp_path: Path) -> None:
    _complete(tmp_path, "P04")
    path = tmp_path / "P04" / "scientific_atomic_verifier_prospective_comparison.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["cohort_changed_after_freeze"] = True
    _write(path, value)
    with pytest.raises(ValueError, match="post-freeze cohort mutation"):
        build_scientific_verifier_prospective_campaign_report(root=tmp_path, case_ids=["P04"])


def test_campaign_rejects_production_contamination(tmp_path: Path) -> None:
    _complete(tmp_path, "P04")
    path = tmp_path / "P04" / "scientific_atomic_verifier_prospective_e2e_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["production_selection_changed"] = True
    _write(path, value)
    with pytest.raises(ValueError, match="production selection mutation"):
        build_scientific_verifier_prospective_campaign_report(root=tmp_path, case_ids=["P04"])


def test_campaign_rejects_nonterminal_case(tmp_path: Path) -> None:
    _write(tmp_path / "P01" / "scientific_atomic_verifier_prospective_e2e_manifest.json", {
        "status": "failed",
        "verifier_result_consumed_by_production": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    })
    with pytest.raises(ValueError, match="not terminal"):
        build_scientific_verifier_prospective_campaign_report(root=tmp_path, case_ids=["P01"])


def test_campaign_report_hash_is_deterministic(tmp_path: Path) -> None:
    _abstention(tmp_path, "P01")
    first = build_scientific_verifier_prospective_campaign_report(root=tmp_path, case_ids=["P01"])
    second = build_scientific_verifier_prospective_campaign_report(root=tmp_path, case_ids=["P01"])
    assert first.report_id == second.report_id
    assert first.report_sha256 == second.report_sha256
