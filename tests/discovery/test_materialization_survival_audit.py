from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.materialization_survival_audit import (
    build_arm_survival_audit,
    build_cohort_survival_audit,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_generation_failure_is_not_misclassified_as_scientific_abstention(tmp_path):
    pool = tmp_path / "pool.json"
    selection = tmp_path / "selection.json"
    materialization = tmp_path / "materialization.json"
    generation = tmp_path / "generation.json"
    verification = tmp_path / "verification.json"

    _write(pool, {"candidates": [{
        "candidate_id": "c1",
        "origin": "EVOLUTION",
        "idea_form": "CROSS_SOURCE_BRIDGE",
        "operator_id": "CROSS_SOURCE_BRIDGE",
    }]})
    _write(selection, {
        "retained_candidate_ids": ["c1"],
        "selected_unique_family_count": 1,
    })
    _write(materialization, {"records": [{
        "candidate_id": "c1",
        "status": "ABSTAINED",
        "issue_codes": ["MODEL_ABSTAINED"],
        "issues": ["structured call failed: InstructorRetryException"],
    }]})
    _write(generation, {"generation_error": {"error_type": "InstructorRetryException"}})
    _write(verification, {"operational_status": "NOT_RUN"})

    audit = build_arm_survival_audit(
        case_id="x",
        arm="EVOLUTION_BALANCED",
        candidate_pool_path=pool,
        selection_path=selection,
        materialization_report_path=materialization,
        generation_path=generation,
        verification_summary_path=verification,
    )
    assert audit is not None
    assert audit.terminal_status_counts == {"GENERATION_FAILED": 1}
    assert audit.verification_ready_count == 0
    assert audit.generation_error_counts == {"InstructorRetryException": 1}


def test_transition_matrix_preserves_origin_operator_and_profile(tmp_path):
    pool = tmp_path / "pool.json"
    selection = tmp_path / "selection.json"
    materialization = tmp_path / "materialization.json"
    verification = tmp_path / "verification.json"

    _write(pool, {"candidates": [
        {"candidate_id": "c1", "origin": "FRONTIER", "idea_form": "RELATION_AXIS", "operator_id": None},
        {"candidate_id": "c2", "origin": "EVOLUTION", "idea_form": "MUTATED_TOPOLOGY", "operator_id": "BACKBONE_MUTATION"},
    ]})
    _write(selection, {
        "retained_candidate_ids": ["c1", "c2"],
        "retained_unique_family_count": 2,
        "entries": [
            {"candidate_id": "c1", "assigned_profile": "TASK_NEAR_VALIDATION"},
            {"candidate_id": "c2", "assigned_profile": "EXPLORATORY_BRIDGE"},
        ],
    })
    _write(materialization, {"records": [
        {"candidate_id": "c1", "status": "MATERIALIZED", "issue_codes": [], "issues": []},
        {"candidate_id": "c2", "status": "HARD_GATE_REJECTED", "issue_codes": ["X"], "issues": ["x"]},
    ]})
    _write(verification, {"operational_status": "COMPLETE"})

    audit = build_arm_survival_audit(
        case_id="x",
        arm="PORTFOLIO_SELECTED",
        candidate_pool_path=pool,
        selection_path=selection,
        materialization_report_path=materialization,
        generation_path=None,
        verification_summary_path=verification,
    )
    assert audit is not None
    assert audit.verification_ready_count == 1
    assert audit.verification_ready_fraction == 0.5
    by_key = {(x.origin, x.operator_or_form, x.selection_profile): x for x in audit.transition_cells}
    assert by_key[("FRONTIER", "RELATION_AXIS", "TASK_NEAR_VALIDATION")].verification_ready_count == 1
    assert by_key[("EVOLUTION", "BACKBONE_MUTATION", "EXPLORATORY_BRIDGE")].status_counts == {"HARD_GATE_REJECTED": 1}


def test_partial_workspace_is_valid_and_diagnostic_only(tmp_path):
    root = tmp_path / "dev"
    case = root / "CASE_A"
    run = tmp_path / "source_run"
    scientific = run / "frontier_augmented" / "scientific_portfolio"
    _write(root / "execution_status.json", {"planned_case_count": 6})
    _write(case / "case.audit.json", {"run_dir": str(run)})
    _write(scientific / "candidate_pool.json", {"candidates": [{
        "candidate_id": "f1", "origin": "FRONTIER", "idea_form": "RELATION_AXIS", "operator_id": None
    }]})
    _write(case / "FRONTIER_BALANCED" / "selection.json", {
        "retained_candidate_ids": ["f1"], "selected_unique_family_count": 1
    })
    _write(case / "FRONTIER_BALANCED" / "materialization.report.json", {
        "records": [{"candidate_id": "f1", "status": "MATERIALIZED", "issue_codes": [], "issues": []}]
    })

    audit = build_cohort_survival_audit(prospective_output_root=root)
    assert audit.observed_case_count == 1
    assert audit.planned_case_count == 6
    assert audit.partial_case_count == 1
    assert audit.arm_selected_candidate_counts["FRONTIER_BALANCED"] == 1
    assert audit.arm_verification_ready_counts["FRONTIER_BALANCED"] == 1
    assert audit.diagnostic_only is True
    assert audit.production_selection_authority is False
    assert audit.scientific_superiority_established is False


def test_interrupted_case_recovers_run_dir_and_candidate_lineage_from_execution_status(tmp_path):
    root = tmp_path / "dev"
    case = root / "CASE_INTERRUPTED"
    run = tmp_path / "source_run"
    scientific = run / "frontier_augmented" / "scientific_portfolio"

    _write(root / "execution_status.json", {
        "planned_case_count": 1,
        "cases": [{
            "case_id": "CASE_INTERRUPTED",
            "run_dir": str(run),
            "status": "PAUSED_PROVIDER_BUDGET_EXHAUSTED",
        }],
    })
    _write(scientific / "candidate_pool.json", {"candidates": [{
        "candidate_id": "c1",
        "origin": "EVOLUTION",
        "idea_form": "CROSS_SOURCE_BRIDGE",
        "operator_id": "CROSS_SOURCE_BRIDGE",
    }]})
    _write(case / "EVOLUTION_BALANCED" / "selection.json", {
        "retained_candidate_ids": ["c1"],
        "selected_unique_family_count": 1,
    })
    _write(case / "EVOLUTION_BALANCED" / "materialization.report.json", {
        "records": [{
            "candidate_id": "c1",
            "status": "ABSTAINED",
            "issue_codes": ["MODEL_ABSTAINED"],
            "issues": ["no grounded materialization"],
        }]
    })

    audit = build_cohort_survival_audit(prospective_output_root=root)
    assert audit.artifact_observed_case_count == 1
    assert audit.execution_observed_case_count == 1
    assert audit.artifact_only_case_count == 0
    arm = audit.cases[0].arms[0]
    assert arm.lineage_resolution_complete is True
    assert arm.unresolved_candidate_ids == []
    assert arm.selected_count_by_origin == {"EVOLUTION": 1}
    assert arm.selected_count_by_operator_or_form == {"CROSS_SOURCE_BRIDGE": 1}


def test_failure_taxonomy_is_exhaustive_and_mutually_exclusive(tmp_path):
    pool = tmp_path / "pool.json"
    selection = tmp_path / "selection.json"
    materialization = tmp_path / "materialization.json"

    _write(pool, {"candidates": [
        {"candidate_id": "a", "origin": "FRONTIER", "idea_form": "RELATION_AXIS"},
        {"candidate_id": "b", "origin": "FRONTIER", "idea_form": "HIGHER_ORDER_TOPOLOGY"},
    ]})
    _write(selection, {
        "retained_candidate_ids": ["a", "b"],
        "selected_unique_family_count": 2,
    })
    _write(materialization, {"records": [
        {
            "candidate_id": "a",
            "status": "ABSTAINED",
            "issue_codes": ["MODEL_ABSTAINED", "PARTIAL_PAPER_ABSENCE_CLAIM"],
            "issues": ["abstained"],
        },
        {
            "candidate_id": "b",
            "status": "HARD_GATE_REJECTED",
            "issue_codes": [],
            "issues": ["hard gate without code"],
        },
    ]})

    audit = build_arm_survival_audit(
        case_id="x",
        arm="FRONTIER_BALANCED",
        candidate_pool_path=pool,
        selection_path=selection,
        materialization_report_path=materialization,
        generation_path=None,
        verification_summary_path=None,
    )
    assert audit is not None
    assert audit.non_materialized_count == 2
    assert audit.taxonomy_classified_failure_count == 2
    assert audit.taxonomy_exhaustive is True
    assert audit.taxonomy_unclassified_failure_count == 1
    assert {x.issue_code: x.count for x in audit.issue_taxonomy} == {
        "MODEL_ABSTAINED": 1,
        "UNCLASSIFIED_HARD_GATE_REJECTION": 1,
    }
    # Raw multiplicity is still retained separately without double-counting the taxonomy.
    assert audit.raw_issue_code_counts == {
        "MODEL_ABSTAINED": 1,
        "PARTIAL_PAPER_ABSENCE_CLAIM": 1,
    }


def test_artifact_and_execution_observation_scopes_are_distinct(tmp_path):
    root = tmp_path / "dev"
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    _write(root / "execution_status.json", {
        "planned_case_count": 2,
        "cases": [
            {"case_id": "CASE_1", "run_dir": str(run1), "status": "COMPLETE"},
            {"case_id": "CASE_2", "run_dir": str(run2), "status": "NOT_STARTED_PROVIDER_BUDGET_PAUSE"},
        ],
    })
    for case_id, run in (("CASE_1", run1), ("CASE_2", run2)):
        case = root / case_id
        sci = run / "frontier_augmented" / "scientific_portfolio"
        _write(sci / "candidate_pool.json", {"candidates": [{
            "candidate_id": "c", "origin": "FRONTIER", "idea_form": "RELATION_AXIS"
        }]})
        _write(case / "FRONTIER_BALANCED" / "selection.json", {
            "retained_candidate_ids": ["c"], "selected_unique_family_count": 1
        })
        _write(case / "FRONTIER_BALANCED" / "materialization.report.json", {
            "records": [{"candidate_id": "c", "status": "MATERIALIZED", "issue_codes": [], "issues": []}]
        })

    audit = build_cohort_survival_audit(prospective_output_root=root)
    assert audit.observed_case_count == 2
    assert audit.artifact_observed_case_count == 2
    assert audit.execution_observed_case_count == 1
    assert audit.artifact_only_case_count == 1


def test_common_case_paired_comparison_uses_identical_case_set(tmp_path):
    root = tmp_path / "dev"
    run = tmp_path / "run"
    case = root / "CASE_A"
    sci = run / "frontier_augmented" / "scientific_portfolio"
    _write(root / "execution_status.json", {
        "planned_case_count": 1,
        "cases": [{"case_id": "CASE_A", "run_dir": str(run), "status": "COMPLETE"}],
    })
    _write(sci / "candidate_pool.json", {"candidates": [
        {"candidate_id": "f1", "origin": "FRONTIER", "idea_form": "RELATION_AXIS"},
        {"candidate_id": "f2", "origin": "FRONTIER", "idea_form": "HIGHER_ORDER_TOPOLOGY"},
        {"candidate_id": "e1", "origin": "EVOLUTION", "idea_form": "CROSS_SOURCE_BRIDGE", "operator_id": "CROSS_SOURCE_BRIDGE"},
        {"candidate_id": "e2", "origin": "EVOLUTION", "idea_form": "MUTATED_TOPOLOGY", "operator_id": "BACKBONE_MUTATION"},
    ]})

    # F: 1/2, E: 0/2.
    for arm, ids, statuses in (
        ("FRONTIER_BALANCED", ["f1", "f2"], ["MATERIALIZED", "ABSTAINED"]),
        ("EVOLUTION_BALANCED", ["e1", "e2"], ["ABSTAINED", "ABSTAINED"]),
    ):
        _write(case / arm / "selection.json", {
            "retained_candidate_ids": ids,
            "selected_unique_family_count": 2,
        })
        _write(case / arm / "materialization.report.json", {
            "records": [
                {"candidate_id": cid, "status": status,
                 "issue_codes": ([] if status == "MATERIALIZED" else ["MODEL_ABSTAINED"]),
                 "issues": []}
                for cid, status in zip(ids, statuses)
            ]
        })

    # D reuses source scientific-portfolio selection/materialization: 2/2.
    _write(sci / "selection.json", {
        "retained_candidate_ids": ["e1", "e2"],
        "retained_unique_family_count": 2,
        "entries": [
            {"candidate_id": "e1", "assigned_profile": "MECHANISM_FOCUSED"},
            {"candidate_id": "e2", "assigned_profile": "DISCRIMINATING_TEST"},
        ],
    })
    _write(sci / "materialization.report.json", {
        "records": [
            {"candidate_id": "e1", "status": "MATERIALIZED", "issue_codes": [], "issues": []},
            {"candidate_id": "e2", "status": "MATERIALIZED", "issue_codes": [], "issues": []},
        ]
    })

    audit = build_cohort_survival_audit(prospective_output_root=root)
    assert audit.common_materialization_case_count == 1
    assert audit.common_materialization_case_ids == ["CASE_A"]
    by_pair = {(x.left_arm, x.right_arm): x for x in audit.paired_comparisons}
    fd = by_pair[("FRONTIER_BALANCED", "PORTFOLIO_SELECTED")]
    ed = by_pair[("EVOLUTION_BALANCED", "PORTFOLIO_SELECTED")]
    assert fd.median_left_yield == 0.5
    assert fd.median_right_yield == 1.0
    assert fd.median_delta_right_minus_left == 0.5
    assert fd.right_higher_case_count == 1
    assert ed.median_left_yield == 0.0
    assert ed.median_right_yield == 1.0
    assert ed.median_delta_right_minus_left == 1.0
