from __future__ import annotations

import inspect

from scripts.discovery import (
    run_dac_discovery_e2e as e2e,
)


def helper():
    return inspect.getsource(
        e2e._run_realization_search_production_stage8
    )


def test_helper_exists():
    assert callable(
        e2e._run_realization_search_production_stage8
    )


def test_width_three_and_retain_one():
    src = helper()

    assert "search_width=3" in src

    assert (
        "retained_hypotheses_per_axis=1"
        in src
    )


def test_plan_is_frozen_before_realization_loop():
    src = helper()

    assert '"--dry-run-plan"' in src

    assert (
        "for slot_index in range("
        in src
    )

    assert (
        "policy.search_width"
        in src
    )


def test_each_realization_uses_frozen_axis_plan():
    trajectory = inspect.getsource(
        e2e._run_realization_candidate_chain
    )

    assert (
        '"--axis-plan-input"'
        in trajectory
    )

    src = helper()

    assert (
        "frozen_axis_plan=("
        in src
    )


def test_realization_semantic_two_pass_observations_used():
    src = helper()

    assert (
        "_realization_semantic_observations("
        in src
    )


def test_cohort_selector_materializer_used():
    src = helper()

    assert (
        "build_axis_realization_cohort("
        in src
    )

    # B9 inserts task-preservation eligibility before semantic-tier
    # production ranking. The E2E therefore calls the task-aware
    # wrapper rather than the lower-level selector directly.
    assert (
        "select_axis_task_aware_production_winner("
        in src
    )

    assert (
        "materialize_realization_winners("
        in src
    )


def test_canonical_portfolio_and_lineage_become_winners():
    src = helper()

    assert (
        "axis_portfolio,"
        in src
    )

    assert (
        "materialized.portfolio"
        in src
    )

    assert (
        "materialized.lineage_report"
        in src
    )


def test_inference_and_context_are_rebound():
    src = helper()

    assert (
        'artifact_kind="inference"'
        in src
    )

    assert (
        'artifact_kind="context"'
        in src
    )


def test_winner_diversity_recomputed():
    src = helper()

    assert (
        "HypothesisEvidenceDiversityAssessor()"
        in src
    )

    assert (
        "materialized.portfolio"
        in src
    )


def test_selection_artifacts_are_durable():
    src = helper()

    assert (
        "realization_search.cohort.production.json"
        in src
    )

    assert (
        "realization_search.selection.production.json"
        in src
    )

    assert (
        "realization_search.materialization.production.json"
        in src
    )


def test_manifest_marks_production_selection_changed():
    src = helper()

    assert (
        '"production_selection_applied":'
        in src
    )

    assert (
        '"production_selection_changed":'
        in src
    )


def test_pipeline_has_explicit_enforcement_branch():
    src = inspect.getsource(
        e2e.run_pipeline
    )

    assert (
        "if args.realization_search_enforce:"
        in src
    )

    assert (
        "_run_realization_search_production_stage8("
        in src
    )


def test_legacy_single_realization_path_still_exists():
    src = inspect.getsource(
        e2e.run_pipeline
    )

    assert (
        '"[8/13] Discovery-axis hypothesis synthesis"'
        in src
    )

    assert "else:" in src


def test_task_preservation_filters_before_production_winner_ranking():
    src = helper()

    assert (
        "evaluate_hypothesis_task_preservation("
        in src
    )

    assert (
        "select_axis_task_aware_production_winner("
        in src
    )

    assert (
        "task_preservation_before_winner_ranking"
        in src
    )

    # Semantic observations are still produced for generated candidates;
    # task preservation restricts winner eligibility rather than deleting
    # raw semantic evaluation.
    assert (
        "_realization_semantic_observations("
        in src
    )

    assert (
        "semantic_evaluation_preserved_for_task_ineligible"
        in src
    )
