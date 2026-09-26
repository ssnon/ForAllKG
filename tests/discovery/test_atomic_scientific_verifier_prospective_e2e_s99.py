from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline_core.discovery.atomic_scientific_verifier_prospective import (
    build_atomic_scientific_verifier_prospective_cohort_freeze,
    build_atomic_scientific_verifier_prospective_comparison,
)


def _freeze_inputs():
    context = SimpleNamespace(
        context_id="ctx:1",
        context_sha256="a" * 64,
        task_id="task:1",
        domain_profile_id="sers_au_ag",
    )
    candidates = SimpleNamespace(
        portfolio_id="candidate:1",
        source_context_id="ctx:1",
        source_context_sha256="a" * 64,
        source_task_id="task:1",
        domain_profile_id="sers_au_ag",
    )
    specs = [SimpleNamespace(claim_id="claim:1")]
    atomic_report = SimpleNamespace(
        report_id="atomic-report:1",
        source_candidate_portfolio_id="candidate:1",
        source_context_id="ctx:1",
        source_task_id="task:1",
        hypotheses=[
            SimpleNamespace(
                hypothesis_id="h1",
                atomic_specifications=specs,
            )
        ],
    )
    atomic_portfolio = SimpleNamespace(
        portfolio_id="portfolio:1",
        source_context_id="ctx:1",
        source_context_sha256="a" * 64,
        domain_profile_id="sers_au_ag",
        hypotheses=[SimpleNamespace(hypothesis_id="h1")],
    )
    query_plan = SimpleNamespace(
        plan_id="plan:1",
        source_portfolio_id="portfolio:1",
        claims=[
            SimpleNamespace(
                hypothesis_id="h1",
                claims=[SimpleNamespace(claim_id="claim:1")],
            )
        ],
    )
    return context, candidates, atomic_report, atomic_portfolio, query_plan


def test_prospective_freeze_binds_atomic_population_before_verification():
    freeze = build_atomic_scientific_verifier_prospective_cohort_freeze(
        context=_freeze_inputs()[0],
        candidate_portfolio=_freeze_inputs()[1],
        atomic_report=_freeze_inputs()[2],
        atomic_portfolio=_freeze_inputs()[3],
        query_plan=_freeze_inputs()[4],
    )

    assert freeze.hypothesis_ids == ["h1"]
    assert freeze.claim_ids == ["claim:1"]
    assert freeze.frozen_before_old_n10 is True
    assert freeze.frozen_before_new_verifier is True
    assert freeze.verifier_results_observed_before_freeze is False
    assert freeze.production_selection_authority is False


def test_prospective_freeze_rejects_query_plan_population_mismatch():
    context, candidates, atomic_report, atomic_portfolio, query_plan = _freeze_inputs()
    query_plan.claims = [
        SimpleNamespace(
            hypothesis_id="other",
            claims=[SimpleNamespace(claim_id="claim:1")],
        )
    ]

    with pytest.raises(ValueError, match="query-plan hypothesis set"):
        build_atomic_scientific_verifier_prospective_cohort_freeze(
            context=context,
            candidate_portfolio=candidates,
            atomic_report=atomic_report,
            atomic_portfolio=atomic_portfolio,
            query_plan=query_plan,
        )


def test_prospective_comparison_records_matrix_and_diagnostic_factors():
    context, candidates, atomic_report, atomic_portfolio, query_plan = _freeze_inputs()
    freeze = build_atomic_scientific_verifier_prospective_cohort_freeze(
        context=context,
        candidate_portfolio=candidates,
        atomic_report=atomic_report,
        atomic_portfolio=atomic_portfolio,
        query_plan=query_plan,
    )

    old = SimpleNamespace(
        report_id="old:1",
        source_portfolio_id="portfolio:1",
        decisions=[
            SimpleNamespace(
                hypothesis_id="h1",
                certification_status="NOVELTY_CERTIFIED",
                selection_class="ELIGIBLE",
                positive_nonobviousness_authority=True,
            )
        ],
    )
    new = SimpleNamespace(
        report_id="new:1",
        decisions=[
            SimpleNamespace(
                hypothesis_id="h1",
                decision="UNRESOLVED",
                bounded_closure_state="PARTIAL_REVIEW_COVERAGE",
                bounded_external_distinctness_state=(
                    "BOUNDED_EXTERNAL_DISTINCTNESS_SUPPORTED"
                ),
                positive_nonobviousness_authority_state="NOT_AUTHORIZED",
                fatal_blocker_state="NONE",
                reason_codes=[
                    "bounded_evidence_review_not_closed",
                    "positive_nonobviousness_not_authorized",
                ],
            )
        ],
    )
    aggregation = SimpleNamespace(
        report_id="agg:1",
        hypothesis_aggregations=[
            SimpleNamespace(
                hypothesis_id="h1",
                strong_pressure_kind_counts={
                    "LOWER_ORDER_RELATION_PRIOR_ART": 1,
                },
            )
        ],
    )

    relation_candidates = SimpleNamespace(
        report_id="candidates:1",
        claims=[
            SimpleNamespace(
                hypothesis_id="h1",
                typed_identity_excluded_work_count=2,
            )
        ],
    )
    report = build_atomic_scientific_verifier_prospective_comparison(
        cohort_freeze=freeze,
        old_n10=old,
        new_verifier=new,
        aggregation=aggregation,
        relation_candidates=relation_candidates,
    )

    assert report.comparison_cells == {
        "OLD_CERTIFIED__NEW_UNRESOLVED": 1,
    }
    assert report.disagreement_count == 1
    assert report.rows[0].diagnostic_factors == [
        "coverage_failure",
        "positive_nonobviousness_absence",
        "lower_order_prior_art",
        "typed_identity_exclusion_present",
    ]
    assert report.production_selection_changed is False


def test_prospective_comparison_rejects_nonfrozen_old_n10_portfolio():
    context, candidates, atomic_report, atomic_portfolio, query_plan = _freeze_inputs()
    freeze = build_atomic_scientific_verifier_prospective_cohort_freeze(
        context=context,
        candidate_portfolio=candidates,
        atomic_report=atomic_report,
        atomic_portfolio=atomic_portfolio,
        query_plan=query_plan,
    )
    old = SimpleNamespace(
        report_id="old:1",
        source_portfolio_id="different",
        decisions=[],
    )
    new = SimpleNamespace(report_id="new:1", decisions=[])
    aggregation = SimpleNamespace(report_id="agg:1", hypothesis_aggregations=[])
    relation_candidates = SimpleNamespace(report_id="candidates:1", claims=[])

    with pytest.raises(ValueError, match="frozen atomic portfolio"):
        build_atomic_scientific_verifier_prospective_comparison(
            cohort_freeze=freeze,
            old_n10=old,
            new_verifier=new,
            aggregation=aggregation,
            relation_candidates=relation_candidates,
        )


def test_prospective_wrapper_never_enables_legacy_smoke_mode():
    from pathlib import Path

    source = (
        Path(__file__).parents[2]
        / "scripts"
        / "discovery"
        / "run_atomic_scientific_verifier_prospective_e2e.py"
    ).read_text(encoding="utf-8")

    assert "--candidate-contract-only" in source
    assert '"--dry-run"' in source
    assert '"--authority-mode"' in source
    assert '"certification_only"' in source
    assert "--allow-legacy-atomic-n10-artifacts" not in source
    assert "--allow-atomic-report-compatibility" not in source
    assert 'old_manifest.get("status") != "complete"' in source


def test_pre_n10_companion_exposes_candidate_contract_only_stop():
    from pathlib import Path

    source = (
        Path(__file__).parents[2]
        / "scripts"
        / "discovery"
        / "run_pre_n10_scientific_synthesis_companion.py"
    ).read_text(encoding="utf-8")

    assert '"--candidate-contract-only"' in source
    assert 'manifest["status"] = "complete_candidate_contract_only"' in source
    assert "if not args.candidate_contract_only:" in source


def test_prospective_wrapper_does_not_poison_fresh_run_dir_before_legacy_e2e():
    from pathlib import Path

    source = (
        Path(__file__).parents[2]
        / "scripts"
        / "discovery"
        / "run_atomic_scientific_verifier_prospective_e2e.py"
    ).read_text(encoding="utf-8")

    pre_dry = source.split("if args.dry_run:", 1)[0]
    assert "_write(manifest_path, manifest)" not in pre_dry
    assert "plan_manifest_path = run.parent" in source
    assert "_write(plan_manifest_path, manifest)" in source

    legacy_call = source.index('"fresh_legacy_discovery_e2e"')
    actual_manifest_write = source.index(
        "_write(manifest_path, manifest)",
        legacy_call,
    )
    assert actual_manifest_write > legacy_call

def test_prospective_freeze_binds_canonical_bundle_identity_and_population():
    context, candidates, atomic_report, atomic_portfolio, query_plan = (
        _freeze_inputs()
    )
    bundle = SimpleNamespace(
        bundle_id="bundle:1",
        bundle_sha256="b" * 64,
        source_report_id=atomic_report.report_id,
        source_contract="atomic-cross-lane-scientific-synthesis-report-v1",
        hypotheses=[
            SimpleNamespace(
                hypothesis_id="h1",
                specifications=[
                    SimpleNamespace(claim_id="claim:1")
                ],
            )
        ],
    )
    atomic_report.schema_version = (
        "atomic-cross-lane-scientific-synthesis-report-v1"
    )

    freeze = build_atomic_scientific_verifier_prospective_cohort_freeze(
        context=context,
        candidate_portfolio=candidates,
        atomic_report=atomic_report,
        atomic_portfolio=atomic_portfolio,
        query_plan=query_plan,
        canonical_bundle=bundle,
    )

    assert freeze.canonical_spec_bundle_id == "bundle:1"
    assert freeze.canonical_spec_bundle_sha256 == "b" * 64
    assert freeze.canonical_spec_bundle_population_verified is True


def test_prospective_wrapper_threads_canonical_bundle_to_verifier():
    from pathlib import Path

    source = (
        Path(__file__).parents[2]
        / "scripts"
        / "discovery"
        / "run_atomic_scientific_verifier_prospective_e2e.py"
    ).read_text(encoding="utf-8")

    assert '"--canonical-spec-output"' in source
    assert '"--canonical-spec-bundle"' in source
    assert "canonical_spec_bundle_population_verified" in source
    assert "canonical specification bundle SHA changed after cohort freeze" in source
