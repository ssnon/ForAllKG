from types import SimpleNamespace

from pipeline_core.discovery.prospective_validation import (
    build_cohort_audit,
    build_family_balanced_arm_selection,
    build_production_integration_readiness_shadow,
    provider_budget_exhausted_from_text,
    verification_summary_needs_resume,
)
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidate,
    ScientificPortfolioCandidatePool,
)
from pipeline_core.discovery.prospective_validation import (
    ProspectiveArmMetrics,
    ProspectiveCaseAudit,
    ProspectiveCohortAudit,
)


def _candidate(
    cid: str,
    *,
    origin: str,
    form: str,
    family: str,
):
    return ScientificPortfolioCandidate.model_construct(
        candidate_id=cid,
        origin=origin,
        source_object_id=cid,
        source_kind="KG_AXIS" if origin == "FRONTIER" else None,
        operator_id=(
            "CROSS_SOURCE_BRIDGE"
            if origin == "EVOLUTION"
            else None
        ),
        idea_form=form,
        title=cid,
        scientific_intent=cid,
        conceptual_change_summary="",
        core_relations=[],
        differential_prediction="",
        falsification_condition="",
        discriminating_observation="",
        task_relation_mode="UNKNOWN",
        conceptual_family_signature=family,
        parent_source_kinds=[],
        external_literature_lineage=False,
        candidate_or_unverified_lineage=False,
        cross_source_composition=False,
        source_artifact_refs=[],
        requires_verification=True,
        epistemic_status="INSPIRATION_ONLY",
        truth_authority=False,
        novelty_authority=False,
        positive_premise_authority=False,
        production_selection_authority=False,
    )


def _pool():
    candidates = [
        _candidate(
            "f1",
            origin="FRONTIER",
            form="RELATION_AXIS",
            family="fam-a",
        ),
        _candidate(
            "f2",
            origin="FRONTIER",
            form="RELATION_AXIS",
            family="fam-a",
        ),
        _candidate(
            "f3",
            origin="FRONTIER",
            form="HIGHER_ORDER_TOPOLOGY",
            family="fam-b",
        ),
        _candidate(
            "e1",
            origin="EVOLUTION",
            form="CROSS_SOURCE_BRIDGE",
            family="fam-c",
        ),
        _candidate(
            "e2",
            origin="EVOLUTION",
            form="MUTATED_TOPOLOGY",
            family="fam-d",
        ),
    ]
    return ScientificPortfolioCandidatePool.model_construct(
        pool_id="pool",
        pool_sha256="p" * 64,
        source_population_id="pop",
        source_population_sha256="q" * 64,
        source_evolution_report_id="evo",
        source_evolution_report_sha256="r" * 64,
        source_context_id="ctx",
        source_context_sha256="s" * 64,
        research_question="Q",
        raw_frontier_idea_count=3,
        raw_evolution_idea_count=2,
        projected_candidate_count=5,
        candidate_count_by_origin={"FRONTIER": 3, "EVOLUTION": 2},
        candidate_count_by_form={},
        projected_topology_backbone_representative_count=1,
        projected_candidate_topology_supplement_count=0,
        raw_topology_variant_count_omitted_from_evaluation=0,
        evaluation_budget_cap=48,
        evaluation_budget_cap_reached=False,
        candidates=candidates,
        structural_projection_only=True,
        scientific_quality_ranking_performed=False,
        candidate_deletion_authority=False,
        production_selection_authority=False,
    )


def test_frontier_arm_is_frontier_only_and_family_unique():
    selection = build_family_balanced_arm_selection(
        pool=_pool(),
        arm="FRONTIER_BALANCED",
        max_hypotheses=8,
    )
    assert selection.selected_candidate_count == 2
    assert selection.selected_unique_family_count == 2
    assert selection.selected_count_by_origin == {"FRONTIER": 2}
    assert len(selection.selected_family_signatures) == len(
        set(selection.selected_family_signatures)
    )
    assert selection.scientific_quality_ranking_performed is False
    assert selection.production_selection_authority is False


def test_evolution_arm_exposes_frontier_and_evolution_without_quality_ranking():
    selection = build_family_balanced_arm_selection(
        pool=_pool(),
        arm="EVOLUTION_BALANCED",
        max_hypotheses=8,
    )
    assert selection.selected_candidate_count == 4
    assert selection.selected_unique_family_count == 4
    assert selection.selected_count_by_origin["FRONTIER"] == 2
    assert selection.selected_count_by_origin["EVOLUTION"] == 2
    assert selection.selection_policy == (
        "DETERMINISTIC_FAMILY_BALANCED_NO_QUALITY_RANKING"
    )


def _arm(case_id: str):
    return ProspectiveArmMetrics.model_construct(
        case_id=case_id,
        arm="PORTFOLIO_SELECTED",
        source_portfolio_path="/tmp/p.json",
        hypothesis_count=4,
        selected_candidate_count=4,
        selected_unique_family_count=4,
        selected_count_by_origin={"FRONTIER": 2, "EVOLUTION": 2},
        selected_count_by_form={},
        materialized_hypothesis_count=4,
        materialization_yield_fraction=1.0,
        conceptual_selected_family_fraction=1.0,
        hypothesis_type_counts={},
        evidence_used_statement_count=2,
        evidence_eligible_statement_coverage=0.5,
        evidence_distinct_premise_set_count=4,
        evidence_exact_duplicate_premise_group_count=0,
        evidence_mean_pairwise_statement_jaccard=0.1,
        evidence_max_pairwise_statement_jaccard=0.2,
        premise_distinct_set_fraction=1.0,
        semantic_status="ACCEPTED",
        external_novelty_status="COMPLETE",
        external_status_counts={},
        n9_status="COMPLETE",
        feasibility_status="SKIPPED_UNSUPPORTED_DOMAIN",
        feasibility_disposition_counts={},
        operational_status="COMPLETE",
        provider_budget_pause_stage=None,
        downstream_verified_hypothesis_count=4,
        downstream_verification_yield_fraction=1.0,
        verification_completed=True,
        authority_ok=True,
        n10_run=False,
        production_selection_authority=False,
    )


def _cohort(kind: str, n: int):
    cases = []
    for i in range(n):
        case_id = f"c{i}"
        selected = _arm(case_id)
        arms = []
        for arm_name in (
            "LEGACY",
            "FRONTIER_BALANCED",
            "EVOLUTION_BALANCED",
            "PORTFOLIO_SELECTED",
        ):
            arms.append(
                selected.model_copy(
                    update={"arm": arm_name}
                )
            )
        cases.append(
            ProspectiveCaseAudit.model_construct(
                case_id=case_id,
                run_dir=f"/tmp/{case_id}",
                context_id="ctx",
                context_sha256="s" * 64,
                arm_count=4,
                arms=arms,
                all_four_arms_present=True,
                selected_arm_verification_complete=True,
                execution_status="COMPLETE",
                paused_arm=None,
                production_selection_authority=False,
            )
        )
    return ProspectiveCohortAudit.model_construct(
        cohort_id="cohort",
        cohort_sha256="z" * 64,
        cohort_kind=kind,
        case_count=n,
        planned_case_count=n,
        complete_case_count=n,
        partial_case_count=0,
        paused_provider_budget_case_count=0,
        four_arm_record_present_case_count=n,
        complete_four_arm_case_count=n,
        cases=cases,
        arm_case_counts={},
        arm_semantic_accepted_counts={},
        arm_external_complete_counts={},
        arm_n9_complete_counts={},
        arm_supported_feasibility_complete_counts={},
        arm_hypothesis_count_medians={},
        arm_materialization_yield_medians={},
        arm_downstream_verification_yield_medians={},
        arm_conceptual_family_fraction_medians={},
        arm_premise_distinct_set_fraction_medians={},
        arm_evidence_coverage_medians={},
        arm_mean_pairwise_premise_jaccard_medians={},
        arm_external_status_counts={},
        arm_operational_status_counts={},
        portfolio_selected_frontier_retention_case_count=n,
        portfolio_selected_evolution_retention_case_count=n,
        all_authority_invariants_hold=True,
        scientific_superiority_established=False,
        cross_case_winner_selected=False,
        production_selection_authority=False,
    )


def test_production_readiness_requires_held_out_after_development():
    report = build_production_integration_readiness_shadow(
        development=_cohort("DEVELOPMENT", 6),
        held_out=None,
        min_held_out_cases=4,
    )
    assert report.status == "HELD_OUT_PROSPECTIVE_REQUIRED"
    assert report.scientific_superiority_established is False
    assert report.automatic_production_promotion_allowed is False
    assert report.production_selection_authority is False


def test_operationally_complete_held_out_only_reaches_human_review():
    report = build_production_integration_readiness_shadow(
        development=_cohort("DEVELOPMENT", 6),
        held_out=_cohort("HELD_OUT", 4),
        min_held_out_cases=4,
    )
    assert report.status == "READY_FOR_HUMAN_PRODUCTION_REVIEW"
    assert report.human_scientific_review_required is True
    assert report.automatic_production_promotion_allowed is False
    assert report.production_selection_authority is False


def test_provider_budget_exhaustion_is_operational_pause_not_scientific_failure():
    assert provider_budget_exhausted_from_text(
        "HTTP/2 429\nx-ratelimit-remaining: 0\napi.openalex.org"
    )
    assert not provider_budget_exhausted_from_text(
        "HTTP/2 500 unrelated provider error"
    )


def test_resume_classifier_reuses_complete_and_retries_paused_or_failed():
    assert not verification_summary_needs_resume(
        {"operational_status": "COMPLETE"}
    )
    assert not verification_summary_needs_resume(
        {"operational_status": "SCIENTIFIC_TERMINAL"}
    )
    assert verification_summary_needs_resume(
        {"operational_status": "PAUSED_PROVIDER_BUDGET_EXHAUSTED"}
    )
    assert verification_summary_needs_resume(
        {"operational_status": "FAILED_OPERATIONAL"}
    )
    assert not verification_summary_needs_resume(
        {"status": "COMPLETE_SHADOW_VERIFICATION"}
    )


def test_cohort_audit_preserves_legacy_selection_metrics_as_not_applicable():
    selected = _arm("c0")
    legacy = selected.model_copy(
        update={
            "arm": "LEGACY",
            "selected_candidate_count": None,
            "selected_unique_family_count": None,
            "materialization_yield_fraction": None,
            "conceptual_selected_family_fraction": None,
        }
    )
    case = ProspectiveCaseAudit.model_construct(
        case_id="c0",
        run_dir="/tmp/c0",
        context_id="ctx",
        context_sha256="s" * 64,
        arm_count=1,
        arms=[legacy],
        all_four_arms_present=False,
        selected_arm_verification_complete=False,
        execution_status="PARTIAL",
        paused_arm=None,
        production_selection_authority=False,
    )
    cohort = build_cohort_audit(
        cohort_kind="DEVELOPMENT",
        cases=[case],
        planned_case_count=1,
    )
    assert cohort.arm_materialization_yield_medians["LEGACY"] is None
    assert cohort.arm_conceptual_family_fraction_medians["LEGACY"] is None


def test_four_arm_presence_is_not_terminal_completion_when_one_arm_is_paused():
    arms = []
    for arm_name in (
        "LEGACY",
        "FRONTIER_BALANCED",
        "EVOLUTION_BALANCED",
        "PORTFOLIO_SELECTED",
    ):
        row = _arm("c0").model_copy(update={"arm": arm_name})
        if arm_name == "PORTFOLIO_SELECTED":
            row = row.model_copy(
                update={
                    "operational_status": "PAUSED_PROVIDER_BUDGET_EXHAUSTED",
                    "verification_completed": False,
                    "external_novelty_status": "NOT_RUN",
                    "n9_status": "NOT_RUN",
                }
            )
        arms.append(row)

    case = ProspectiveCaseAudit.model_construct(
        case_id="c0",
        run_dir="/tmp/c0",
        context_id="ctx",
        context_sha256="s" * 64,
        arm_count=4,
        arms=arms,
        all_four_arms_present=True,
        selected_arm_verification_complete=False,
        execution_status="PAUSED_PROVIDER_BUDGET_EXHAUSTED",
        paused_arm="PORTFOLIO_SELECTED",
        production_selection_authority=False,
    )
    cohort = build_cohort_audit(
        cohort_kind="DEVELOPMENT",
        cases=[case],
        planned_case_count=1,
    )
    assert cohort.four_arm_record_present_case_count == 1
    assert cohort.complete_four_arm_case_count == 0
