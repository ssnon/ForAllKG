import pytest
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceStatement,
    PredictedObservationDraft,
)
from pipeline_core.discovery.scientific_portfolio_materialization import (
    ScientificPortfolioMaterializationBatchDraft,
    ScientificPortfolioMaterializationItemDraft,
    compile_scientific_portfolio_materialization,
)
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidate,
    ScientificPortfolioCandidatePool,
    ScientificPortfolioSelectionEntry,
    ScientificPortfolioSelectionReport,
)


def _context():
    return HypothesisContext(
        context_id="ctx",
        context_sha256="csha",
        source_packet_id="packet",
        source_packet_sha256="psha",
        source_report_id="report",
        source_report_sha256="rsha",
        task_id="task",
        question="How does A affect B?",
        corpus_id="c",
        domain_profile_id="dac_her",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="s1", text="A correlates with B.", epistemic_role="reported",
                claim_kind="result", paper_ids=["p1"], eligible_as_premise=True,
                eligible_as_gap=False,
            ),
            HypothesisEvidenceStatement(
                statement_id="g1", text="The mediator remains unresolved.", epistemic_role="unresolved",
                claim_kind="gap", paper_ids=["p1"], eligible_as_premise=False,
                eligible_as_gap=True,
            ),
        ],
    )


def _pool_and_selection():
    candidate = ScientificPortfolioCandidate(
        candidate_id="cand:1", origin="EVOLUTION", source_object_id="evo:1",
        operator_id="CROSS_SOURCE_BRIDGE", idea_form="CROSS_SOURCE_BRIDGE",
        title="bridge", scientific_intent="A may affect B through a mediator.",
        task_relation_mode="UNKNOWN", conceptual_family_signature="fam1",
        parent_source_kinds=["KG_AXIS", "OPEN_WORLD_AXIS"],
        external_literature_lineage=True, cross_source_composition=True,
    )
    pool = ScientificPortfolioCandidatePool(
        pool_id="pool", pool_sha256="x", source_population_id="pop",
        source_population_sha256="ps", source_evolution_report_id="e",
        source_evolution_report_sha256="es", source_context_id="ctx",
        source_context_sha256="csha", research_question="Q",
        raw_frontier_idea_count=1, raw_evolution_idea_count=1,
        projected_candidate_count=1, candidate_count_by_origin={"EVOLUTION":1},
        candidate_count_by_form={"CROSS_SOURCE_BRIDGE":1},
        projected_topology_backbone_representative_count=0,
        projected_candidate_topology_supplement_count=0,
        raw_topology_variant_count_omitted_from_evaluation=0,
        evaluation_budget_cap=48, evaluation_budget_cap_reached=False,
        candidates=[candidate],
    )
    selection = ScientificPortfolioSelectionReport(
        selection_id="sel", selection_sha256="ss", source_pool_id="pool",
        source_pool_sha256="x", source_evaluation_report_id="eval",
        source_evaluation_report_sha256="es", max_retained_candidates=8,
        max_retained_per_profile=2, retained_count=1, retained_unique_family_count=1,
        retained_count_by_profile={"EXPLORATORY_BRIDGE":1},
        retained_count_by_origin={"EVOLUTION":1}, retained_candidate_ids=["cand:1"],
        entries=[ScientificPortfolioSelectionEntry(
            candidate_id="cand:1", source_object_id="evo:1", origin="EVOLUTION",
            assigned_profile="EXPLORATORY_BRIDGE", eligible_profiles=["EXPLORATORY_BRIDGE"],
            conceptual_family_signature="fam1", pareto_layer=1,
            verification_burden="HIGH",
        )],
    )
    return pool, selection


def test_materialization_uses_existing_grounded_hypothesis_guards():
    pool, selection = _pool_and_selection()
    draft = ScientificPortfolioMaterializationBatchDraft(items=[
        ScientificPortfolioMaterializationItemDraft(
            candidate_id="cand:1", title="Grounded bridge",
            hypothesis_statement="A may affect B through an unresolved mediator.",
            hypothesis_type="mechanistic_extension", premise_statement_ids=["s1"],
            gap_statement_ids=["g1"], inferential_bridge="Extend the reported association into a bounded mediator hypothesis.",
            predicted_observations=[PredictedObservationDraft(
                local_id="p", observable="B", expected_direction="qualitative_change",
                rationale="Tests whether the proposed mediator changes B.",
            )],
            falsification_criteria=[FalsificationCriterionDraft(
                local_id="f", observable="B", falsifying_outcome="No reproducible change in B under the relevant contrast.",
            )],
        )
    ])
    report, portfolio = compile_scientific_portfolio_materialization(
        context=_context(), pool=pool, selection=selection, draft=draft
    )
    assert report.materialized_hypothesis_count == 1
    assert portfolio.hypotheses[0].premise_statement_ids == ["s1"]
    assert report.external_literature_as_positive_premise is False
    assert report.production_selection_authority is False


def test_ineligible_premise_is_rejected_not_laundered():
    pool, selection = _pool_and_selection()
    draft = ScientificPortfolioMaterializationBatchDraft(items=[
        ScientificPortfolioMaterializationItemDraft(
            candidate_id="cand:1", title="Bad",
            hypothesis_statement="Unsupported premise.", hypothesis_type="mechanistic_extension",
            premise_statement_ids=["g1"], inferential_bridge="bad",
            predicted_observations=[PredictedObservationDraft(local_id="p", observable="B", expected_direction="unspecified", rationale="test")],
            falsification_criteria=[FalsificationCriterionDraft(local_id="f", observable="B", falsifying_outcome="no effect")],
        )
    ])
    report, portfolio = compile_scientific_portfolio_materialization(
        context=_context(), pool=pool, selection=selection, draft=draft
    )
    assert report.materialized_hypothesis_count == 0
    assert report.compile_rejected_count == 1
    assert portfolio.hypotheses == []


def test_materialization_draft_rejects_unmatched_falsifier_observable():
    with pytest.raises(ValueError):
        ScientificPortfolioMaterializationItemDraft(
            candidate_id="cand:1",
            title="Bad testability alignment",
            hypothesis_statement="A may affect B.",
            hypothesis_type="mechanistic_extension",
            premise_statement_ids=["s1"],
            inferential_bridge="bounded bridge",
            predicted_observations=[
                PredictedObservationDraft(
                    local_id="p",
                    observable="observable B",
                    expected_direction="qualitative_change",
                    rationale="test",
                )
            ],
            falsification_criteria=[
                FalsificationCriterionDraft(
                    local_id="f",
                    observable="unrelated observable C",
                    falsifying_outcome="no effect",
                )
            ],
        )


def test_materialization_generation_failure_preserves_exploration_selection():
    pool, selection = _pool_and_selection()
    report, portfolio = compile_scientific_portfolio_materialization(
        context=_context(),
        pool=pool,
        selection=selection,
        draft=ScientificPortfolioMaterializationBatchDraft(),
        generation_error="InstructorRetryException: structured output failed",
    )
    assert report.selected_candidate_count == 1
    assert report.materialized_hypothesis_count == 0
    assert report.generation_failed_count == 1
    assert report.status_counts == {"GENERATION_FAILED": 1}
    assert report.records[0].issue_codes == ["MATERIALIZATION_GENERATION_FAILED"]
    assert report.materialization_failure_is_not_exploration_rejection is True
    assert portfolio.hypotheses == []
