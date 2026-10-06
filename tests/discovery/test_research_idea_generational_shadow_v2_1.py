from types import SimpleNamespace as NS

from pipeline_core.discovery.research_idea_feedback import (
    compile_idea_feedback_directives,
    compile_idea_outcome_observations,
)
from pipeline_core.discovery.research_idea_generational_shadow import (
    build_generational_idea_search_shadow,
)
from pipeline_core.discovery.research_idea_projection import (
    project_current_idea_artifacts,
)


CONTEXT_ID = "ctx:test"
CONTEXT_SHA = "c" * 64


def _lineage():
    return NS(source_artifact="frontier.json")


def _frontier(idea_id: str, subject: str, relation: str, obj: str):
    return NS(
        idea_id=idea_id,
        idea_form="RELATION_AXIS",
        source_kind="KG_AXIS",
        source_context_id=CONTEXT_ID,
        source_context_sha256=CONTEXT_SHA,
        source_lineage=[_lineage()],
        rendered_scientific_intent=f"{subject} {relation} {obj}",
        task_relation_mode="DIRECT",
        relation_signature=NS(
            subject=subject,
            relation=relation,
            object=obj,
            candidate_unit_id="",
        ),
        topology_signature=None,
        tension_signature=None,
        competing_explanation_signature=None,
        exact_scientific_signature=f"legacy:{idea_id}",
    )


def _evolution(
    evolution_id: str,
    *,
    parent_ids: list[str],
    core: list[str],
    operator_id: str = "BACKBONE_MUTATION",
):
    return NS(
        evolution_id=evolution_id,
        operator_id=operator_id,
        idea_form=(
            "SCIENTIFIC_REFRAME"
            if operator_id == "REGIME_BOUNDARY"
            else "MUTATED_TOPOLOGY"
        ),
        source_context_id=CONTEXT_ID,
        source_context_sha256=CONTEXT_SHA,
        parent_idea_ids=parent_ids,
        parent_source_kinds=["KG_AXIS"] if parent_ids else [],
        lineage_refs=[NS(source_artifact="evolution.json")],
        title=f"title {evolution_id}",
        scientific_intent="; ".join(core),
        conceptual_change_summary="test",
        core_relations=core,
        transformed_question=(
            "Does the regime change the mechanism?"
            if operator_id == "REGIME_BOUNDARY"
            else None
        ),
        mutation_kind=(
            "MECHANISM_INSERTION"
            if operator_id == "BACKBONE_MUTATION"
            else None
        ),
        challenged_assumption=None,
        alternative_explanations=[],
        differential_prediction="different response",
        falsification_condition="no difference",
        discriminating_observation="measure response",
        task_relation_mode=(
            "REFRAME" if operator_id == "REGIME_BOUNDARY" else "UNKNOWN"
        ),
        conceptual_family_signature=f"legacy-family:{evolution_id}",
    )


def _candidate(candidate_id: str, source_object_id: str, family: str):
    return NS(
        candidate_id=candidate_id,
        source_object_id=source_object_id,
        conceptual_family_signature=family,
    )


def _assessment(level: str):
    return NS(level=level)


def _evaluation(candidate_id: str, *, profiles=None, burden="MODERATE"):
    return NS(
        candidate_id=candidate_id,
        task_relevance=_assessment("HIGH"),
        mechanistic_coherence=_assessment("HIGH"),
        falsifiability=_assessment("HIGH"),
        discriminating_power=_assessment("HIGH"),
        operationalizability=_assessment("HIGH"),
        information_gain=_assessment("HIGH"),
        eligible_profiles=list(profiles or ["TASK_NEAR_VALIDATION"]),
        verification_burden=burden,
    )


def _fixture():
    frontier = [
        _frontier("f1", "A", "CAUSES", "B"),
        _frontier("f2", "C", "MODULATES", "D"),
        # Exact scientific-kernel duplicate from a distinct source object.
        _frontier("f3", "C", "MODULATES", "D"),
    ]
    evolutions = [
        _evolution(
            "e1",
            parent_ids=["f1"],
            core=["A --CAUSES--> M", "M --CAUSES--> B"],
        ),
        # Mutation operator that is actually a semantic no-op.
        _evolution(
            "e2",
            parent_ids=["f2"],
            core=["C --MODULATES--> D"],
        ),
        _evolution(
            "e3",
            parent_ids=[],
            core=["C --MODULATES--> D under regime R"],
            operator_id="REGIME_BOUNDARY",
        ),
    ]
    population = NS(
        population_id="pop:1",
        population_sha256="p" * 64,
        source_context_id=CONTEXT_ID,
        source_context_sha256=CONTEXT_SHA,
        research_question="How do the mechanisms vary?",
        task_source="A",
        task_target="B",
        ideas=frontier,
    )
    evolution_report = NS(
        report_id="evo-report:1",
        report_sha256="e" * 64,
        source_population_id="pop:1",
        source_context_id=CONTEXT_ID,
        source_context_sha256=CONTEXT_SHA,
        ideas=evolutions,
    )

    candidates = [
        _candidate("c:f1", "f1", "legacy-family:shared"),
        _candidate("c:f2", "f2", "legacy-family:duplicate"),
        _candidate("c:f3", "f3", "legacy-family:duplicate"),
        # Deliberately shares legacy family with f1. v2.1 must not use this as
        # a hard family gate because the kernels differ.
        _candidate("c:e1", "e1", "legacy-family:shared"),
        _candidate("c:e2", "e2", "legacy-family:e2"),
        _candidate("c:e3", "e3", "legacy-family:e3"),
    ]
    pool = NS(
        pool_id="pool:1",
        pool_sha256="q" * 64,
        source_population_id="pop:1",
        source_evolution_report_id="evo-report:1",
        candidates=candidates,
    )
    evaluation = NS(
        report_id="eval:1",
        report_sha256="v" * 64,
        source_pool_id="pool:1",
        evaluations=[
            _evaluation("c:f1"),
            _evaluation("c:f2"),
            _evaluation("c:f3"),
            _evaluation("c:e1"),
            _evaluation("c:e2"),
            _evaluation("c:e3", profiles=["REFRAME"]),
        ],
    )
    selection = NS(
        selection_id="selection:1",
        selection_sha256="s" * 64,
        source_pool_id="pool:1",
        source_evaluation_report_id="eval:1",
        retained_candidate_ids=["c:e1", "c:f2", "c:f3"],
    )
    materialization = NS(
        report_id="materialization:1",
        source_selection_id="selection:1",
        records=[
            NS(
                candidate_id="c:e1",
                hypothesis_id="h:e1",
                status="MATERIALIZED",
                issue_codes=[],
            ),
            NS(
                candidate_id="c:f2",
                hypothesis_id="h:f2",
                status="MATERIALIZED",
                issue_codes=[],
            ),
            NS(
                candidate_id="c:f3",
                hypothesis_id=None,
                status="ABSTAINED",
                issue_codes=["MODEL_ABSTAINED"],
            ),
        ],
    )
    residual = {
        "report_id": "residual:1",
        "hypotheses": [
            {
                "hypothesis_id": "h:e1",
                "final_epistemic_state": "RESIDUAL_AUTHORITY_CANDIDATE_SHADOW",
                "state_reason": "novelty_bearing_residual_survives",
                "fresh_external_status": "INSUFFICIENT_SEARCH_EVIDENCE",
            },
            {
                "hypothesis_id": "h:f2",
                "final_epistemic_state": "PRIOR_ART_BACKED_OR_NO_RESIDUAL",
                "state_reason": "novelty_bearing_relation_prior_art_backed",
                "fresh_external_status": "WELL_ESTABLISHED",
            },
        ],
    }
    prospective = {
        "h:e1": {
            "source_hypothesis_id": "h:e1",
            "prospective_identifiability": "PROSPECTIVELY_IDENTIFIABLE",
            "current_evidence_status": "PARTIAL_GROUNDING",
            "directionality_mode": "GROUNDED_PREDICTION",
            "measurement_compatibility_mode": "PROSPECTIVE_MATCH_REQUIRED",
            "contract_integrity_passed": True,
        },
        "h:f2": {
            "source_hypothesis_id": "h:f2",
            "prospective_identifiability": "NOT_OPERATIONALIZABLE",
            "current_evidence_status": "CONTEXT_ONLY",
            "directionality_mode": "NOT_IDENTIFIABLE",
            "measurement_compatibility_mode": "NOT_COMPARABLE",
            "contract_integrity_passed": True,
        },
    }
    return (
        population,
        evolution_report,
        pool,
        evaluation,
        selection,
        materialization,
        residual,
        prospective,
    )


def test_feedback_observation_is_policy_free_and_candidate_maps_to_idea():
    population, evolution, pool, _, _, materialization, residual, prospective = _fixture()
    nodes = project_current_idea_artifacts(
        population=population,
        evolution_report=evolution,
    )
    observations = compile_idea_outcome_observations(
        pool=pool,
        materialization=materialization,
        nodes=nodes,
        residual_state=residual,
        prospective_by_hypothesis=prospective,
    )
    assert len(observations) == 3
    row = next(x for x in observations if x.hypothesis_id == "h:e1")
    assert row.residual_epistemic_state == "RESIDUAL_AUTHORITY_CANDIDATE_SHADOW"
    assert row.prospective_identifiability == "PROSPECTIVELY_IDENTIFIABLE"
    assert row.observation_only is True
    assert row.search_policy_authority is False
    assert not hasattr(row, "preferred_actions")


def test_not_operationalizable_realization_does_not_kill_research_idea():
    population, evolution, pool, _, _, materialization, residual, prospective = _fixture()
    nodes = project_current_idea_artifacts(
        population=population,
        evolution_report=evolution,
    )
    observations = compile_idea_outcome_observations(
        pool=pool,
        materialization=materialization,
        nodes=nodes,
        residual_state=residual,
        prospective_by_hypothesis=prospective,
    )
    directives = compile_idea_feedback_directives(
        observations,
        all_idea_ids=[x.idea_id for x in nodes],
    )
    observation = next(x for x in observations if x.hypothesis_id == "h:f2")
    directive = next(x for x in directives if x.idea_id == observation.idea_id)
    assert directive.direct_realization_reproductive is False
    assert directive.idea_reproductive is True
    assert "NOT_OPERATIONALIZABLE_REALIZATION_DOES_NOT_KILL_IDEA" in directive.reason_codes
    assert "AXIS_MUTATION" in directive.preferred_actions


def test_materialization_abstention_preserves_idea_reproduction():
    population, evolution, pool, _, _, materialization, residual, prospective = _fixture()
    nodes = project_current_idea_artifacts(
        population=population,
        evolution_report=evolution,
    )
    observations = compile_idea_outcome_observations(
        pool=pool,
        materialization=materialization,
        nodes=nodes,
        residual_state=residual,
        prospective_by_hypothesis=prospective,
    )
    directives = compile_idea_feedback_directives(observations)
    abstained = next(x for x in observations if x.materialization_status == "ABSTAINED")
    directive = next(x for x in directives if x.idea_id == abstained.idea_id)
    assert directive.direct_realization_reproductive is False
    assert directive.idea_reproductive is True
    assert "ALTERNATE_REALIZATION" in directive.preferred_actions
    assert "MATERIALIZATION_FAILURE_DOES_NOT_REJECT_IDEA" in directive.reason_codes


def test_integrated_shadow_audits_transitions_and_allocates_g2_parents():
    args = _fixture()
    report = build_generational_idea_search_shadow(
        population=args[0],
        evolution_report=args[1],
        pool=args[2],
        evaluation=args[3],
        selection=args[4],
        materialization=args[5],
        residual_state=args[6],
        prospective_by_hypothesis=args[7],
        max_g2_parents=4,
        max_g2_per_profile=4,
    )
    assert report.generation0_idea_count == 3
    assert report.generation1_idea_count == 3
    assert report.transition_count == 3
    assert report.identity_relation_counts["SAME_IDEA"] == 1
    assert report.semantic_noop_count == 1
    assert report.genealogy_relation_counts["IMPORTED_ROOT"] == 1
    assert report.observation_count == 3
    assert report.g2_parent_count == 4
    assert report.search_parent_selection_authority is True
    assert report.scientific_truth_authority is False
    assert report.production_selection_authority is False
    assert report.new_llm_calls is False


def test_g2_suppresses_only_exact_kernel_duplicates_not_legacy_family():
    args = _fixture()
    report = build_generational_idea_search_shadow(
        population=args[0],
        evolution_report=args[1],
        pool=args[2],
        evaluation=args[3],
        selection=args[4],
        materialization=args[5],
        residual_state=args[6],
        prospective_by_hypothesis=args[7],
        max_g2_parents=5,
        max_g2_per_profile=5,
    )
    selected = [x for x in report.candidate_states if x.selected_for_g2]
    assert selected

    # f2 and f3 have the same exact ResearchIdea kernel, so at most one can be
    # allocated for abstract idea search.
    node_by_source = {
        node.source_object_id: node
        for node in project_current_idea_artifacts(
            population=args[0], evolution_report=args[1]
        )
    }
    selected_idea_ids = {x.idea_id for x in selected}
    duplicate_ids = {node_by_source["f2"].idea_id, node_by_source["f3"].idea_id}
    assert len(selected_idea_ids & duplicate_ids) <= 1

    # f1 and e1 deliberately share the legacy family signature but are distinct
    # scientific kernels. Both remain eligible; no legacy family hard gate exists.
    shared_family_selected = [
        row
        for row in selected
        if "legacy-family:shared" in row.legacy_family_signatures
    ]
    assert len(shared_family_selected) == 2
    assert report.soft_family_not_used_as_hard_gate is True


def test_search_state_counts_observed_work_without_claiming_llm_compute():
    args = _fixture()
    report = build_generational_idea_search_shadow(
        population=args[0],
        evolution_report=args[1],
        pool=args[2],
        evaluation=args[3],
        selection=args[4],
        materialization=args[5],
        residual_state=args[6],
        prospective_by_hypothesis=args[7],
        max_g2_parents=3,
        max_g2_per_profile=3,
    )
    observed_ids = {x.idea_id for x in report.directives if x.observation_ids}
    assert observed_ids
    for state in report.search_states:
        if state.idea_id in observed_ids:
            assert state.materialization_attempt_count >= 1
        assert state.compute_spent.llm_calls == 0
        assert state.compute_spent.retrieval_calls == 0


def test_report_has_three_generation_snapshots_with_g2_as_parent_allocation_only():
    args = _fixture()
    report = build_generational_idea_search_shadow(
        population=args[0],
        evolution_report=args[1],
        pool=args[2],
        evaluation=args[3],
        selection=args[4],
        materialization=args[5],
        max_g2_parents=2,
        max_g2_per_profile=2,
    )
    assert [x.generation_index for x in report.generation_states] == [0, 1, 2]
    g2 = report.generation_states[2]
    assert g2.population_idea_ids == []
    assert g2.selected_parent_count == 2
    assert g2.selected_parent_idea_ids == report.g2_parent_idea_ids
    assert report.existing_frontier_or_evolution_mutated is False


def test_prospective_summary_rows_envelope_is_consumed():
    population, evolution, pool, _, _, materialization, residual, prospective = _fixture()
    nodes = project_current_idea_artifacts(
        population=population,
        evolution_report=evolution,
    )
    summary = {
        "schema_version": "prospective-identification-materialization-shadow-summary-v1",
        "rows": list(prospective.values()),
    }
    observations = compile_idea_outcome_observations(
        pool=pool,
        materialization=materialization,
        nodes=nodes,
        residual_state=residual,
        prospective_by_hypothesis=summary,
    )
    assert {
        row.hypothesis_id: row.prospective_identifiability
        for row in observations
        if row.hypothesis_id
    }["h:e1"] == "PROSPECTIVELY_IDENTIFIABLE"


def test_residual_portfolio_lineage_mismatch_fails_closed():
    import pytest

    population, evolution, pool, _, _, materialization, _, _ = _fixture()
    materialization.output_portfolio_id = "portfolio:expected"
    nodes = project_current_idea_artifacts(
        population=population,
        evolution_report=evolution,
    )
    with pytest.raises(ValueError, match="lineage mismatch"):
        compile_idea_outcome_observations(
            pool=pool,
            materialization=materialization,
            nodes=nodes,
            residual_state={
                "source_portfolio_id": "portfolio:wrong",
                "hypotheses": [],
            },
        )


def _feedback_selection_fixture():
    frontier = [
        _frontier("fa", "A", "CAUSES", "B"),
        _frontier("fb", "C", "CAUSES", "D"),
    ]
    population = NS(
        population_id="pop:feedback",
        population_sha256="p" * 64,
        source_context_id=CONTEXT_ID,
        source_context_sha256=CONTEXT_SHA,
        research_question="Q",
        task_source="A",
        task_target="B",
        ideas=frontier,
    )
    evolution = NS(
        report_id="evo:feedback",
        report_sha256="e" * 64,
        source_population_id="pop:feedback",
        source_context_id=CONTEXT_ID,
        source_context_sha256=CONTEXT_SHA,
        ideas=[],
    )
    pool = NS(
        pool_id="pool:feedback",
        pool_sha256="q" * 64,
        source_population_id="pop:feedback",
        source_evolution_report_id="evo:feedback",
        candidates=[
            _candidate("ca", "fa", "legacy:a"),
            _candidate("cb", "fb", "legacy:b"),
        ],
    )
    strong = _evaluation("ca", profiles=["TASK_NEAR_VALIDATION"])
    weak = NS(
        candidate_id="cb",
        task_relevance=_assessment("MODERATE"),
        mechanistic_coherence=_assessment("MODERATE"),
        falsifiability=_assessment("MODERATE"),
        discriminating_power=_assessment("MODERATE"),
        operationalizability=_assessment("MODERATE"),
        information_gain=_assessment("MODERATE"),
        eligible_profiles=["TASK_NEAR_VALIDATION"],
        verification_burden="MODERATE",
    )
    evaluation = NS(
        report_id="eval:feedback",
        report_sha256="v" * 64,
        source_pool_id="pool:feedback",
        evaluations=[strong, weak],
    )
    selection = NS(
        selection_id="selection:feedback",
        selection_sha256="s" * 64,
        source_pool_id="pool:feedback",
        source_evaluation_report_id="eval:feedback",
        retained_candidate_ids=["ca", "cb"],
    )
    materialization = NS(
        report_id="materialization:feedback",
        source_selection_id="selection:feedback",
        output_portfolio_id="portfolio:feedback",
        source_context_id=CONTEXT_ID,
        records=[
            NS(candidate_id="ca", hypothesis_id="ha", status="MATERIALIZED", issue_codes=[]),
            NS(candidate_id="cb", hypothesis_id="hb", status="MATERIALIZED", issue_codes=[]),
        ],
    )
    residual = {
        "schema_version": "scientific-portfolio-residual-epistemic-state-v1",
        "report_id": "residual:feedback",
        "source_portfolio_id": "portfolio:feedback",
        "hypotheses": [
            {
                "hypothesis_id": "ha",
                "final_epistemic_state": "PRIOR_ART_BACKED_OR_NO_RESIDUAL",
                "state_reason": "known",
            },
            {
                "hypothesis_id": "hb",
                "final_epistemic_state": "RESIDUAL_AUTHORITY_CANDIDATE_SHADOW",
                "state_reason": "survived",
            },
        ],
    }
    return population, evolution, pool, evaluation, selection, materialization, residual


def test_feedback_priority_can_change_g2_parent_without_scalar_score():
    args = _feedback_selection_fixture()
    report = build_generational_idea_search_shadow(
        population=args[0],
        evolution_report=args[1],
        pool=args[2],
        evaluation=args[3],
        selection=args[4],
        materialization=args[5],
        residual_state=args[6],
        max_g2_parents=1,
        max_g2_per_profile=1,
    )
    assert report.baseline_g2_parent_count == 1
    assert report.g2_parent_count == 1
    assert report.feedback_changed_parent_set is True
    assert report.feedback_parent_added_idea_ids
    assert report.feedback_parent_dropped_idea_ids
    assert report.baseline_priority_band_counts == {"MEDIUM": 2}
    assert report.feedback_priority_band_counts == {"HIGH": 1, "LOW": 1}
    assert report.single_scalar_search_score_used is False
    assert report.feedback_coverage.residual_coverage_fraction == 1.0
