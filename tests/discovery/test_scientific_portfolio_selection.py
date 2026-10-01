from types import SimpleNamespace

from pipeline_core.discovery.frontier_exploration_audit import FrontierExplorationAudit
from pipeline_core.discovery.frontier_idea_population import FrontierIdea, FrontierIdeaPopulation
from pipeline_core.discovery.idea_evolution import IdeaEvolutionIdea, IdeaEvolutionReport
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidateEvaluationDraft,
    ScientificPortfolioDimensionAssessment,
    ScientificPortfolioEvaluationBatchDraft,
    ScientificPortfolioNormalizedSketch,
    build_scientific_portfolio_candidate_pool,
    build_scientific_portfolio_selection,
    compile_scientific_portfolio_evaluation,
    evaluation_prompt_payload,
)


def _a(level):
    return ScientificPortfolioDimensionAssessment(level=level, rationale=f"{level} rationale")


def _sketch(label="bounded"):
    return ScientificPortfolioNormalizedSketch(
        hypothesis_frame=f"{label} hypothesis frame",
        predicted_observation=f"{label} predicted observation",
        falsification_condition=f"{label} falsification condition",
        discriminating_observation=f"{label} discriminating observation",
    )


def _frontier_idea(i, form="RELATION_AXIS", family="f", candidate=False):
    relation = SimpleNamespace(subject="A", relation="VARIES_WITH", object=f"B{i}") if form == "RELATION_AXIS" else None
    topology = None
    if form == "HIGHER_ORDER_TOPOLOGY":
        topology = SimpleNamespace(
            modifier_text=f"m{i}",
            backbone_relation_texts=["A --PROMOTES--> B"],
        )
    return FrontierIdea.model_construct(
        idea_id=f"idea:{i}",
        idea_form=form,
        source_kind="KG_AXIS" if i % 2 else "OPEN_WORLD_AXIS",
        source_context_id="ctx",
        source_context_sha256="csha",
        source_lineage=[SimpleNamespace(
            source_kind="KG_AXIS" if i % 2 else "OPEN_WORLD_AXIS",
            source_artifact="a.json",
            external_literature_lineage=not bool(i % 2),
            candidate_or_unverified_lineage=candidate,
        )],
        rendered_scientific_intent=f"intent {i}",
        task_relation_mode="SUBORDINATE",
        relation_signature=relation,
        topology_signature=topology,
        tension_signature=None,
        competing_explanation_signature=None,
        exact_scientific_signature=family,
    )


def test_projection_collapses_topology_variants_and_selection_preserves_family_diversity():
    ideas = [
        _frontier_idea(1, "RELATION_AXIS", "r1"),
        _frontier_idea(2, "RELATION_AXIS", "r2"),
        _frontier_idea(3, "HIGHER_ORDER_TOPOLOGY", "raw3", candidate=True),
        _frontier_idea(4, "HIGHER_ORDER_TOPOLOGY", "raw4"),
        _frontier_idea(5, "HIGHER_ORDER_TOPOLOGY", "raw5"),
    ]
    population = FrontierIdeaPopulation.model_construct(
        population_id="pop",
        population_sha256="psha",
        source_context_id="ctx",
        source_context_sha256="csha",
        research_question="Q",
        task_source="A",
        task_target="B",
        ideas=ideas,
        total_idea_count=len(ideas),
    )
    audit = FrontierExplorationAudit.model_construct(
        population_id="pop",
        population_sha256="psha",
        topology_layer=SimpleNamespace(
            backbone_families=[
                SimpleNamespace(family_id="backbone:one", topology_idea_ids=["idea:3", "idea:4", "idea:5"])
            ]
        ),
    )
    evo_idea = IdeaEvolutionIdea.model_construct(
        evolution_id="evo:1",
        operator_id="CROSS_SOURCE_BRIDGE",
        idea_form="CROSS_SOURCE_BRIDGE",
        title="bridge",
        scientific_intent="bridge intent",
        conceptual_change_summary="cross source",
        core_relations=["X --MODULATES--> Y"],
        differential_prediction="different response",
        falsification_condition="no difference",
        discriminating_observation="measure response",
        task_relation_mode="UNKNOWN",
        conceptual_family_signature="evofam",
        parent_source_kinds=["KG_AXIS", "OPEN_WORLD_AXIS"],
        inherited_external_literature_lineage=True,
        inherited_candidate_or_unverified_lineage=False,
        cross_source_composition=True,
        lineage_refs=[SimpleNamespace(source_artifact="e.json")],
    )
    evo = IdeaEvolutionReport.model_construct(
        report_id="evo-report",
        report_sha256="esha",
        source_population_id="pop",
        source_context_id="ctx",
        idea_count=1,
        ideas=[evo_idea],
    )
    pool = build_scientific_portfolio_candidate_pool(
        population=population,
        exploration_audit=audit,
        evolution_report=evo,
        max_candidate_topology_supplements=1,
    )
    # one backbone representative + one candidate supplement, not all 3 variants
    assert sum(x.idea_form == "HIGHER_ORDER_TOPOLOGY" for x in pool.candidates) == 2
    assert pool.raw_topology_variant_count_omitted_from_evaluation == 1

    evaluations = []
    for idx, candidate in enumerate(pool.candidates):
        high = "HIGH" if idx % 2 == 0 else "MODERATE"
        evaluations.append(
            ScientificPortfolioCandidateEvaluationDraft(
                candidate_id=candidate.candidate_id,
                normalized_sketch=_sketch(candidate.candidate_id),
                task_relevance=_a("HIGH"),
                mechanistic_coherence=_a(high),
                falsifiability=_a("HIGH"),
                discriminating_power=_a(high),
                operationalizability=_a("MODERATE"),
                information_gain=_a("HIGH"),
                overall_rationale="bounded assessment",
            )
        )
    report = compile_scientific_portfolio_evaluation(
        pool=pool,
        draft=ScientificPortfolioEvaluationBatchDraft(evaluations=evaluations),
        evaluator_model="test-model",
    )
    selection = build_scientific_portfolio_selection(
        pool=pool,
        evaluation=report,
        max_retained_candidates=4,
        max_retained_per_profile=1,
    )
    assert selection.single_scalar_score_used is False
    assert selection.overall_winner_selected is False
    assert len({x.conceptual_family_signature for x in selection.entries}) == selection.retained_count
    assert selection.production_selection_authority is False


def test_evaluation_prompt_is_origin_blind_and_requires_common_sketch():
    population = FrontierIdeaPopulation.model_construct(
        population_id="pop", population_sha256="psha", source_context_id="ctx",
        source_context_sha256="csha", research_question="Q", task_source="A", task_target="B",
        ideas=[_frontier_idea(1), _frontier_idea(2)], total_idea_count=2,
    )
    audit = FrontierExplorationAudit.model_construct(
        population_id="pop", population_sha256="psha",
        topology_layer=SimpleNamespace(backbone_families=[]),
    )
    evo = IdeaEvolutionReport.model_construct(
        report_id="e", report_sha256="esha", source_population_id="pop",
        source_context_id="ctx", idea_count=0, ideas=[],
    )
    pool = build_scientific_portfolio_candidate_pool(
        population=population, exploration_audit=audit, evolution_report=evo
    )
    payload = evaluation_prompt_payload(
        pool=pool,
        task_source="A",
        task_target="B",
    )
    assert payload.candidates
    forbidden = {
        "origin",
        "source_kind",
        "operator_id",
        "idea_form",
        "parent_source_kinds",
        "external_literature_lineage",
        "candidate_or_unverified_lineage",
        "cross_source_composition",
        "differential_prediction",
        "falsification_condition",
        "discriminating_observation",
    }
    assert all(not (forbidden & set(row)) for row in payload.candidates)


def test_evaluation_must_cover_pool_exactly():
    population = FrontierIdeaPopulation.model_construct(
        population_id="pop", population_sha256="psha", source_context_id="ctx",
        source_context_sha256="csha", research_question="Q", task_source="A", task_target="B",
        ideas=[_frontier_idea(1)], total_idea_count=1,
    )
    audit = FrontierExplorationAudit.model_construct(
        population_id="pop", population_sha256="psha",
        topology_layer=SimpleNamespace(backbone_families=[]),
    )
    evo = IdeaEvolutionReport.model_construct(
        report_id="e", report_sha256="esha", source_population_id="pop",
        source_context_id="ctx", idea_count=0, ideas=[],
    )
    pool = build_scientific_portfolio_candidate_pool(
        population=population, exploration_audit=audit, evolution_report=evo
    )
    try:
        compile_scientific_portfolio_evaluation(
            pool=pool,
            draft=ScientificPortfolioEvaluationBatchDraft(evaluations=[]),
            evaluator_model="test",
        )
    except ValueError as exc:
        assert "exactly one row" in str(exc)
    else:
        raise AssertionError("missing evaluation coverage must fail closed")
