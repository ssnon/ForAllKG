from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.novelty_guided_discovery_closure import (
    bind_novelty_guided_regeneration_closure,
    build_novelty_guided_negative_space_plan,
)


def _statement(
    sid,
    *,
    eligible=True,
    restrictions=None,
):
    return SimpleNamespace(
        statement_id=sid,
        eligible_as_premise=eligible,
        premise_restrictions=list(restrictions or []),
    )


def _fixture(*, action="gap_sharpen"):
    context = SimpleNamespace(
        context_id="context:1",
        evidence_statements=[
            _statement("A"),
            _statement("B"),
            _statement("C"),
        ],
    )

    portfolio = SimpleNamespace(
        portfolio_id="portfolio:1",
        hypotheses=[
            SimpleNamespace(
                hypothesis_id="h1",
                premise_statement_ids=["A", "B"],
            ),
        ],
    )

    gap = SimpleNamespace(
        gap_id="gap:1",
        hypothesis_id="h1",
        source_external_status="NEW_COMBINATION_OF_KNOWN_EFFECTS",
        action=action,
        target_claim_ids=["claim:1"],
        differentiator="known relation needs a deeper boundary",
        already_known_boundary=["baseline relation"],
        unresolved_boundary=["second-order condition"],
        targeted_queries=[SimpleNamespace(query_text="q")],
        sharpening_operators=(
            ["MODERATOR", "BOUNDARY"]
            if action == "gap_sharpen"
            else []
        ),
        reason_codes=["fixture"],
    )

    gap_plan = SimpleNamespace(
        plan_id="gap_plan:1",
        gaps=[gap],
    )

    evidence_card = SimpleNamespace(
        hypothesis_id="h1",
        portfolio_unique_premise_statement_ids=[],
    )

    evidence = SimpleNamespace(
        report_id="evidence:1",
        source_portfolio_id="portfolio:1",
        unused_eligible_statement_ids=["C"],
        shared_core_statement_ids=["A", "B"],
        cards=[evidence_card],
    )

    return context, portfolio, gap_plan, evidence


def test_negative_space_plan_preserves_operator_and_evidence_authority():
    context, portfolio, gap_plan, evidence = _fixture()

    plan = build_novelty_guided_negative_space_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        evidence_diversity=evidence,
    )

    assert plan.target_count == 1
    assert plan.regeneration_eligible_count == 1
    assert plan.operator_target_count == 1
    assert plan.operator_opportunity_count == 2

    target = plan.targets[0]
    assert target.closure_mode == "SAME_PREMISE_OPERATOR_SHARPEN"
    assert target.operator_allocation.operators == [
        "MODERATOR",
        "BOUNDARY",
    ]
    assert (
        target.operator_allocation
        .operator_inferred_from_hypothesis_text
        is False
    )

    assert target.evidence_allocation.candidate_alternative_premise_ids == [
        "C"
    ]
    assert (
        target.evidence_allocation.alternative_relevance_established
        is False
    )
    assert target.production_selection_authority is False


def test_fresh_reaxis_with_fresh_external_check_closes_loop():
    context, portfolio, gap_plan, evidence = _fixture(
        action="targeted_search_then_refine"
    )

    plan = build_novelty_guided_negative_space_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        evidence_diversity=evidence,
    )

    attempt = SimpleNamespace(
        gap_id="gap:1",
        original_hypothesis_id="h1",
        decision="accepted_reaxis",
        generation_mode="fresh_context_reaxis",
        candidate_hypothesis_id="h2",
        final_hypothesis_id="h3",
        refinement_generated=True,
        context_grounding_valid=True,
        grounding_preserved=False,
        targeted_external_status="LITERATURE_SUPPORTED_EXTENSION",
        final_external_status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        post_generation_semantic_stable=True,
        post_generation_scientific_action="KEEP",
        post_generation_selection_class="CANDIDATE",
        reason_codes=["fresh_context_reaxis"],
    )

    report = SimpleNamespace(
        report_id="refinement:1",
        final_portfolio_id="portfolio:final",
        attempts=[attempt],
    )

    closure = bind_novelty_guided_regeneration_closure(
        negative_space_plan=plan,
        refinement_report=report,
    )

    assert closure.regeneration_attempted_count == 1
    assert closure.regeneration_generated_count == 1
    assert closure.accepted_regeneration_count == 1
    assert closure.fresh_external_verification_count == 1
    assert closure.closed_loop_observed_count == 1
    assert closure.traces[0].generation_mode == "fresh_context_reaxis"
    assert closure.traces[0].closed_loop_observed is True
    assert closure.production_selection_changed is False


def test_keep_is_observed_but_not_regeneration_eligible():
    context, portfolio, gap_plan, evidence = _fixture(
        action="keep"
    )

    plan = build_novelty_guided_negative_space_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        evidence_diversity=evidence,
    )

    assert plan.targets[0].closure_mode == "OBSERVE_ONLY"
    assert plan.targets[0].eligible_for_guided_regeneration is False
    assert plan.regeneration_eligible_count == 0
