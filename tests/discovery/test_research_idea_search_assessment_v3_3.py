from __future__ import annotations

from pipeline_core.discovery.research_idea_search_assessment import (
    ResearchIdeaNoveltySignal,
    ResearchIdeaSearchAssessmentItem,
    _decision_for,
)


def _novelty(state: str) -> ResearchIdeaNoveltySignal:
    return ResearchIdeaNoveltySignal(
        idea_id="idea:1",
        novelty_state=state,
        direct_prior_art_claim_count=0,
        partial_prior_art_claim_count=0,
        residual_claim_count=0,
        unresolved_claim_count=0,
        source_mode=(
            "NONE"
            if state == "NOT_ASSESSED"
            else "RETROSPECTIVE_TERMINAL_EXTERNAL_NOVELTY"
        ),
    )


def _row(
    *,
    novelty: str,
    usable: int = 1,
    exhausted: bool = True,
    family_size: int = 1,
    remains_active: bool = True,
    debts: list[str] | None = None,
    discrimination_complete: bool = True,
) -> ResearchIdeaSearchAssessmentItem:
    debt_ids = debts or []
    return ResearchIdeaSearchAssessmentItem(
        idea_id="idea:1",
        idea_birth_generation_index=0,
        cycle_generation_index=2,
        remains_active=remains_active,
        realization_count=2,
        usable_grounded_realization_count=usable,
        failed_or_abstained_count=0,
        not_operationalizable_count=0,
        local_search_budget=2,
        local_search_budget_used=2 if exhausted else 1,
        local_search_exhausted=exhausted,
        active_maturity_counts={"STRICT_GROUNDED": usable} if usable else {"PARTIALLY_GROUNDED": 1},
        active_grounded_member_count=usable,
        active_non_grounded_member_count=0 if usable else 1,
        speculative_member_count=0,
        epistemic_debt_ids=debt_ids,
        epistemic_debt_kinds=["RELATION_OR_MECHANISM_EVIDENCE"] if debt_ids else [],
        acquisition_escalation_eligible_debt_count=0,
        program_family_key="program:1",
        program_family_size=family_size,
        tight_neighborhood_key="tight:1",
        tight_neighborhood_size=family_size,
        family_pressure=(
            "ISOLATED" if family_size == 1 else "REPRESENTED" if family_size == 2 else "DENSE"
        ),
        differential_prediction_present=discrimination_complete,
        falsification_condition_present=discrimination_complete,
        discriminating_observation_present=discrimination_complete,
        discrimination_sketch_complete=discrimination_complete,
        novelty=_novelty(novelty),
    )


def test_grounded_without_prior_art_is_probe_not_hold():
    decision = _decision_for(_row(novelty="NOT_ASSESSED"))
    assert decision.persistence == "RETAIN"
    assert decision.search_action == "PROBE"
    assert not decision.search_complete_candidate
    assert "GROUNDED_OR_REALIZABLE_DOES_NOT_IMPLY_SEARCH_COMPLETE" in decision.reason_codes


def test_residual_novelty_reopens_grounded_idea_for_evolution():
    decision = _decision_for(_row(novelty="RESIDUAL", family_size=1))
    assert decision.persistence == "RETAIN"
    assert decision.search_action == "EVOLVE"
    assert "NOVELTY_RESIDUE_REMAINS" in decision.reason_codes


def test_dense_residual_family_diversifies_instead_of_local_variant_proliferation():
    decision = _decision_for(_row(novelty="RESIDUAL", family_size=4))
    assert decision.search_action == "DIVERSIFY"
    assert "DIVERSIFY_BEYOND_LOCAL_VARIANTS" in decision.reason_codes


def test_saturated_grounded_program_is_not_search_complete_by_default():
    decision = _decision_for(_row(novelty="SATURATED", family_size=2))
    assert decision.search_action == "DIVERSIFY"
    assert "CURRENT_PROGRAM_PRIOR_ART_SATURATED" in decision.reason_codes


def test_strong_residue_can_be_search_complete_candidate_without_debt():
    decision = _decision_for(
        _row(
            novelty="STRONG_RESIDUE",
            usable=1,
            family_size=1,
            debts=[],
            discrimination_complete=True,
        )
    )
    assert decision.persistence == "RETAIN"
    assert decision.search_action == "NONE"
    assert decision.search_complete_candidate


def test_strong_residue_with_evidence_debt_still_evolves():
    decision = _decision_for(
        _row(
            novelty="STRONG_RESIDUE",
            debts=["debt:1"],
            discrimination_complete=True,
        )
    )
    assert decision.search_action == "EVOLVE"
    assert not decision.search_complete_candidate


def test_ungrounded_open_local_search_realizes_before_mutating():
    decision = _decision_for(
        _row(
            novelty="RESIDUAL",
            usable=0,
            exhausted=False,
        )
    )
    assert decision.search_action == "REALIZE"


def test_inactive_idea_archives_without_action():
    decision = _decision_for(
        _row(
            novelty="RESIDUAL",
            remains_active=False,
        )
    )
    assert decision.persistence == "ARCHIVE"
    assert decision.search_action == "NONE"
