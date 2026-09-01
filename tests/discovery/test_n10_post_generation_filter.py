from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.nonobviousness_post_generation import (
    filter_alpha6_portfolio_by_nonobviousness,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyRefinementReport,
    RefinementAttempt,
)


def _card(
    hypothesis_id,
):
    return HypothesisCard(
        hypothesis_id=hypothesis_id,
        domain_profile_id="sers_au_ag",
        source_context_id="context:c",
        source_context_sha256="ctxsha",
        source_report_id="report:r",
        source_report_sha256="rsha",
        title=hypothesis_id,
        hypothesis_statement="Test hypothesis.",
        hypothesis_type="context_dependency",
        premise_statement_ids=["statement:s"],
        gap_statement_ids=[],
        inferential_bridge="Bridge.",
        predicted_observations=[
            PredictedObservation(
                observation_id="prediction:p",
                observable="Observable",
                expected_direction="shift",
                rationale="Rationale",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:f",
                observable="Observable",
                falsifying_outcome="No shift.",
            )
        ],
        source_paper_ids=["paper:p"],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )


def _portfolio(
    ids,
):
    return HypothesisPortfolio(
        portfolio_id="portfolio:alpha6",
        domain_profile_id="sers_au_ag",
        source_context_id="context:c",
        source_context_sha256="ctxsha",
        source_report_id="report:r",
        source_report_sha256="rsha",
        hypotheses=[
            _card(row)
            for row in ids
        ],
    )


def _attempt(
    *,
    original,
    candidate,
    final,
    decision,
):
    action = (
        "keep"
        if decision == "kept_original"
        else "targeted_search_then_refine"
    )

    return RefinementAttempt(
        original_hypothesis_id=original,
        candidate_hypothesis_id=candidate,
        final_hypothesis_id=final,
        gap_id="gap:g",
        action=action,
        decision=decision,
        original_external_status="PLAUSIBLY_NOVEL",
        targeted_external_status="PLAUSIBLY_NOVEL",
        final_external_status="PLAUSIBLY_NOVEL",
        grounding_preserved=True,
        refinement_generated=(
            decision != "kept_original"
        ),
        interpretation="Synthetic control.",
    )


def _report(
    attempts,
):
    return NoveltyRefinementReport(
        report_id="refinement:r",
        report_sha256="sha",
        source_portfolio_id="portfolio:source",
        source_external_report_id="external:r",
        source_gap_plan_id="gapplan:g",
        final_portfolio_id="portfolio:alpha6",
        attempts=attempts,
        targeted_searches=[],
        accepted_refinement_count=sum(
            row.decision == "accepted_refinement"
            for row in attempts
        ),
        accepted_reaxis_count=sum(
            row.decision == "accepted_reaxis"
            for row in attempts
        ),
        kept_original_count=sum(
            row.decision == "kept_original"
            for row in attempts
        ),
        rejected_count=0,
        max_refinements_per_hypothesis=1,
        max_reaxes_per_hypothesis=1,
        external_prior_art_can_be_positive_premise=False,
        policy_version="novelty-refinement-policy-v2",
    )


def _gate(
    candidate_id,
    *,
    allowed,
):
    return {
        "schema_version":
            "scientific-novelty-fallback-gate-v1",
        "production_authority":
            True,
        "authority_source":
            "n10_nonobviousness",
        "gates": [
            {
                "hypothesis_id":
                    candidate_id,
                "fallback_allowed":
                    allowed,
                "selection_class":
                    (
                        "ELIGIBLE"
                        if allowed
                        else "INELIGIBLE"
                    ),
                "action":
                    (
                        "KEEP_NONOBVIOUS_CANDIDATE"
                        if allowed
                        else "RESOLVE_NONOBVIOUSNESS_EVIDENCE"
                    ),
                "reason_codes":
                    [],
            }
        ],
    }


def test_kept_original_does_not_need_second_n10_gate():
    portfolio = _portfolio(
        ["hypothesis:final"]
    )

    report = _report(
        [
            _attempt(
                original="hypothesis:orig",
                candidate="hypothesis:orig",
                final="hypothesis:final",
                decision="kept_original",
            )
        ]
    )

    filtered, audit = (
        filter_alpha6_portfolio_by_nonobviousness(
            portfolio=portfolio,
            refinement_report=report,
            gates_by_candidate_id={},
        )
    )

    assert len(
        filtered.hypotheses
    ) == 1

    assert (
        audit["decisions"][0][
            "n10_required"
        ]
        is False
    )


def test_eligible_refinement_survives_fresh_n10():
    portfolio = _portfolio(
        ["hypothesis:final"]
    )

    report = _report(
        [
            _attempt(
                original="hypothesis:orig",
                candidate="hypothesis:candidate",
                final="hypothesis:final",
                decision="accepted_refinement",
            )
        ]
    )

    filtered, audit = (
        filter_alpha6_portfolio_by_nonobviousness(
            portfolio=portfolio,
            refinement_report=report,
            gates_by_candidate_id={
                "hypothesis:candidate":
                    _gate(
                        "hypothesis:candidate",
                        allowed=True,
                    )
            },
        )
    )

    assert [
        row.hypothesis_id
        for row in filtered.hypotheses
    ] == [
        "hypothesis:final"
    ]

    assert (
        audit["removed_by_post_generation_n10_count"]
        == 0
    )


def test_ineligible_reaxis_is_removed():
    portfolio = _portfolio(
        ["hypothesis:final"]
    )

    report = _report(
        [
            _attempt(
                original="hypothesis:orig",
                candidate="hypothesis:candidate",
                final="hypothesis:final",
                decision="accepted_reaxis",
            )
        ]
    )

    filtered, audit = (
        filter_alpha6_portfolio_by_nonobviousness(
            portfolio=portfolio,
            refinement_report=report,
            gates_by_candidate_id={
                "hypothesis:candidate":
                    _gate(
                        "hypothesis:candidate",
                        allowed=False,
                    )
            },
        )
    )

    assert filtered.hypotheses == []

    assert (
        "fresh N10 post-generation"
        in filtered.abstention_reason
    )

    assert (
        audit["removed_by_post_generation_n10_count"]
        == 1
    )


def test_generated_survivor_missing_fresh_gate_fails_closed():
    portfolio = _portfolio(
        ["hypothesis:final"]
    )

    report = _report(
        [
            _attempt(
                original="hypothesis:orig",
                candidate="hypothesis:candidate",
                final="hypothesis:final",
                decision="accepted_refinement",
            )
        ]
    )

    try:
        filter_alpha6_portfolio_by_nonobviousness(
            portfolio=portfolio,
            refinement_report=report,
            gates_by_candidate_id={},
        )
    except ValueError as exc:
        assert (
            "missing fresh N10 gate"
            in str(exc)
        )
    else:
        raise AssertionError(
            "missing gate must fail closed"
        )


def test_mixed_original_and_generated_candidate_filter_independently():
    portfolio = _portfolio(
        [
            "hypothesis:final-original",
            "hypothesis:final-generated",
        ]
    )

    report = _report(
        [
            _attempt(
                original="hypothesis:o1",
                candidate="hypothesis:o1",
                final="hypothesis:final-original",
                decision="kept_original",
            ),
            _attempt(
                original="hypothesis:o2",
                candidate="hypothesis:c2",
                final="hypothesis:final-generated",
                decision="accepted_refinement",
            ),
        ]
    )

    filtered, _audit = (
        filter_alpha6_portfolio_by_nonobviousness(
            portfolio=portfolio,
            refinement_report=report,
            gates_by_candidate_id={
                "hypothesis:c2":
                    _gate(
                        "hypothesis:c2",
                        allowed=False,
                    )
            },
        )
    )

    assert [
        row.hypothesis_id
        for row in filtered.hypotheses
    ] == [
        "hypothesis:final-original"
    ]
