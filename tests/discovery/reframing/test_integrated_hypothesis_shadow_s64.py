from __future__ import annotations

import pytest

from pipeline_core.discovery.reframing.cross_lane_synthesis import (
    CrossLaneSynthesisShadowReport,
    CrossLaneSynthesizedHypothesis,
    SynthesizedDiscriminatingTest,
    SynthesizedFalsifier,
    SynthesizedPrediction,
)
from pipeline_core.discovery.reframing.integrated_hypothesis_shadow import (
    build_integrated_scientific_hypothesis_shadow,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionCandidateFalsifier,
    ProductionCandidateLineageValidation,
    ProductionCandidatePrediction,
    ProductionFacingScientificCandidatePortfolio,
    ProductionScientificCandidate,
)


def _rel(candidate_id: str = "pc:rel") -> ProductionScientificCandidate:
    return ProductionScientificCandidate(
        candidate_id=candidate_id,
        source_entry_id=f"entry:{candidate_id}",
        source_lane="RELATIONAL_DISCOVERY",
        source_object_kind="relational_hypothesis",
        source_schema_version="hypothesis-card-v1",
        source_object_id=f"hypothesis:{candidate_id}",
        source_epistemic_status="grounded",
        title="Matrix changes design reliability",
        reasoning_label="descriptor_interaction",
        scientific_proposal="Matrix composition conditionally changes design reliability.",
        reasoning_rationale="Observed matrix and reliability relations motivate the interaction.",
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        assumptions=["matched analyte concentration"],
        predictions=[
            ProductionCandidatePrediction(
                observable="replicate variance",
                expected_direction="qualitative_change",
                rationale="variance should depend on matrix",
            )
        ],
        falsifiers=[
            ProductionCandidateFalsifier(
                observable="replicate variance",
                falsifying_outcome="variance is invariant across matrices",
            )
        ],
    )


def _ref(candidate_id: str = "pc:ref") -> ProductionScientificCandidate:
    return ProductionScientificCandidate(
        candidate_id=candidate_id,
        source_entry_id=f"entry:{candidate_id}",
        source_lane="SCIENTIFIC_REFRAMING",
        source_object_kind="scientific_reframe",
        source_schema_version="scientific-reframe-candidate-v1",
        source_object_id=f"scientific_reframe:{candidate_id}",
        source_epistemic_status="grounded_shadow",
        title="Interfacial state is latent",
        reasoning_label="LATENT_VARIABLE",
        scientific_proposal="Interfacial state mediates matrix response.",
        reasoning_rationale="Bulk matrix identity may not capture local exposure.",
        premise_statement_ids=["p2", "p3"],
        gap_statement_ids=["g1"],
        assumptions=["interfacial state is measurable"],
        predictions=[
            ProductionCandidatePrediction(
                observable="interfacial accumulation",
                baseline_expectation="tracks matrix identity only",
                alternative_expectation="varies by design within matrix",
                discriminating_outcome="local accumulation explains residual performance",
            )
        ],
        falsifiers=[
            ProductionCandidateFalsifier(
                falsifying_outcome="local state adds no explanatory value"
            )
        ],
    )


def _portfolio(*candidates: ProductionScientificCandidate) -> ProductionFacingScientificCandidatePortfolio:
    rows = list(candidates) or [_rel(), _ref()]
    rel = sum(row.source_lane == "RELATIONAL_DISCOVERY" for row in rows)
    kinds: dict[str, int] = {}
    for row in rows:
        kinds[row.source_object_kind] = kinds.get(row.source_object_kind, 0) + 1
    return ProductionFacingScientificCandidatePortfolio(
        portfolio_id="production_facing:1",
        source_cross_lane_portfolio_id="cross:1",
        source_cross_lane_portfolio_file_sha256="a" * 64,
        source_task_id="task:1",
        source_context_id="ctx:1",
        source_context_sha256="c" * 64,
        question="Does matrix alter design ranking?",
        domain_profile_id="sers",
        candidates=rows,
        lineage_validation=ProductionCandidateLineageValidation(),
        candidate_count=len(rows),
        relational_candidate_count=rel,
        reframing_candidate_count=len(rows) - rel,
        source_kind_counts=kinds,
    )


def _synthesized() -> CrossLaneSynthesizedHypothesis:
    return CrossLaneSynthesizedHypothesis(
        hypothesis_id="synth:1",
        source_task_id="task:1",
        source_context_id="ctx:1",
        source_candidate_portfolio_id="production_facing:1",
        source_candidate_ids=["pc:rel", "pc:ref"],
        source_object_ids=["hypothesis:pc:rel", "scientific_reframe:pc:ref"],
        source_lanes=["RELATIONAL_DISCOVERY", "SCIENTIFIC_REFRAMING"],
        synthesis_kind="complementary_mechanism_integration",
        title="Interfacial state mediates matrix-conditioned ranking",
        hypothesis_statement="Matrix-specific interfacial state mediates design-rank changes.",
        synthesis_rationale="This combines the observed conditional relation with a local-state mechanism.",
        premise_statement_ids=["p1", "p2", "p3"],
        gap_statement_ids=["g1"],
        assumptions=["interfacial state is independently measurable"],
        predictions=[
            SynthesizedPrediction(
                observable="rank order and local accumulation",
                expected_result="rank changes covary with local accumulation",
                rationale="mediation predicts coupled changes",
                source_candidate_ids=["pc:rel", "pc:ref"],
            )
        ],
        falsifiers=[
            SynthesizedFalsifier(
                observable="rank order",
                falsifying_outcome="rank is stable and unrelated to local accumulation",
                source_candidate_ids=["pc:rel", "pc:ref"],
            )
        ],
        discriminating_test=SynthesizedDiscriminatingTest(
            test_design="Compare matched matrices and independently measure local accumulation.",
            primary_observables=["rank order", "local accumulation"],
            source_candidate_ids=["pc:rel", "pc:ref"],
            favoring_outcomes=["covariation favors mediation", "no covariation favors parallel matrix effect"],
        ),
        unresolved_questions=["which matrix constituent dominates"],
    )


def _report(portfolio: ProductionFacingScientificCandidatePortfolio, *, hypotheses=None):
    from pipeline_core.discovery.reframing.integrated_hypothesis_shadow import _sha256_json

    rows = [_synthesized()] if hypotheses is None else hypotheses
    return CrossLaneSynthesisShadowReport(
        report_id="synthesis-report:1",
        source_candidate_portfolio_id=portfolio.portfolio_id,
        source_candidate_portfolio_sha256=_sha256_json(portfolio),
        source_task_id=portfolio.source_task_id,
        source_context_id=portfolio.source_context_id,
        question=portfolio.question,
        backend_name="fake",
        model_name="fake",
        source_candidate_ids=[row.candidate_id for row in portfolio.candidates],
        hypotheses=rows,
        synthesized_hypothesis_ids=[row.hypothesis_id for row in rows],
        abstention_reason=None if rows else "no justified synthesis",
        source_candidate_count=portfolio.candidate_count,
        synthesized_hypothesis_count=len(rows),
        llm_calls_performed=1 if rows else 0,
        cross_lane_synthesis_performed=bool(rows),
    )


def test_integrated_view_is_legacy_plus_synthesis():
    portfolio = _portfolio()
    output = build_integrated_scientific_hypothesis_shadow(
        candidate_portfolio=portfolio,
        synthesis_report=_report(portfolio),
    )
    assert output.legacy_hypothesis_count == 1
    assert output.synthesized_addition_count == 1
    assert output.integrated_hypothesis_count == 2
    assert output.integrated_hypotheses == output.legacy_hypotheses + output.synthesized_additions
    assert output.shadow_output_differs_from_legacy is True


def test_raw_reframing_candidate_is_not_presented_but_is_not_discarded():
    portfolio = _portfolio()
    output = build_integrated_scientific_hypothesis_shadow(
        candidate_portfolio=portfolio,
        synthesis_report=_report(portfolio),
    )
    assert output.raw_reframing_source_candidate_count == 1
    assert output.raw_reframing_candidates_presented_as_final_hypotheses is False
    assert output.raw_reframing_candidates_discarded is False
    assert all(row.origin != "SCIENTIFIC_REFRAMING" for row in output.integrated_hypotheses)


def test_legacy_relational_content_is_preserved():
    rel = _rel()
    portfolio = _portfolio(rel, _ref())
    output = build_integrated_scientific_hypothesis_shadow(
        candidate_portfolio=portfolio,
        synthesis_report=_report(portfolio),
    )
    row = output.legacy_hypotheses[0]
    assert row.title == rel.title
    assert row.hypothesis_statement == rel.scientific_proposal
    assert row.source_candidate_ids == [rel.candidate_id]
    assert row.source_object_ids == [rel.source_object_id]


def test_synthesis_content_is_preserved():
    portfolio = _portfolio()
    synthesis = _synthesized()
    output = build_integrated_scientific_hypothesis_shadow(
        candidate_portfolio=portfolio,
        synthesis_report=_report(portfolio, hypotheses=[synthesis]),
    )
    row = output.synthesized_additions[0]
    assert row.title == synthesis.title
    assert row.hypothesis_statement == synthesis.hypothesis_statement
    assert row.synthesis_kind == synthesis.synthesis_kind
    assert row.discriminating_test is not None
    assert row.discriminating_test.test_design == synthesis.discriminating_test.test_design


def test_coverage_tracks_both_lanes():
    portfolio = _portfolio(_rel("pc:rel"), _rel("pc:rel2"), _ref("pc:ref"), _ref("pc:ref2"))
    synthesis = _synthesized()
    output = build_integrated_scientific_hypothesis_shadow(
        candidate_portfolio=portfolio,
        synthesis_report=_report(portfolio, hypotheses=[synthesis]),
    )
    assert output.synthesis_covered_relational_candidate_ids == ["pc:rel"]
    assert output.synthesis_covered_reframing_candidate_ids == ["pc:ref"]
    assert output.synthesis_uncovered_relational_candidate_ids == ["pc:rel2"]
    assert output.synthesis_uncovered_reframing_candidate_ids == ["pc:ref2"]


def test_abstention_keeps_integrated_view_identical_to_legacy():
    portfolio = _portfolio()
    output = build_integrated_scientific_hypothesis_shadow(
        candidate_portfolio=portfolio,
        synthesis_report=_report(portfolio, hypotheses=[]),
    )
    assert output.synthesized_addition_count == 0
    assert output.integrated_hypotheses == output.legacy_hypotheses
    assert output.shadow_output_differs_from_legacy is False


def test_rejects_candidate_portfolio_id_mismatch():
    portfolio = _portfolio()
    report = _report(portfolio).model_copy(update={"source_candidate_portfolio_id": "wrong"})
    with pytest.raises(ValueError, match="portfolio ID mismatch"):
        build_integrated_scientific_hypothesis_shadow(
            candidate_portfolio=portfolio,
            synthesis_report=report,
        )


def test_rejects_candidate_portfolio_sha_mismatch():
    portfolio = _portfolio()
    report = _report(portfolio).model_copy(update={"source_candidate_portfolio_sha256": "0" * 64})
    with pytest.raises(ValueError, match="SHA mismatch"):
        build_integrated_scientific_hypothesis_shadow(
            candidate_portfolio=portfolio,
            synthesis_report=report,
        )


def test_rejects_candidate_inventory_mismatch():
    portfolio = _portfolio()
    report = _report(portfolio).model_copy(update={"source_candidate_ids": ["pc:rel"]})
    with pytest.raises(ValueError, match="inventory mismatch"):
        build_integrated_scientific_hypothesis_shadow(
            candidate_portfolio=portfolio,
            synthesis_report=report,
        )


def test_no_ranking_or_production_selection_authority_created():
    portfolio = _portfolio()
    output = build_integrated_scientific_hypothesis_shadow(
        candidate_portfolio=portfolio,
        synthesis_report=_report(portfolio),
    )
    assert output.synthesized_hypotheses_selected_or_ranked is False
    assert output.source_candidates_ranked is False
    assert output.source_candidates_pruned is False
    assert output.final_production_selection_performed is False
    assert output.production_selection_changed is False
    assert output.canonical_graph_mutated is False


def test_builder_is_deterministic():
    portfolio = _portfolio()
    report = _report(portfolio)
    first = build_integrated_scientific_hypothesis_shadow(
        candidate_portfolio=portfolio,
        synthesis_report=report,
    )
    second = build_integrated_scientific_hypothesis_shadow(
        candidate_portfolio=portfolio,
        synthesis_report=report,
    )
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
