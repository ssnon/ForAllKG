from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisContext,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.reframing.scientific_synthesis_n10_authority import (
    SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_CERTIFICATION_ONLY,
    SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_HARD_FILTER,
    SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODES,
    StrictLegacyN10Resolution,
    merge_legacy_strict_with_scientific_certification,
    split_scientific_synthesis_portfolio_by_n10_certification,
)


def _card(hypothesis_id: str) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hypothesis_id,
        domain_profile_id="sers_au_ag",
        source_context_id="ctx",
        source_context_sha256="ctx-sha",
        source_report_id="report",
        source_report_sha256="report-sha",
        title=hypothesis_id,
        hypothesis_statement="A testable scientific relation.",
        hypothesis_type="cross_evidence_synthesis",
        premise_statement_ids=["p1"],
        inferential_bridge="Premise p1 supports the testable relation.",
        predicted_observations=[
            PredictedObservation(
                observation_id=hypothesis_id + ":pred",
                observable="signal",
                expected_direction="unspecified",
                rationale="A measurable outcome.",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id=hypothesis_id + ":false",
                observable="signal",
                falsifying_outcome="The predicted relation is absent.",
            )
        ],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )


def _portfolio(*ids: str) -> HypothesisPortfolio:
    return HypothesisPortfolio(
        portfolio_id="portfolio:" + "-".join(ids),
        domain_profile_id="sers_au_ag",
        source_context_id="ctx",
        source_context_sha256="ctx-sha",
        source_report_id="report",
        source_report_sha256="report-sha",
        hypotheses=[_card(value) for value in ids],
    )


def _gate() -> dict:
    return {
        "schema_version": "scientific-novelty-fallback-gate-v2",
        "production_authority": True,
        "authority_scope": "scientific_cross_lane_synthesis_candidate",
        "authority_source": "n10_role_aware_nonobviousness_v2",
        "conditional_is_positive": False,
        "absence_is_novelty": False,
        "gates": [
            {
                "hypothesis_id": "h1",
                "selection_class": "ELIGIBLE",
                "positive_nonobviousness_authority": True,
                "fallback_allowed": True,
                "action": "KEEP",
                "reason_codes": ["role_aware_v2_eligible_and_positive"],
            },
            {
                "hypothesis_id": "h2",
                "selection_class": "CONDITIONAL",
                "positive_nonobviousness_authority": False,
                "fallback_allowed": False,
                "action": "RESOLVE_NOVELTY_BEARING_EVIDENCE",
                "reason_codes": ["role_aware_v2_conditional_fail_closed"],
            },
            {
                "hypothesis_id": "h3",
                "selection_class": "INELIGIBLE",
                "positive_nonobviousness_authority": False,
                "fallback_allowed": False,
                "action": "REJECT",
                "reason_codes": ["role_aware_v2_ineligible"],
            },
        ],
    }


def test_authority_modes_default_contract_contains_certification_only():
    assert SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODES == (
        SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_HARD_FILTER,
        SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_CERTIFICATION_ONLY,
    )


def test_certification_only_preserves_candidates_and_certifies_only_positive():
    source = _portfolio("h1", "h2", "h3")
    candidate, certified, report = (
        split_scientific_synthesis_portfolio_by_n10_certification(
            portfolio=source,
            production_gate=_gate(),
        )
    )

    assert candidate.portfolio_id == source.portfolio_id
    assert [x.hypothesis_id for x in candidate.hypotheses] == [
        "h1",
        "h2",
        "h3",
    ]
    assert [x.hypothesis_id for x in certified.hypotheses] == ["h1"]
    assert report.candidate_count == 3
    assert report.certified_count == 1
    assert report.unresolved_count == 1
    assert report.rejected_count == 1

    by_id = {
        row.hypothesis_id: row
        for row in report.decisions
    }
    assert by_id["h1"].certification_status == "NOVELTY_CERTIFIED"
    assert by_id["h2"].certification_status == "NOVELTY_UNRESOLVED"
    assert by_id["h3"].certification_status == "NOVELTY_REJECTED"
    assert all(row.candidate_retained for row in report.decisions)


def test_certification_only_conditional_is_not_positive_authority():
    source = _portfolio("h1", "h2", "h3")
    _, _, report = split_scientific_synthesis_portfolio_by_n10_certification(
        portfolio=source,
        production_gate=_gate(),
    )
    unresolved = next(
        row for row in report.decisions
        if row.hypothesis_id == "h2"
    )
    assert unresolved.novelty_certified is False
    assert unresolved.positive_nonobviousness_authority is False
    assert unresolved.unresolved_dimensions == ["EVIDENCE_CLOSURE"]


def test_merge_builds_distinct_candidate_and_certified_views(tmp_path: Path):
    legacy = _portfolio("legacy")
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        legacy.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    legacy_resolution = StrictLegacyN10Resolution(
        run_dir=str(tmp_path),
        portfolio_path=str(legacy_path),
        authority_kind="hard_filter_n10",
        hypothesis_count=1,
    )

    source = _portfolio("h1", "h2", "h3")
    candidates, certified, certification = (
        split_scientific_synthesis_portfolio_by_n10_certification(
            portfolio=source,
            production_gate=_gate(),
        )
    )
    context = HypothesisContext.model_construct(
        context_id="ctx",
        context_sha256="ctx-sha",
        domain_profile_id="sers_au_ag",
        source_report_id="report",
        source_report_sha256="report-sha",
    )

    merged_candidates, merged_certified, report = (
        merge_legacy_strict_with_scientific_certification(
            context=context,
            legacy_resolution=legacy_resolution,
            scientific_candidates=candidates,
            scientific_certified=certified,
            certification_report=certification,
        )
    )

    assert [x.hypothesis_id for x in merged_candidates.hypotheses] == [
        "legacy",
        "h1",
        "h2",
        "h3",
    ]
    assert [x.hypothesis_id for x in merged_certified.hypotheses] == [
        "legacy",
        "h1",
    ]
    assert report.merged_candidate_count == 4
    assert report.merged_certified_count == 2
    assert report.scientific_unresolved_count == 1
    assert report.scientific_rejected_count == 1
