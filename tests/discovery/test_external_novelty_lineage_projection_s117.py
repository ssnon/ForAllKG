from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReview,
    ClaimSearchCoverage,
    ExternalNoveltyCard,
    ExternalNoveltyPolicy,
    ExternalNoveltyReport,
    HypothesisNoveltyDepthProfile,
    HypothesisSearchCoverage,
)
from pipeline_core.discovery.external_novelty_lineage_projection import (
    project_external_novelty_report_to_final_hypothesis,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingHypothesisPlan,
    RelationalAtomicBindingPlan,
)


CANDIDATE = "hypothesis:candidate"
FINAL = "hypothesis:final"


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _plan() -> RelationalAtomicBindingPlan:
    row = RelationalAtomicBindingHypothesisPlan(
        original_hypothesis_id="hypothesis:original",
        candidate_hypothesis_id=CANDIDATE,
        final_hypothesis_id=FINAL,
        alpha6_decision="RETAIN",
        certification_status="UNRESOLVED",
        n10_selection_class="RETAINED",
        source_candidate_portfolio="/tmp/candidate.json",
        source_candidate_portfolio_sha256="a" * 64,
        source_query_plan="/tmp/query.json",
        source_query_plan_sha256="b" * 64,
        claim_count=0,
        binding_ready_claim_count=0,
        novelty_bearing_binding_ready_claim_count=0,
        binding_status="NO_BINDABLE_CLAIMS",
        claims=[],
    )

    body = {
        "schema_version": "relational-atomic-binding-plan-v1",
        "run_dir": "/tmp/run",
        "source_alpha6_candidate_portfolio": "/tmp/final.json",
        "source_alpha6_candidate_portfolio_sha256": "c" * 64,
        "source_certification_report": "/tmp/cert.json",
        "source_certification_report_sha256": "d" * 64,
        "hypotheses": [row.model_dump(mode="json")],
        "hypothesis_count": 1,
        "ready_hypothesis_count": 0,
        "not_ready_hypothesis_count": 1,
        "claim_count": 0,
        "binding_ready_claim_count": 0,
        "novelty_bearing_binding_ready_claim_count": 0,
        "hypothesis_status_counts": {"NO_BINDABLE_CLAIMS": 1},
        "claim_status_counts": {},
        "certification_only_source_required": True,
        "candidate_final_authority_equivalence_required": True,
        "endpoint_binding_performed": False,
        "verifier_result_observed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return RelationalAtomicBindingPlan(
        **body,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )


def _source_report() -> ExternalNoveltyReport:
    claim_review = ClaimPriorArtReview(
        hypothesis_id=CANDIDATE,
        claim_id="claim:1",
        claim_text="Candidate claim.",
        importance="core",
        status="COMPONENTS_ONLY",
        matches=[],
        coverage=ClaimSearchCoverage(
            claim_id="claim:1",
            query_count=1,
            successful_query_count=1,
            unique_work_count=4,
            abstract_work_count=2,
            reviewed_work_count=2,
        ),
        reason_codes=["bounded_example"],
        reviewer_unknown_work_ids=[],
        interpretation="No direct relation established.",
    )
    coverage = HypothesisSearchCoverage(
        hypothesis_id=CANDIDATE,
        query_count=1,
        successful_query_count=1,
        provider_success_count=2,
        unique_work_count=4,
        abstract_work_count=2,
        core_claim_count=1,
        core_claims_with_minimum_abstract_coverage=0,
        sufficient_for_absence_based_novelty=False,
    )
    depth = HypothesisNoveltyDepthProfile(
        hypothesis_id=CANDIDATE,
        role_binding_complete=True,
        core_claim_count=1,
        relation_backed_core_claim_count=0,
        known_core_relation_fraction=0.0,
        novelty_bearing_claim_ids=["claim:1"],
        novelty_bearing_claim_count=1,
        novelty_bearing_relation_backed_claim_ids=[],
        novelty_bearing_gap_like_claim_ids=[],
        novelty_bearing_conflicting_claim_ids=[],
        novelty_bearing_unresolved_claim_ids=["claim:1"],
        novelty_bearing_relation_backed_fraction=0.0,
        gap_centrality="UNRESOLVED",
        novelty_bearing_prior_art_state="UNRESOLVED",
        reason_codes=["bounded_example"],
    )
    card = ExternalNoveltyCard(
        hypothesis_id=CANDIDATE,
        title="Candidate title",
        status="INSUFFICIENT_SEARCH_EVIDENCE",
        claim_reviews=[claim_review],
        coverage=coverage,
        novelty_depth_profile=depth,
        reason_codes=["insufficient_search_evidence"],
        interpretation="Search coverage is insufficient.",
        search_limitations=["bounded test fixture"],
    )
    return ExternalNoveltyReport(
        report_id="external_novelty_report:source",
        report_sha256="e" * 64,
        source_portfolio_id="portfolio:candidate",
        source_prior_art_packet_id="packet:1",
        searched_at_utc="2026-09-23T00:00:00Z",
        cards=[card],
        status_counts={"INSUFFICIENT_SEARCH_EVIDENCE": 1},
        policy=ExternalNoveltyPolicy(),
    )


def test_lineage_projection_changes_only_hypothesis_namespace() -> None:
    source = _source_report()
    projected, audit = project_external_novelty_report_to_final_hypothesis(
        plan=_plan(),
        source_report=source,
        final_hypothesis_id=FINAL,
    )

    assert len(projected.cards) == 1
    card = projected.cards[0]
    assert card.hypothesis_id == FINAL
    assert card.coverage.hypothesis_id == FINAL
    assert card.claim_reviews[0].hypothesis_id == FINAL
    assert card.novelty_depth_profile is not None
    assert card.novelty_depth_profile.hypothesis_id == FINAL

    assert card.status == "INSUFFICIENT_SEARCH_EVIDENCE"
    assert card.coverage.sufficient_for_absence_based_novelty is False
    assert card.claim_reviews[0].status == "COMPONENTS_ONLY"
    assert card.interpretation == "Search coverage is insufficient."
    assert projected.source_portfolio_id == "portfolio:candidate"
    assert projected.source_prior_art_packet_id == "packet:1"
    assert projected.searched_at_utc == "2026-09-23T00:00:00Z"

    assert audit.candidate_hypothesis_id == CANDIDATE
    assert audit.final_hypothesis_id == FINAL
    assert audit.external_novelty_reassessed is False
    assert audit.literature_search_rerun is False
    assert audit.scientific_content_added is False
    assert audit.coverage_counts_changed is False


def test_lineage_projection_does_not_mutate_source_report() -> None:
    source = _source_report()
    before = source.model_dump(mode="json")

    project_external_novelty_report_to_final_hypothesis(
        plan=_plan(),
        source_report=source,
        final_hypothesis_id=FINAL,
    )

    assert source.model_dump(mode="json") == before


def test_lineage_projection_fails_closed_for_wrong_source_namespace() -> None:
    source = _source_report()
    source.cards[0].coverage.hypothesis_id = "hypothesis:wrong"

    try:
        project_external_novelty_report_to_final_hypothesis(
            plan=_plan(),
            source_report=source,
            final_hypothesis_id=FINAL,
        )
    except ValueError as exc:
        assert "coverage hypothesis namespace" in str(exc)
    else:
        raise AssertionError("expected fail-closed namespace rejection")
