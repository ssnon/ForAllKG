from __future__ import annotations

from types import SimpleNamespace

from domains.registry import get_domain_profile
from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReview,
    ClaimSearchCoverage,
    ExternalNoveltyCard,
    ExternalNoveltyPolicy,
    ExternalNoveltyReport,
    HypothesisSearchCoverage,
    PriorArtMatch,
    PriorArtPacket,
    PriorArtWork,
)
from pipeline_core.discovery.prior_art_domain_authority_audit import (
    build_prior_art_domain_authority_audit,
)
from pipeline_core.discovery.prior_art_metadata_resolution_s227 import (
    PriorArtMetadataResolver,
)


class FakeParentProvider:
    provider_name = "fixture"

    def search(self, query, *, limit):
        if query.query_text == "10.1021/acs.nanolett.6c02505":
            return [
                PriorArtWork(
                    work_id="parent",
                    title="Beyond SERS Hotspot Localization",
                    doi="10.1021/acs.nanolett.6c02505",
                    abstract="Parent article abstract with SERS reproducibility evidence.",
                    providers=["fixture"],
                    provider_ids={"fixture": "parent"},
                    retrieval_query_ids=[query.query_id],
                )
            ]
        return []


def _packet(work):
    return PriorArtPacket(
        packet_id="packet",
        packet_sha256="a" * 64,
        source_portfolio_id="portfolio",
        source_query_plan_id="plan",
        searched_at_utc="2026-10-02T00:00:00+00:00",
        providers_requested=["fixture"],
        works=[work],
        raw_work_count=1,
        canonical_work_count=1,
    )


def test_m11_parent_metadata_hydration_recovers_abstract_and_parent_doi():
    supplementary = PriorArtWork(
        work_id="supp",
        title="Beyond SERS Hotspot Localization",
        doi="10.1021/acs.nanolett.6c02505.s001",
        abstract=None,
        providers=["crossref"],
        provider_ids={"crossref": "supp"},
        retrieval_query_ids=["q-original"],
        retrieval_claim_ids=["c-original"],
    )
    resolved, audit = PriorArtMetadataResolver(
        [FakeParentProvider()],
        lookup_limit=5,
    ).resolve(
        _packet(supplementary),
        target_work_ids={"supp"},
    )
    assert len(resolved.works) == 1
    work = resolved.works[0]
    assert work.doi == "10.1021/acs.nanolett.6c02505"
    assert work.abstract
    assert work.retrieval_query_ids == ["q-original"]
    assert audit["target_abstract_recovered_count"] == 1
    assert audit["target_supplementary_doi_count"] == 1
    assert audit["works"][0]["resolved_doi"] == "10.1021/acs.nanolett.6c02505"


def _report_for(match):
    coverage = ClaimSearchCoverage(
        claim_id="c1",
        query_count=3,
        successful_query_count=3,
        unique_work_count=8,
        abstract_work_count=5,
        reviewed_work_count=5,
    )
    review = ClaimPriorArtReview(
        hypothesis_id="h1",
        claim_id="c1",
        claim_text="SERS architecture relation",
        importance="core",
        status=(
            "PARTIAL_PRIOR_ART"
            if match.relationship == "PARTIAL_PRIOR_ART"
            else "COMPONENTS_ONLY"
        ),
        matches=[match],
        coverage=coverage,
        interpretation="fixture",
    )
    hcov = HypothesisSearchCoverage(
        hypothesis_id="h1",
        query_count=3,
        successful_query_count=3,
        provider_success_count=2,
        unique_work_count=8,
        abstract_work_count=5,
        core_claim_count=1,
        core_claims_with_minimum_abstract_coverage=1,
        sufficient_for_absence_based_novelty=True,
    )
    card = ExternalNoveltyCard(
        hypothesis_id="h1",
        title="fixture",
        status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        claim_reviews=[review],
        coverage=hcov,
        interpretation="fixture",
    )
    return ExternalNoveltyReport(
        report_id="report",
        report_sha256="b" * 64,
        source_portfolio_id="portfolio",
        source_prior_art_packet_id="packet",
        searched_at_utc="2026-10-02T00:00:00+00:00",
        cards=[card],
        status_counts={"KNOWN_COMPONENTS_WITH_RELATIONAL_GAP": 1},
        policy=ExternalNoveltyPolicy(),
    )


def _match(relationship):
    return PriorArtMatch(
        work_id="x",
        relationship=relationship,
        confidence=0.9,
        rationale="fixture",
        evidence_spans=["fixture"],
        relevance_score=0.9,
        semantic_similarity=0.9,
        lexical_coverage=0.5,
        reaction_domain_relevance=0.0,
        catalyst_scope_relevance=0.0,
        scope_compatible_for_conflict=False,
        scope_reason_codes=["low_sers_system_scope_overlap"],
        title="Periodic modulation of tokamak plasma confinement",
        abstract_available=True,
    )


def test_m11_domain_audit_allows_raw_cross_domain_neighbor_without_authority():
    work = PriorArtWork(
        work_id="x",
        title="Periodic modulation of tokamak plasma confinement",
        abstract="Tokamak plasma confinement and vibration behavior.",
    )
    packet = _packet(work)
    report = _report_for(_match("COMPONENT_ONLY"))
    audit = build_prior_art_domain_authority_audit(
        domain_profile=get_domain_profile("sers_au_ag"),
        packet=packet,
        report=report,
    )
    assert audit["domain_incompatible_retrieved_work_count"] == 1
    assert audit["domain_incompatible_compiled_match_work_count"] == 1
    assert audit["domain_authority_violation_count"] == 0
    assert audit["pass"] is True


def test_m11_domain_audit_flags_incompatible_partial_authority():
    work = PriorArtWork(
        work_id="x",
        title="Periodic modulation of tokamak plasma confinement",
        abstract="Tokamak plasma confinement and vibration behavior.",
    )
    packet = _packet(work)
    report = _report_for(_match("PARTIAL_PRIOR_ART"))
    audit = build_prior_art_domain_authority_audit(
        domain_profile=get_domain_profile("sers_au_ag"),
        packet=packet,
        report=report,
    )
    assert audit["domain_authority_violation_count"] == 1
    assert audit["pass"] is False
