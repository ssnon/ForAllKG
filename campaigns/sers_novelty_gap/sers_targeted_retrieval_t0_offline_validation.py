from __future__ import annotations

import hashlib
import json
from typing import Any

from dac_her.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
    PriorArtWork,
)
from dac_her.literature_retrieval import canonicalize_prior_art_works
from dac_her.targeted_novelty_retrieval import merge_prior_art_packets


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


def _work(
    work_id: str,
    *,
    title: str,
    doi: str | None,
    provider: str,
    query_id: str,
    claim_id: str,
    citation_count: int = 0,
    year: int = 2025,
    abstract: str | None = None,
) -> PriorArtWork:
    return PriorArtWork(
        work_id=work_id,
        title=title,
        year=year,
        publication_date=f"{year}-01-01",
        doi=doi,
        url=f"https://example.invalid/{work_id}",
        abstract=abstract,
        authors=[f"Author {provider}"],
        venue="Synthetic T0 Venue",
        citation_count=citation_count,
        providers=[provider],
        provider_ids={provider: work_id},
        retrieval_query_ids=[query_id],
        retrieval_claim_ids=[claim_id],
    )


def _packet(
    packet_id: str,
    *,
    plan_id: str,
    works: list[PriorArtWork],
    raw_work_count: int | None = None,
    supplementary_records_collapsed: int = 0,
    provider: str = "synthetic",
) -> PriorArtPacket:
    raw_count = len(works) if raw_work_count is None else raw_work_count
    body = {
        "schema_version": "prior-art-packet-v1",
        "packet_id": packet_id,
        "source_portfolio_id": "portfolio:t0",
        "source_query_plan_id": plan_id,
        "searched_at_utc": "2026-08-17T00:00:00+00:00",
        "providers_requested": [provider],
        "works": [row.model_dump(mode="json") for row in works],
        "executions": [],
        "raw_work_count": raw_count,
        "canonical_work_count": len(works),
        "deduplicated_work_count": max(0, raw_count - len(works)),
        "supplementary_records_collapsed": supplementary_records_collapsed,
        "epistemic_usage": "prior_art_only_not_positive_premise",
    }
    return PriorArtPacket(
        **body,
        packet_sha256=_sha256_json(body),
    )


def _plan() -> LiteratureQueryPlan:
    body = {
        "schema_version": "literature-query-plan-v1",
        "plan_id": "plan:t0-augmented",
        "source_portfolio_id": "portfolio:t0",
        "queries": [],
        "claims": [],
        "policy_version": "external-novelty-query-policy-v1",
    }
    return LiteratureQueryPlan(
        **body,
        plan_sha256=_sha256_json(body),
    )


def _merge(
    base_works: list[PriorArtWork],
    delta_works: list[PriorArtWork],
    *,
    base_raw: int | None = None,
    delta_raw: int | None = None,
    base_supp: int = 0,
    delta_supp: int = 0,
) -> PriorArtPacket:
    return merge_prior_art_packets(
        _packet(
            "packet:base",
            plan_id="plan:base",
            works=base_works,
            raw_work_count=base_raw,
            supplementary_records_collapsed=base_supp,
            provider="openalex",
        ),
        _packet(
            "packet:delta",
            plan_id="plan:delta",
            works=delta_works,
            raw_work_count=delta_raw,
            supplementary_records_collapsed=delta_supp,
            provider="crossref",
        ),
        _plan(),
    )


def build_t0_offline_report() -> dict[str, Any]:
    shared_title = "Shared DOI identity across base and targeted retrieval records"
    same_doi = _merge(
        [_work(
            "work:base-shared",
            title=shared_title,
            doi="10.1000/shared",
            provider="openalex",
            query_id="q:base",
            claim_id="c:base",
            citation_count=5,
            abstract="short",
        )],
        [_work(
            "work:delta-shared",
            title=shared_title + " extended",
            doi="https://doi.org/10.1000/shared",
            provider="crossref",
            query_id="q:delta",
            claim_id="c:delta",
            citation_count=7,
            abstract="a substantially longer abstract for deterministic merge",
        )],
        base_raw=3,
        delta_raw=2,
    )
    same = same_doi.works[0]

    collision_title = (
        "Exact normalized title with intentionally conflicting DOI families"
    )
    distinct_doi = _merge(
        [_work(
            "work:doi-a",
            title=collision_title,
            doi="10.1000/family-a",
            provider="openalex",
            query_id="q:a",
            claim_id="c:a",
        )],
        [_work(
            "work:doi-b",
            title=collision_title,
            doi="10.1000/family-b",
            provider="crossref",
            query_id="q:b",
            claim_id="c:b",
        )],
    )

    unambiguous_title = (
        "DOI less record may join one unambiguous DOI family by exact title"
    )
    one_family = _merge(
        [_work(
            "work:doi-less",
            title=unambiguous_title,
            doi=None,
            provider="openalex",
            query_id="q:less",
            claim_id="c:less",
        )],
        [_work(
            "work:doi-one",
            title=unambiguous_title,
            doi="10.1000/one-family",
            provider="crossref",
            query_id="q:one",
            claim_id="c:one",
        )],
    )

    ambiguous_title = (
        "DOI less record must remain unresolved across conflicting DOI families"
    )
    ambiguous = _merge(
        [
            _work(
                "work:amb-less",
                title=ambiguous_title,
                doi=None,
                provider="openalex",
                query_id="q:amb-less",
                claim_id="c:amb-less",
            ),
            _work(
                "work:amb-a",
                title=ambiguous_title,
                doi="10.1000/amb-a",
                provider="openalex",
                query_id="q:amb-a",
                claim_id="c:amb-a",
            ),
        ],
        [_work(
            "work:amb-b",
            title=ambiguous_title,
            doi="10.1000/amb-b",
            provider="crossref",
            query_id="q:amb-b",
            claim_id="c:amb-b",
        )],
    )

    supplementary_title = (
        "Main article and supplementary DOI family canonicalization case"
    )
    supplementary = _merge(
        [_work(
            "work:main",
            title=supplementary_title,
            doi="10.1000/supp-case",
            provider="openalex",
            query_id="q:main",
            claim_id="c:main",
        )],
        [_work(
            "work:supp",
            title=supplementary_title,
            doi="10.1000/supp-case.s1",
            provider="crossref",
            query_id="q:supp",
            claim_id="c:supp",
        )],
        base_supp=2,
        delta_supp=1,
    )

    deterministic_1 = _merge(
        list(distinct_doi.works),
        [_work(
            "work:high-cite",
            title="Independent high citation targeted retrieval record",
            doi="10.1000/high-cite",
            provider="crossref",
            query_id="q:high",
            claim_id="c:high",
            citation_count=100,
            year=2024,
        )],
    )
    deterministic_2 = _merge(
        list(distinct_doi.works),
        [_work(
            "work:high-cite",
            title="Independent high citation targeted retrieval record",
            doi="10.1000/high-cite",
            provider="crossref",
            query_id="q:high",
            claim_id="c:high",
            citation_count=100,
            year=2024,
        )],
    )

    direct_canonical, direct_supp = canonicalize_prior_art_works(
        [
            _work(
                "work:direct-a",
                title=collision_title,
                doi="10.1000/direct-a",
                provider="openalex",
                query_id="q:direct-a",
                claim_id="c:direct-a",
            ),
            _work(
                "work:direct-b",
                title=collision_title,
                doi="10.1000/direct-b",
                provider="crossref",
                query_id="q:direct-b",
                claim_id="c:direct-b",
            ),
        ]
    )

    checks = {
        "same_doi_cross_packet_merges": len(same_doi.works) == 1,
        "same_doi_provider_provenance_unioned":
            same.providers == ["crossref", "openalex"],
        "same_doi_query_provenance_unioned":
            same.retrieval_query_ids == ["q:base", "q:delta"],
        "same_doi_claim_provenance_unioned":
            same.retrieval_claim_ids == ["c:base", "c:delta"],
        "same_doi_longer_abstract_preserved":
            same.abstract
            == "a substantially longer abstract for deterministic merge",
        "same_doi_packet_raw_accounting": same_doi.raw_work_count == 5,
        "same_doi_packet_canonical_accounting":
            same_doi.canonical_work_count == 1,
        "same_doi_packet_dedup_accounting":
            same_doi.deduplicated_work_count == 4,
        "distinct_doi_same_title_preserved": len(distinct_doi.works) == 2,
        "distinct_doi_families_exact":
            sorted(row.doi for row in distinct_doi.works)
            == ["10.1000/family-a", "10.1000/family-b"],
        "doi_less_one_family_merges": len(one_family.works) == 1,
        "doi_less_one_family_retains_doi":
            one_family.works[0].doi == "10.1000/one-family",
        "doi_less_two_families_remains_unresolved":
            len(ambiguous.works) == 3,
        "supplementary_family_merges": len(supplementary.works) == 1,
        "supplementary_cross_packet_accounting":
            supplementary.supplementary_records_collapsed == 4,
        "shared_seam_preserves_distinct_doi_families":
            len(direct_canonical) == 2,
        "shared_seam_direct_supplementary_count_zero": direct_supp == 0,
        "targeted_sort_policy_retained":
            deterministic_1.works[0].work_id == "work:high-cite",
        "merge_repeat_deterministic":
            _canonical_json(deterministic_1)
            == _canonical_json(deterministic_2),
        "epistemic_usage_unchanged":
            deterministic_1.epistemic_usage
            == "prior_art_only_not_positive_premise",
    }

    body = {
        "schema_version":
            "sers-targeted-retrieval-t0-canonicalization-offline-v1",
        "scope": "T0_SHARED_CANONICALIZATION_SEAM",
        "checks": checks,
        "scenario_counts": {
            "same_doi": len(same_doi.works),
            "distinct_doi_same_title": len(distinct_doi.works),
            "doi_less_one_family": len(one_family.works),
            "doi_less_two_families": len(ambiguous.works),
            "supplementary_family": len(supplementary.works),
        },
        "targeted_retrieval_called": False,
        "provider_calls": 0,
        "network_calls": 0,
        "llm_calls": 0,
        "ranker_recomputed": False,
        "claim_reviewer_recomputed": False,
        "hypothesis_rewrite_called": False,
        "fresh_reserve_c_consumed": False,
        "automatic_next_stage_authorized": False,
    }
    structural_pass = all(checks.values())
    body["structural_outcome"] = (
        "SERS_TARGETED_RETRIEVAL_T0_CANONICALIZATION_OFFLINE_PASS"
        if structural_pass
        else
        "SERS_TARGETED_RETRIEVAL_T0_CANONICALIZATION_OFFLINE_FAIL"
    )
    body["run_sha256"] = _sha256_json(body)
    body["run_id"] = (
        "sers_targeted_retrieval_t0_canonicalization_offline:"
        + body["run_sha256"][:20]
    )
    return body
