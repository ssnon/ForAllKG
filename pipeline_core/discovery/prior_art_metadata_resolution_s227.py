from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQuery,
    PriorArtPacket,
    PriorArtWork,
)
from pipeline_core.discovery.prior_art_retrieval import (
    LiteratureSearchProvider,
    canonicalize_prior_art_works,
)


_SUPPLEMENTARY_DOI_RE = re.compile(r"\.s\d+$", re.I)


def canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def norm_doi(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith("https://doi.org/"):
        text = text[len("https://doi.org/") :]
    if text.startswith("doi:"):
        text = text[4:]
    return text or None


def doi_family(value: object) -> str | None:
    doi = norm_doi(value)
    if not doi:
        return None
    return _SUPPLEMENTARY_DOI_RE.sub("", doi)


def is_supplementary_doi(value: object) -> bool:
    doi = norm_doi(value) or ""
    return bool(_SUPPLEMENTARY_DOI_RE.search(doi))


def norm_title(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9α-ω가-힣]+", " ", text)
    return " ".join(text.split())


def work_identity_key(work: PriorArtWork) -> str:
    family = doi_family(work.doi)
    if family:
        return "doi_family:" + family
    title = norm_title(work.title)
    if title:
        return "title:" + hashlib.sha256(
            title.encode("utf-8")
        ).hexdigest()[:24]
    return "work_id:" + work.work_id


def inherited_candidate(
    candidate: PriorArtWork,
    *,
    source: PriorArtWork,
) -> PriorArtWork:
    """
    Metadata-resolution lookups must not create scientific query coverage.

    Provider/document provenance is retained from the candidate, while
    retrieval query/claim bindings are inherited from the original work.
    """
    return candidate.model_copy(
        update={
            "retrieval_query_ids":
                list(source.retrieval_query_ids),
            "retrieval_claim_ids":
                list(source.retrieval_claim_ids),
        }
    )


def candidate_matches_source(
    source: PriorArtWork,
    candidate: PriorArtWork,
) -> tuple[bool, str]:
    source_family = doi_family(source.doi)
    candidate_family = doi_family(candidate.doi)

    if source_family:
        if candidate_family == source_family:
            return True, "DOI_FAMILY_EXACT"
        return False, "DOI_FAMILY_MISMATCH"

    source_title = norm_title(source.title)
    candidate_title = norm_title(candidate.title)

    if (
        len(source_title) >= 20
        and candidate_title == source_title
    ):
        return True, "EXACT_NORMALIZED_TITLE"

    return False, "TITLE_MISMATCH"


@dataclass(frozen=True)
class ResolutionAttempt:
    work_id: str
    provider: str
    query_mode: str
    query_text: str
    success: bool
    candidate_count: int
    accepted_count: int
    elapsed_seconds: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "provider": self.provider,
            "query_mode": self.query_mode,
            "query_text": self.query_text,
            "success": self.success,
            "candidate_count": self.candidate_count,
            "accepted_count": self.accepted_count,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }


class PriorArtMetadataResolver:
    """
    Validation-first metadata completion for already-retrieved prior art.

    This class never changes which scientific literature query retrieved a
    work. It only attempts to complete metadata for explicitly supplied work
    IDs, using DOI-family identity first and exact normalized title second.
    """

    def __init__(
        self,
        providers: list[LiteratureSearchProvider],
        *,
        lookup_limit: int = 5,
        enable_title_fallback: bool = True,
    ) -> None:
        if not providers:
            raise ValueError(
                "metadata resolver requires at least one provider"
            )
        self.providers = list(providers)
        self.lookup_limit = max(1, int(lookup_limit))
        self.enable_title_fallback = bool(
            enable_title_fallback
        )

    def _queries_for_work(
        self,
        work: PriorArtWork,
    ) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []

        family = doi_family(work.doi)
        if family:
            result.append(("DOI_FAMILY", family))

        title = " ".join(str(work.title or "").split()).strip()
        if (
            self.enable_title_fallback
            and len(norm_title(title)) >= 20
        ):
            result.append(("EXACT_TITLE", title))

        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for mode, text in result:
            key = mode + "|" + text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append((mode, text))
        return deduped

    def resolve(
        self,
        packet: PriorArtPacket,
        *,
        target_work_ids: set[str],
    ) -> tuple[PriorArtPacket, dict[str, Any]]:
        by_id = {
            row.work_id: row
            for row in packet.works
        }

        missing_ids = sorted(
            work_id
            for work_id in target_work_ids
            if work_id not in by_id
        )
        targets = [
            by_id[work_id]
            for work_id in sorted(target_work_ids)
            if work_id in by_id
        ]

        attempts: list[ResolutionAttempt] = []
        accepted_candidates: list[PriorArtWork] = []
        work_audits: list[dict[str, Any]] = []

        for source in targets:
            source_candidates: list[PriorArtWork] = []
            candidate_audit: list[dict[str, Any]] = []

            for query_mode, query_text in self._queries_for_work(
                source
            ):
                query = LiteratureQuery(
                    query_id=stable_id(
                        "s227_metadata_query",
                        source.work_id,
                        query_mode,
                        query_text,
                    ),
                    hypothesis_id=(
                        "s227_metadata_resolution:"
                        + source.work_id
                    ),
                    claim_id=None,
                    query_kind="claim_exact_verification",
                    query_text=query_text,
                )

                for provider in self.providers:
                    started = time.perf_counter()
                    try:
                        rows = provider.search(
                            query,
                            limit=self.lookup_limit,
                        )
                        accepted = []
                        for candidate in rows:
                            ok, basis = candidate_matches_source(
                                source,
                                candidate,
                            )
                            candidate_audit.append(
                                {
                                    "provider":
                                        provider.provider_name,
                                    "query_mode":
                                        query_mode,
                                    "candidate_work_id":
                                        candidate.work_id,
                                    "candidate_title":
                                        candidate.title,
                                    "candidate_doi":
                                        candidate.doi,
                                    "candidate_abstract_available":
                                        bool(candidate.abstract),
                                    "accepted": ok,
                                    "identity_basis": basis,
                                }
                            )
                            if not ok:
                                continue
                            accepted.append(
                                inherited_candidate(
                                    candidate,
                                    source=source,
                                )
                            )

                        elapsed = (
                            time.perf_counter() - started
                        )
                        attempts.append(
                            ResolutionAttempt(
                                work_id=source.work_id,
                                provider=provider.provider_name,
                                query_mode=query_mode,
                                query_text=query_text,
                                success=True,
                                candidate_count=len(rows),
                                accepted_count=len(accepted),
                                elapsed_seconds=elapsed,
                            )
                        )
                        source_candidates.extend(accepted)
                    except Exception as exc:
                        elapsed = (
                            time.perf_counter() - started
                        )
                        attempts.append(
                            ResolutionAttempt(
                                work_id=source.work_id,
                                provider=provider.provider_name,
                                query_mode=query_mode,
                                query_text=query_text,
                                success=False,
                                candidate_count=0,
                                accepted_count=0,
                                elapsed_seconds=elapsed,
                                error=(
                                    f"{type(exc).__name__}: {exc}"
                                ),
                            )
                        )

            accepted_candidates.extend(source_candidates)

            source_plus = [source, *source_candidates]
            merged_rows, _ = canonicalize_prior_art_works(
                source_plus
            )

            resolved = None
            source_key = work_identity_key(source)
            for row in merged_rows:
                if work_identity_key(row) == source_key:
                    resolved = row
                    break

            work_audits.append(
                {
                    "work_id": source.work_id,
                    "source_title": source.title,
                    "source_doi": source.doi,
                    "source_doi_family":
                        doi_family(source.doi),
                    "source_is_supplementary":
                        is_supplementary_doi(source.doi),
                    "source_abstract_available":
                        bool(source.abstract),
                    "accepted_candidate_count":
                        len(source_candidates),
                    "resolved_abstract_available":
                        bool(resolved and resolved.abstract),
                    "resolved_doi":
                        resolved.doi if resolved else source.doi,
                    "resolved_title":
                        resolved.title if resolved else source.title,
                    "abstract_recovered":
                        bool(
                            not source.abstract
                            and resolved
                            and resolved.abstract
                        ),
                    "candidate_audit": candidate_audit,
                }
            )

        combined = [
            *packet.works,
            *accepted_candidates,
        ]
        canonical, supplementary_collapsed = (
            canonicalize_prior_art_works(combined)
        )
        canonical = sorted(
            canonical,
            key=lambda row: (
                -(row.citation_count or 0),
                -(row.year or 0),
                row.title.lower(),
            ),
        )

        body = {
            "schema_version": "prior-art-packet-v1",
            "packet_id": stable_id(
                "prior_art_packet",
                packet.packet_id,
                "s227-resolution",
                sha256_json(
                    [
                        {
                            "work_id": row.work_id,
                            "doi": row.doi,
                            "title": row.title,
                            "abstract": row.abstract,
                            "providers": row.providers,
                        }
                        for row in canonical
                    ]
                ),
            ),
            "source_portfolio_id":
                packet.source_portfolio_id,
            "source_query_plan_id":
                packet.source_query_plan_id,
            "searched_at_utc":
                packet.searched_at_utc,
            "providers_requested":
                list(packet.providers_requested),
            "works": [
                row.model_dump(mode="json")
                for row in canonical
            ],
            # These counts describe the validation packet after metadata
            # candidate injection. Scientific QueryExecution coverage remains
            # unchanged below.
            "executions": [
                row.model_dump(mode="json")
                for row in packet.executions
            ],
            "raw_work_count": (
                packet.raw_work_count
                + len(accepted_candidates)
            ),
            "canonical_work_count": len(canonical),
            "deduplicated_work_count": max(
                0,
                (
                    packet.raw_work_count
                    + len(accepted_candidates)
                    - len(canonical)
                ),
            ),
            "supplementary_records_collapsed":
                max(
                    packet.supplementary_records_collapsed,
                    supplementary_collapsed,
                ),
            "epistemic_usage":
                "prior_art_only_not_positive_premise",
        }
        resolved_packet = PriorArtPacket(
            **body,
            packet_sha256=sha256_json(body),
        )

        before_abstracts = sum(
            bool(row.abstract)
            for row in packet.works
        )
        after_abstracts = sum(
            bool(row.abstract)
            for row in resolved_packet.works
        )

        audit = {
            "schema_version":
                "prior-art-metadata-resolution-s227-v1",
            "source_packet_id": packet.packet_id,
            "resolved_packet_id":
                resolved_packet.packet_id,
            "requested_target_work_count":
                len(target_work_ids),
            "matched_target_work_count":
                len(targets),
            "missing_target_work_ids":
                missing_ids,
            "lookup_provider_names": [
                row.provider_name
                for row in self.providers
            ],
            "lookup_limit": self.lookup_limit,
            "enable_title_fallback":
                self.enable_title_fallback,
            "source_work_count": len(packet.works),
            "resolved_work_count":
                len(resolved_packet.works),
            "source_abstract_work_count":
                before_abstracts,
            "resolved_abstract_work_count":
                after_abstracts,
            "net_abstract_gain":
                after_abstracts - before_abstracts,
            "target_abstract_recovered_count":
                sum(
                    bool(row["abstract_recovered"])
                    for row in work_audits
                ),
            "target_supplementary_doi_count":
                sum(
                    bool(row["source_is_supplementary"])
                    for row in work_audits
                ),
            "accepted_metadata_candidate_count":
                len(accepted_candidates),
            "provider_attempt_count":
                len(attempts),
            "provider_failure_count":
                sum(not row.success for row in attempts),
            "attempts": [
                row.as_dict()
                for row in attempts
            ],
            "works": work_audits,
            "authority": {
                "posthoc_validation_only": True,
                "scientific_query_coverage_changed": False,
                "positive_premise_authority": False,
                "novelty_authority_created": False,
                "production_selection_authority": False,
            },
        }

        return resolved_packet, audit


__all__ = [
    "PriorArtMetadataResolver",
    "ResolutionAttempt",
    "candidate_matches_source",
    "doi_family",
    "is_supplementary_doi",
    "norm_doi",
    "norm_title",
    "work_identity_key",
]
