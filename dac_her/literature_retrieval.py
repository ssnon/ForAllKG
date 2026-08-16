from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dac_her.provider_resilience import (
    ProviderRequestPacer,
    RequestTelemetry,
    resilient_request_json,
)

from dac_her.external_novelty_contracts import (
    LiteratureQuery,
    LiteratureQueryPlan,
    PriorArtPacket,
    PriorArtWork,
    QueryExecution,
)


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _strip_markup(value: Any) -> str | None:
    if value is None:
        return None
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    return text or None


def _norm_doi(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith("https://doi.org/"):
        text = text[len("https://doi.org/") :]
    if text.startswith("doi:"):
        text = text[4:]
    return text or None


def _norm_title(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9α-ω가-힣]+", " ", text)
    return " ".join(text.split())


_SUPPLEMENTARY_DOI_RE = re.compile(r"\.s\d+$", re.I)


def _doi_family(value: Any) -> str | None:
    doi = _norm_doi(value)
    if not doi:
        return None
    return _SUPPLEMENTARY_DOI_RE.sub("", doi)


def _is_supplementary_doi(value: Any) -> bool:
    doi = _norm_doi(value) or ""
    return bool(_SUPPLEMENTARY_DOI_RE.search(doi))


def _work_key(work: PriorArtWork) -> str:
    family = _doi_family(work.doi)
    if family:
        return "doi_family:" + family
    title = _norm_title(work.title)
    if title:
        return "title:" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:24]
    return work.work_id


def _date_from_parts(parts: Any) -> tuple[int | None, str | None]:
    try:
        row = parts[0]
        if not row:
            return None, None
        year = int(row[0])
        month = int(row[1]) if len(row) > 1 else 1
        day = int(row[2]) if len(row) > 2 else 1
        return year, f"{year:04d}-{month:02d}-{day:02d}"
    except Exception:
        return None, None


def _request_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    retries: int = 2,
    retry_backoff: float = 1.0,
    pacer: ProviderRequestPacer | None = None,
    telemetry: RequestTelemetry | None = None,
) -> Any:
    return resilient_request_json(
        url,
        headers=headers,
        timeout=timeout,
        retries=retries,
        retry_backoff=retry_backoff,
        pacer=pacer,
        telemetry=telemetry,
    )


class LiteratureSearchProvider(Protocol):
    provider_name: str

    def search(self, query: LiteratureQuery, *, limit: int) -> list[PriorArtWork]: ...


class SemanticScholarProvider:
    """Semantic Scholar Academic Graph relevance-search adapter.

    Uses the official /graph/v1/paper/search endpoint. An API key is optional
    but, when supplied, is sent in the documented x-api-key header.
    """

    provider_name = "semantic_scholar"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "SEMANTIC_SCHOLAR_API_KEY",
        timeout: float = 30.0,
        delay_seconds: float = 0.35,
        minimum_interval_seconds: float = 1.05,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv(api_key_env)
        self.timeout = float(timeout)
        self.delay_seconds = float(delay_seconds)
        self.minimum_interval_seconds = max(
            0.0,
            float(minimum_interval_seconds),
            self.delay_seconds,
        )
        self._request_pacer = ProviderRequestPacer(
            self.minimum_interval_seconds
        )
        self._request_telemetry = RequestTelemetry()

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "api_key_configured": bool(self.api_key),
            "minimum_interval_seconds": self.minimum_interval_seconds,
            "telemetry": self._request_telemetry.snapshot(),
        }

    def search(self, query: LiteratureQuery, *, limit: int) -> list[PriorArtWork]:
        fields = (
            "title,abstract,year,authors,venue,url,externalIds,citationCount,"
            "openAccessPdf,publicationDate"
        )
        params = urlencode(
            {
                "query": query.query_text,
                "limit": max(1, min(100, int(limit))),
                "fields": fields,
            }
        )
        headers = {"User-Agent": "GraphAgentsDAC-ExternalNovelty/alpha5"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        payload = _request_json(
            "https://api.semanticscholar.org/graph/v1/paper/search?" + params,
            headers=headers,
            timeout=self.timeout,
            pacer=self._request_pacer,
            telemetry=self._request_telemetry,
        )

        rows: list[PriorArtWork] = []
        for item in payload.get("data", []) if isinstance(payload, dict) else []:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            external = item.get("externalIds") or {}
            doi = _norm_doi(external.get("DOI")) if isinstance(external, dict) else None
            authors = [
                str(row.get("name") or "").strip()
                for row in item.get("authors") or []
                if isinstance(row, dict) and str(row.get("name") or "").strip()
            ]
            oa = item.get("openAccessPdf") or {}
            oa_url = str(oa.get("url") or "").strip() if isinstance(oa, dict) else ""
            provider_id = str(item.get("paperId") or "")
            work_id = _stable_id("prior_art_work", doi or provider_id or title)
            rows.append(
                PriorArtWork(
                    work_id=work_id,
                    title=title,
                    year=(int(item["year"]) if item.get("year") is not None else None),
                    publication_date=(str(item.get("publicationDate")) if item.get("publicationDate") else None),
                    doi=doi,
                    url=(str(item.get("url")) if item.get("url") else None),
                    open_access_url=oa_url or None,
                    abstract=_strip_markup(item.get("abstract")),
                    authors=authors,
                    venue=(str(item.get("venue")) if item.get("venue") else None),
                    citation_count=(
                        int(item["citationCount"])
                        if item.get("citationCount") is not None
                        else None
                    ),
                    providers=[self.provider_name],
                    provider_ids=({self.provider_name: provider_id} if provider_id else {}),
                    retrieval_query_ids=[query.query_id],
                    retrieval_claim_ids=([query.claim_id] if query.claim_id else []),
                )
            )
        return rows



class OpenAlexProviderError(RuntimeError):
    """Sanitized OpenAlex request failure.

    OpenAlex requires api_key in the query string. This exception deliberately
    omits request URLs so QueryExecution.error cannot persist the API key.
    """


def reconstruct_openalex_abstract(
    value: Any,
) -> str | None:
    """Reconstruct OpenAlex abstract_inverted_index deterministically.

    Malformed/colliding position maps return None rather than guessing.
    """
    if not isinstance(value, dict) or not value:
        return None

    by_position: dict[int, str] = {}
    for token, raw_positions in value.items():
        word = str(token).strip()
        if not word or not isinstance(raw_positions, list):
            return None
        for raw_position in raw_positions:
            if (
                not isinstance(raw_position, int)
                or isinstance(raw_position, bool)
                or raw_position < 0
            ):
                return None
            existing = by_position.get(raw_position)
            if existing is not None and existing != word:
                return None
            by_position[raw_position] = word

    if not by_position:
        return None

    # Avoid constructing pathological sparse records while preserving ordinary
    # OpenAlex abstracts. We join only observed positions; gaps are not filled.
    if len(by_position) > 100_000:
        return None
    ordered = [
        by_position[position]
        for position in sorted(by_position)
    ]
    text = " ".join(ordered).strip()
    return text or None


class OpenAlexProvider:
    """OpenAlex /works search adapter.

    Uses the same frozen LiteratureQuery.query_text as other providers. The
    API key is supplied only to the outbound query string and is never stored
    in PriorArtWork, provider-plan provenance, or sanitized exceptions.
    """

    provider_name = "openalex"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENALEX_API_KEY",
        timeout: float = 30.0,
        minimum_interval_seconds: float = 0.0,
    ) -> None:
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv(api_key_env)
        )
        self.api_key = str(self.api_key or "").strip()
        if not self.api_key:
            raise ValueError(
                "OpenAlexProvider requires OPENALEX_API_KEY."
            )
        self.timeout = float(timeout)
        self.minimum_interval_seconds = max(
            0.0,
            float(minimum_interval_seconds),
        )
        self._request_pacer = ProviderRequestPacer(
            self.minimum_interval_seconds
        )
        self._request_telemetry = RequestTelemetry()

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "api_key_configured": bool(self.api_key),
            "minimum_interval_seconds":
                self.minimum_interval_seconds,
            "telemetry":
                self._request_telemetry.snapshot(),
        }

    def _safe_request(
        self,
        url: str,
    ) -> Any:
        try:
            return _request_json(
                url,
                timeout=self.timeout,
                pacer=self._request_pacer,
                telemetry=self._request_telemetry,
            )
        except HTTPError as exc:
            reason = str(
                getattr(exc, "reason", None)
                or getattr(exc, "msg", None)
                or "HTTP error"
            )
            raise OpenAlexProviderError(
                f"HTTPError: HTTP Error {int(exc.code)}: {reason}"
            ) from None
        except URLError as exc:
            reason = getattr(exc, "reason", None)
            reason_type = (
                type(reason).__name__
                if reason is not None
                else "unknown"
            )
            raise OpenAlexProviderError(
                f"URLError: {reason_type}"
            ) from None
        except Exception as exc:
            raise OpenAlexProviderError(
                f"{type(exc).__name__}: OpenAlex request failed"
            ) from None

    def search(
        self,
        query: LiteratureQuery,
        *,
        limit: int,
    ) -> list[PriorArtWork]:
        params = urlencode(
            {
                "search": query.query_text,
                "per_page": max(
                    1,
                    min(100, int(limit)),
                ),
                "api_key": self.api_key,
                "select": (
                    "id,doi,display_name,publication_year,"
                    "publication_date,cited_by_count,"
                    "abstract_inverted_index,authorships,"
                    "primary_location,best_oa_location"
                ),
            }
        )
        payload = self._safe_request(
            "https://api.openalex.org/works?"
            + params
        )

        results = (
            payload.get("results", [])
            if isinstance(payload, dict)
            else []
        )
        rows: list[PriorArtWork] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = str(
                item.get("display_name")
                or item.get("title")
                or ""
            ).strip()
            if not title:
                continue

            provider_id = str(
                item.get("id") or ""
            ).strip()
            doi = _norm_doi(
                item.get("doi")
            )

            authors: list[str] = []
            for authorship in (
                item.get("authorships")
                or []
            ):
                if not isinstance(
                    authorship,
                    dict,
                ):
                    continue
                author = (
                    authorship.get("author")
                    or {}
                )
                if not isinstance(author, dict):
                    continue
                name = str(
                    author.get(
                        "display_name"
                    )
                    or ""
                ).strip()
                if name:
                    authors.append(name)

            primary = (
                item.get(
                    "primary_location"
                )
                or {}
            )
            if not isinstance(primary, dict):
                primary = {}
            best_oa = (
                item.get(
                    "best_oa_location"
                )
                or {}
            )
            if not isinstance(best_oa, dict):
                best_oa = {}

            source = (
                primary.get("source")
                or {}
            )
            if not isinstance(source, dict):
                source = {}

            primary_url = str(
                primary.get(
                    "landing_page_url"
                )
                or ""
            ).strip()
            open_access_url = str(
                best_oa.get("pdf_url")
                or best_oa.get(
                    "landing_page_url"
                )
                or ""
            ).strip()

            year = item.get(
                "publication_year"
            )
            citation_count = item.get(
                "cited_by_count"
            )

            rows.append(
                PriorArtWork(
                    work_id=_stable_id(
                        "prior_art_work",
                        doi
                        or provider_id
                        or title,
                    ),
                    title=title,
                    year=(
                        int(year)
                        if year is not None
                        else None
                    ),
                    publication_date=(
                        str(
                            item.get(
                                "publication_date"
                            )
                        )
                        if item.get(
                            "publication_date"
                        )
                        else None
                    ),
                    doi=doi,
                    url=(
                        primary_url
                        or provider_id
                        or None
                    ),
                    open_access_url=(
                        open_access_url
                        or None
                    ),
                    abstract=(
                        reconstruct_openalex_abstract(
                            item.get(
                                "abstract_inverted_index"
                            )
                        )
                    ),
                    authors=authors,
                    venue=(
                        str(
                            source.get(
                                "display_name"
                            )
                        ).strip()
                        or None
                    ),
                    citation_count=(
                        int(citation_count)
                        if citation_count
                        is not None
                        else None
                    ),
                    providers=[
                        self.provider_name
                    ],
                    provider_ids=(
                        {
                            self.provider_name:
                                provider_id
                        }
                        if provider_id
                        else {}
                    ),
                    retrieval_query_ids=[
                        query.query_id
                    ],
                    retrieval_claim_ids=(
                        [query.claim_id]
                        if query.claim_id
                        else []
                    ),
                )
            )
        return rows


class CrossrefProvider:
    """Crossref REST /works adapter used as a complementary metadata source."""

    provider_name = "crossref"

    def __init__(
        self,
        *,
        mailto: str | None = None,
        mailto_env: str = "CROSSREF_MAILTO",
        timeout: float = 30.0,
        delay_seconds: float = 0.10,
    ) -> None:
        self.mailto = mailto if mailto is not None else os.getenv(mailto_env)
        self.timeout = float(timeout)
        self.delay_seconds = float(delay_seconds)

    def search(self, query: LiteratureQuery, *, limit: int) -> list[PriorArtWork]:
        params: dict[str, Any] = {
            "query.bibliographic": query.query_text,
            "rows": max(1, min(100, int(limit))),
        }
        if self.mailto:
            params["mailto"] = self.mailto
        headers = {"User-Agent": "GraphAgentsDAC-ExternalNovelty/alpha5"}
        started = time.perf_counter()
        payload = _request_json(
            "https://api.crossref.org/works?" + urlencode(params),
            headers=headers,
            timeout=self.timeout,
        )
        elapsed = time.perf_counter() - started
        if self.delay_seconds > elapsed:
            time.sleep(self.delay_seconds - elapsed)

        message = payload.get("message", {}) if isinstance(payload, dict) else {}
        rows: list[PriorArtWork] = []
        for item in message.get("items", []) if isinstance(message, dict) else []:
            titles = item.get("title") or []
            title = str(titles[0] if titles else "").strip()
            if not title:
                continue
            doi = _norm_doi(item.get("DOI"))
            year, publication_date = _date_from_parts(
                (item.get("published-online") or item.get("published-print") or {}).get("date-parts", [])
            )
            authors: list[str] = []
            for row in item.get("author") or []:
                if not isinstance(row, dict):
                    continue
                name = " ".join(
                    x for x in [str(row.get("given") or "").strip(), str(row.get("family") or "").strip()] if x
                )
                if name:
                    authors.append(name)
            containers = item.get("container-title") or []
            provider_id = doi or str(item.get("URL") or title)
            rows.append(
                PriorArtWork(
                    work_id=_stable_id("prior_art_work", doi or provider_id),
                    title=title,
                    year=year,
                    publication_date=publication_date,
                    doi=doi,
                    url=(str(item.get("URL")) if item.get("URL") else None),
                    abstract=_strip_markup(item.get("abstract")),
                    authors=authors,
                    venue=(str(containers[0]) if containers else None),
                    citation_count=(
                        int(item["is-referenced-by-count"])
                        if item.get("is-referenced-by-count") is not None
                        else None
                    ),
                    providers=[self.provider_name],
                    provider_ids={self.provider_name: provider_id},
                    retrieval_query_ids=[query.query_id],
                    retrieval_claim_ids=([query.claim_id] if query.claim_id else []),
                )
            )
        return rows


def _prefer_doi(left: str | None, right: str | None) -> str | None:
    values = [x for x in [left, right] if x]
    if not values:
        return None
    main = [x for x in values if not _is_supplementary_doi(x)]
    return main[0] if main else values[0]


def _merge_work(left: PriorArtWork, right: PriorArtWork) -> PriorArtWork:
    abstract_candidates = [x for x in [left.abstract, right.abstract] if x]
    abstract = max(abstract_candidates, key=len) if abstract_candidates else None
    title = left.title if len(left.title) >= len(right.title) else right.title
    return PriorArtWork(
        work_id=left.work_id,
        title=title,
        year=left.year if left.year is not None else right.year,
        publication_date=left.publication_date or right.publication_date,
        doi=_prefer_doi(left.doi, right.doi),
        url=left.url or right.url,
        open_access_url=left.open_access_url or right.open_access_url,
        abstract=abstract,
        authors=sorted(set(left.authors) | set(right.authors)),
        venue=left.venue or right.venue,
        citation_count=max(
            [x for x in [left.citation_count, right.citation_count] if x is not None],
            default=None,
        ),
        providers=sorted(set(left.providers) | set(right.providers)),
        provider_ids={**left.provider_ids, **right.provider_ids},
        retrieval_query_ids=sorted(
            set(left.retrieval_query_ids) | set(right.retrieval_query_ids)
        ),
        retrieval_claim_ids=sorted(
            set(left.retrieval_claim_ids) | set(right.retrieval_claim_ids)
        ),
    )


def _deduplicate_by_title(works: list[PriorArtWork]) -> list[PriorArtWork]:
    """Conservatively collapse exact normalized-title duplicates.

    DOI-family identity is stronger than title identity. After DOI-family
    merging, exact-title fallback is permitted only when the title group does
    not contain multiple distinct non-empty DOI families.

    Rules:
    - zero DOI families: exact-title fallback may merge DOI-less duplicates;
    - one DOI family: DOI-less records may merge into that unambiguous family;
    - two or more DOI families: no title-based merging is performed for the
      group, because title equality cannot override conflicting strong IDs.

    Very short titles are not merged to avoid collisions on generic names.
    """
    by_title: dict[str, list[PriorArtWork]] = {}
    passthrough: list[PriorArtWork] = []

    for work in works:
        title = _norm_title(work.title)
        if len(title) < 20:
            passthrough.append(work)
            continue
        by_title.setdefault(title, []).append(work)

    canonical: list[PriorArtWork] = []
    for rows in by_title.values():
        families = {
            family
            for family in (_doi_family(row.doi) for row in rows)
            if family
        }

        if len(families) >= 2:
            # Conflicting strong identifiers: preserve every record that
            # survived DOI-family merging. A title alone is not sufficient
            # authority to collapse these records.
            canonical.extend(rows)
            continue

        merged = rows[0]
        for row in rows[1:]:
            merged = _merge_work(merged, row)
        canonical.append(merged)

    return canonical + passthrough


def _canonicalize_works(
    raw_works: list[PriorArtWork],
) -> tuple[list[PriorArtWork], int]:
    by_key: dict[str, PriorArtWork] = {}
    family_counts: dict[str, int] = {}
    supplementary_families: list[str] = []
    for work in raw_works:
        family = _doi_family(work.doi)
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1
            if _is_supplementary_doi(work.doi):
                supplementary_families.append(family)
        key = _work_key(work)
        if key in by_key:
            by_key[key] = _merge_work(by_key[key], work)
        else:
            by_key[key] = work
    canonical = _deduplicate_by_title(list(by_key.values()))
    supplementary_family_set = set(supplementary_families)
    supplementary_collapsed = sum(
        max(0, count - 1)
        for family, count in family_counts.items()
        if family in supplementary_family_set
    )
    return canonical, supplementary_collapsed


def canonicalize_prior_art_packet(packet: PriorArtPacket) -> PriorArtPacket:
    """Re-canonicalize an alpha5 packet without repeating network retrieval.

    Useful for alpha5.1 benchmark reruns. Existing packet records have already
    undergone some provider merging, so raw_work_count here means the number of
    records available in the reused packet, not the unrecoverable provider-row
    count from the original run.
    """
    raw = list(packet.works)
    works, supplementary_collapsed = _canonicalize_works(raw)
    works = sorted(
        works,
        key=lambda row: (-(row.citation_count or 0), -(row.year or 0), row.title.lower()),
    )
    packet_id = _stable_id(
        "prior_art_packet",
        packet.source_query_plan_id,
        packet.searched_at_utc,
        "alpha5.1-canonical",
        *[row.work_id for row in works],
    )
    body = {
        "schema_version": "prior-art-packet-v1",
        "packet_id": packet_id,
        "source_portfolio_id": packet.source_portfolio_id,
        "source_query_plan_id": packet.source_query_plan_id,
        "searched_at_utc": packet.searched_at_utc,
        "providers_requested": list(packet.providers_requested),
        "works": [row.model_dump(mode="json") for row in works],
        "executions": [row.model_dump(mode="json") for row in packet.executions],
        "raw_work_count": len(raw),
        "canonical_work_count": len(works),
        "deduplicated_work_count": max(0, len(raw) - len(works)),
        "supplementary_records_collapsed": supplementary_collapsed,
        "epistemic_usage": "prior_art_only_not_positive_premise",
    }
    return PriorArtPacket(**body, packet_sha256=_sha256_json(body))


@dataclass(frozen=True)
class RetrievalOutcome:
    packet: PriorArtPacket


class LiteratureRetriever:
    def __init__(
        self,
        providers: list[LiteratureSearchProvider],
        *,
        results_per_query: int = 12,
    ) -> None:
        if not providers:
            raise ValueError("at least one literature provider is required")
        self.providers = list(providers)
        self.results_per_query = int(results_per_query)

    def retrieve(self, plan: LiteratureQueryPlan) -> RetrievalOutcome:
        raw_works: list[PriorArtWork] = []
        executions: list[QueryExecution] = []

        for query in plan.queries:
            for provider in self.providers:
                started = time.perf_counter()
                try:
                    rows = provider.search(query, limit=self.results_per_query)
                    raw_works.extend(rows)
                    elapsed = time.perf_counter() - started
                    executions.append(
                        QueryExecution(
                            query_id=query.query_id,
                            provider=provider.provider_name,
                            success=True,
                            result_count=len(rows),
                            elapsed_seconds=elapsed,
                        )
                    )
                except Exception as exc:
                    elapsed = time.perf_counter() - started
                    executions.append(
                        QueryExecution(
                            query_id=query.query_id,
                            provider=provider.provider_name,
                            success=False,
                            result_count=0,
                            elapsed_seconds=elapsed,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )

        searched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        canonical, supplementary_records_collapsed = _canonicalize_works(raw_works)
        works = sorted(
            canonical,
            key=lambda row: (
                -(row.citation_count or 0),
                -(row.year or 0),
                row.title.lower(),
            ),
        )
        raw_work_count = len(raw_works)

        packet_id = _stable_id(
            "prior_art_packet",
            plan.plan_id,
            searched_at,
            *[row.work_id for row in works],
        )
        body = {
            "schema_version": "prior-art-packet-v1",
            "packet_id": packet_id,
            "source_portfolio_id": plan.source_portfolio_id,
            "source_query_plan_id": plan.plan_id,
            "searched_at_utc": searched_at,
            "providers_requested": [row.provider_name for row in self.providers],
            "works": [row.model_dump(mode="json") for row in works],
            "executions": [row.model_dump(mode="json") for row in executions],
            "raw_work_count": raw_work_count,
            "canonical_work_count": len(works),
            "deduplicated_work_count": max(0, raw_work_count - len(works)),
            "supplementary_records_collapsed": supplementary_records_collapsed,
            "epistemic_usage": "prior_art_only_not_positive_premise",
        }
        return RetrievalOutcome(
            packet=PriorArtPacket(**body, packet_sha256=_sha256_json(body))
        )
