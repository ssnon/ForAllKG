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

from pipeline_core.literature.catalog_contracts import (
    CatalogQuery,
    CatalogQueryExecution,
    CatalogWork,
    LiteratureCatalogPacket,
)


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


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _strip_markup(value: Any) -> str | None:
    if value is None:
        return None
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    return text or None


def normalize_doi(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith("https://doi.org/"):
        text = text[len("https://doi.org/") :]
    if text.startswith("http://doi.org/"):
        text = text[len("http://doi.org/") :]
    if text.startswith("doi:"):
        text = text[4:]
    return text or None


def normalize_title(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9α-ω가-힣]+", " ", text)
    return " ".join(text.split())


_SUPPLEMENTARY_DOI_RE = re.compile(r"\.s\d+$", re.I)


def doi_family(value: Any) -> str | None:
    doi = normalize_doi(value)
    if not doi:
        return None
    return _SUPPLEMENTARY_DOI_RE.sub("", doi)


def is_supplementary_doi(value: Any) -> bool:
    doi = normalize_doi(value) or ""
    return bool(_SUPPLEMENTARY_DOI_RE.search(doi))


def _date_from_parts(
    parts: Any,
) -> tuple[int | None, str | None]:
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
) -> Any:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers=headers or {})
            with urlopen(
                request,
                timeout=timeout,
            ) as response:
                return json.loads(
                    response.read().decode("utf-8")
                )
        except HTTPError as exc:
            last = exc
            if (
                exc.code not in {429, 500, 502, 503, 504}
                or attempt >= retries
            ):
                raise
        except URLError as exc:
            last = exc
            if attempt >= retries:
                raise
        time.sleep(retry_backoff * (2**attempt))
    if last is not None:
        raise last
    raise RuntimeError("request failed without an exception")


class CatalogSearchProvider(Protocol):
    provider_name: str

    def search(
        self,
        query: CatalogQuery,
        *,
        limit: int,
    ) -> list[CatalogWork]:
        ...


class SemanticScholarCatalogProvider:
    """Neutral Semantic Scholar catalog adapter.

    This deliberately does not import PriorArtWork or any hypothesis/novelty
    contract. Retrieved metadata remains candidate-only until a later
    acquisition/materialization/extraction stage promotes a source into the
    positive-evidence lane.
    """

    provider_name = "semantic_scholar"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "SEMANTIC_SCHOLAR_API_KEY",
        timeout: float = 30.0,
        delay_seconds: float = 0.35,
    ) -> None:
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv(api_key_env)
        )
        self.timeout = float(timeout)
        self.delay_seconds = float(delay_seconds)

    def search(
        self,
        query: CatalogQuery,
        *,
        limit: int,
    ) -> list[CatalogWork]:
        fields = (
            "title,abstract,year,authors,venue,url,externalIds,"
            "citationCount,openAccessPdf,publicationDate,"
            "publicationTypes"
        )
        params = urlencode(
            {
                "query": query.query_text,
                "limit": max(1, min(100, int(limit))),
                "fields": fields,
            }
        )
        headers = {
            "User-Agent": (
                "GraphAgentsDAC-CorpusAcquisition/M1"
            )
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        started = time.perf_counter()
        payload = _request_json(
            (
                "https://api.semanticscholar.org/"
                "graph/v1/paper/search?"
                + params
            ),
            headers=headers,
            timeout=self.timeout,
        )
        elapsed = time.perf_counter() - started
        if self.delay_seconds > elapsed:
            time.sleep(self.delay_seconds - elapsed)

        rows: list[CatalogWork] = []
        items = (
            payload.get("data", [])
            if isinstance(payload, dict)
            else []
        )
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            external = item.get("externalIds") or {}
            doi = (
                normalize_doi(external.get("DOI"))
                if isinstance(external, dict)
                else None
            )
            authors = [
                str(row.get("name") or "").strip()
                for row in item.get("authors") or []
                if isinstance(row, dict)
                and str(row.get("name") or "").strip()
            ]
            oa = item.get("openAccessPdf") or {}
            oa_url = (
                str(oa.get("url") or "").strip()
                if isinstance(oa, dict)
                else ""
            )
            provider_id = str(item.get("paperId") or "")
            publication_types = [
                str(value).strip()
                for value in item.get("publicationTypes") or []
                if str(value).strip()
            ]
            rows.append(
                CatalogWork(
                    work_id=_stable_id(
                        "catalog_work",
                        doi or provider_id or title,
                    ),
                    title=title,
                    year=(
                        int(item["year"])
                        if item.get("year") is not None
                        else None
                    ),
                    publication_date=(
                        str(item.get("publicationDate"))
                        if item.get("publicationDate")
                        else None
                    ),
                    doi=doi,
                    url=(
                        str(item.get("url"))
                        if item.get("url")
                        else None
                    ),
                    open_access_url=oa_url or None,
                    abstract=_strip_markup(
                        item.get("abstract")
                    ),
                    authors=authors,
                    venue=(
                        str(item.get("venue"))
                        if item.get("venue")
                        else None
                    ),
                    citation_count=(
                        int(item["citationCount"])
                        if item.get("citationCount")
                        is not None
                        else None
                    ),
                    publication_types=publication_types,
                    providers=[self.provider_name],
                    provider_ids=(
                        {
                            self.provider_name: provider_id
                        }
                        if provider_id
                        else {}
                    ),
                    retrieval_query_ids=[query.query_id],
                    retrieval_axis_ids=[query.axis_id],
                )
            )
        return rows


class CrossrefCatalogProvider:
    """Neutral Crossref metadata adapter."""

    provider_name = "crossref"

    def __init__(
        self,
        *,
        mailto: str | None = None,
        mailto_env: str = "CROSSREF_MAILTO",
        timeout: float = 30.0,
        delay_seconds: float = 0.10,
    ) -> None:
        self.mailto = (
            mailto
            if mailto is not None
            else os.getenv(mailto_env)
        )
        self.timeout = float(timeout)
        self.delay_seconds = float(delay_seconds)

    def search(
        self,
        query: CatalogQuery,
        *,
        limit: int,
    ) -> list[CatalogWork]:
        params: dict[str, Any] = {
            "query.bibliographic": query.query_text,
            "rows": max(1, min(100, int(limit))),
        }
        if self.mailto:
            params["mailto"] = self.mailto
        headers = {
            "User-Agent": (
                "GraphAgentsDAC-CorpusAcquisition/M1"
            )
        }
        started = time.perf_counter()
        payload = _request_json(
            "https://api.crossref.org/works?"
            + urlencode(params),
            headers=headers,
            timeout=self.timeout,
        )
        elapsed = time.perf_counter() - started
        if self.delay_seconds > elapsed:
            time.sleep(self.delay_seconds - elapsed)

        message = (
            payload.get("message", {})
            if isinstance(payload, dict)
            else {}
        )
        items = (
            message.get("items", [])
            if isinstance(message, dict)
            else []
        )
        rows: list[CatalogWork] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            titles = item.get("title") or []
            title = str(
                titles[0] if titles else ""
            ).strip()
            if not title:
                continue
            doi = normalize_doi(item.get("DOI"))
            date_source = (
                item.get("published-online")
                or item.get("published-print")
                or item.get("published")
                or {}
            )
            year, publication_date = _date_from_parts(
                date_source.get("date-parts", [])
                if isinstance(date_source, dict)
                else []
            )
            authors: list[str] = []
            for row in item.get("author") or []:
                if not isinstance(row, dict):
                    continue
                name = " ".join(
                    value
                    for value in (
                        str(row.get("given") or "").strip(),
                        str(row.get("family") or "").strip(),
                    )
                    if value
                )
                if name:
                    authors.append(name)
            containers = item.get("container-title") or []
            provider_id = (
                doi
                or str(item.get("URL") or title)
            )
            publication_type = str(
                item.get("type") or ""
            ).strip()
            rows.append(
                CatalogWork(
                    work_id=_stable_id(
                        "catalog_work",
                        doi or provider_id,
                    ),
                    title=title,
                    year=year,
                    publication_date=publication_date,
                    doi=doi,
                    url=(
                        str(item.get("URL"))
                        if item.get("URL")
                        else None
                    ),
                    abstract=_strip_markup(
                        item.get("abstract")
                    ),
                    authors=authors,
                    venue=(
                        str(containers[0])
                        if containers
                        else None
                    ),
                    citation_count=(
                        int(item["is-referenced-by-count"])
                        if item.get(
                            "is-referenced-by-count"
                        )
                        is not None
                        else None
                    ),
                    publication_types=(
                        [publication_type]
                        if publication_type
                        else []
                    ),
                    providers=[self.provider_name],
                    provider_ids={
                        self.provider_name: provider_id
                    },
                    retrieval_query_ids=[query.query_id],
                    retrieval_axis_ids=[query.axis_id],
                )
            )
        return rows


def _prefer_doi(
    left: str | None,
    right: str | None,
) -> str | None:
    values = [
        value
        for value in (left, right)
        if value
    ]
    if not values:
        return None
    main = [
        value
        for value in values
        if not is_supplementary_doi(value)
    ]
    return main[0] if main else values[0]


def _canonical_work_id(work: CatalogWork) -> str:
    doi = doi_family(work.doi)
    if doi:
        return _stable_id("catalog_work", "doi", doi)

    title = normalize_title(work.title)

    # Exact-title canonical identity is only safe when the normalized
    # title carries enough information. Non-Latin titles can collapse
    # to very short strings such as "sers"; using those strings as
    # global identity keys can merge unrelated works.
    if len(title) >= 20:
        return _stable_id(
            "catalog_work",
            "title",
            title,
        )

    # Provider adapters already assign deterministic provider-derived
    # work IDs. Preserve that stronger identity for weak-title,
    # DOI-less records rather than manufacturing a global title ID.
    return work.work_id


def _merge_work(
    left: CatalogWork,
    right: CatalogWork,
) -> CatalogWork:
    abstract_candidates = [
        value
        for value in (left.abstract, right.abstract)
        if value
    ]
    abstract = (
        max(abstract_candidates, key=len)
        if abstract_candidates
        else None
    )
    title = (
        left.title
        if len(left.title) >= len(right.title)
        else right.title
    )
    merged = CatalogWork(
        work_id=left.work_id,
        title=title,
        year=(
            left.year
            if left.year is not None
            else right.year
        ),
        publication_date=(
            left.publication_date
            or right.publication_date
        ),
        doi=_prefer_doi(left.doi, right.doi),
        url=left.url or right.url,
        open_access_url=(
            left.open_access_url
            or right.open_access_url
        ),
        abstract=abstract,
        authors=sorted(
            set(left.authors) | set(right.authors)
        ),
        venue=left.venue or right.venue,
        citation_count=max(
            [
                value
                for value in (
                    left.citation_count,
                    right.citation_count,
                )
                if value is not None
            ],
            default=None,
        ),
        publication_types=sorted(
            set(left.publication_types)
            | set(right.publication_types)
        ),
        providers=sorted(
            set(left.providers) | set(right.providers)
        ),
        provider_ids={
            **left.provider_ids,
            **right.provider_ids,
        },
        retrieval_query_ids=sorted(
            set(left.retrieval_query_ids)
            | set(right.retrieval_query_ids)
        ),
        retrieval_axis_ids=sorted(
            set(left.retrieval_axis_ids)
            | set(right.retrieval_axis_ids)
        ),
    )
    return merged.model_copy(
        update={"work_id": _canonical_work_id(merged)}
    )


def canonicalize_catalog_works(
    raw_works: list[CatalogWork],
) -> tuple[list[CatalogWork], int]:
    """Canonicalize provider rows without assigning scientific meaning.

    DOI families are merged first; exact normalized-title duplicates are
    merged second. The operation only consolidates bibliographic records.
    """

    by_key: dict[str, CatalogWork] = {}
    family_counts: dict[str, int] = {}
    supplementary_families: set[str] = set()

    for work in sorted(
        raw_works,
        key=lambda row: (
            doi_family(row.doi) or "",
            normalize_title(row.title),
            row.work_id,
        ),
    ):
        family = doi_family(work.doi)
        if family:
            family_counts[family] = (
                family_counts.get(family, 0) + 1
            )
            if is_supplementary_doi(work.doi):
                supplementary_families.add(family)
            key = "doi_family:" + family
        else:
            title = normalize_title(work.title)

            # A weak normalized title is not a safe bibliographic
            # identity. Keep provider-derived records separate unless
            # the normalized exact title is sufficiently specific.
            if len(title) >= 20:
                key = (
                    "title:"
                    + hashlib.sha256(
                        title.encode("utf-8")
                    ).hexdigest()[:24]
                )
            else:
                key = "provider_work:" + work.work_id

        if key in by_key:
            by_key[key] = _merge_work(
                by_key[key],
                work,
            )
        else:
            by_key[key] = work.model_copy(
                update={
                    "work_id": _canonical_work_id(work)
                }
            )

    # Catch a DOI-bearing record and a DOI-less provider record that share
    # the same sufficiently specific normalized title.
    by_title: dict[str, CatalogWork] = {}
    passthrough: list[CatalogWork] = []
    for work in by_key.values():
        title = normalize_title(work.title)
        if len(title) < 20:
            passthrough.append(work)
            continue
        if title in by_title:
            by_title[title] = _merge_work(
                by_title[title],
                work,
            )
        else:
            by_title[title] = work

    canonical = list(by_title.values()) + passthrough
    canonical = sorted(
        canonical,
        key=lambda row: (
            -(row.citation_count or 0),
            -(row.year or 0),
            row.title.casefold(),
            row.work_id,
        ),
    )
    supplementary_collapsed = sum(
        max(0, count - 1)
        for family, count in family_counts.items()
        if family in supplementary_families
    )
    return canonical, supplementary_collapsed


@dataclass(frozen=True)
class CatalogRetrievalOutcome:
    packet: LiteratureCatalogPacket


class LiteratureCatalogRetriever:
    def __init__(
        self,
        providers: list[CatalogSearchProvider],
        *,
        results_per_query: int = 50,
        progress_callback=None,
    ) -> None:
        if not providers:
            raise ValueError(
                "at least one catalog provider is required"
            )
        self.providers = list(providers)
        self.results_per_query = int(
            results_per_query
        )
        self.progress_callback = progress_callback

    def retrieve(
        self,
        *,
        profile_id: str,
        queries: list[CatalogQuery],
    ) -> CatalogRetrievalOutcome:
        raw_works: list[CatalogWork] = []
        executions: list[
            CatalogQueryExecution
        ] = []

        total_executions = len(queries) * len(self.providers)
        completed_executions = 0

        for query in queries:
            for provider in self.providers:
                current_index = completed_executions + 1
                if self.progress_callback is not None:
                    self.progress_callback(
                        {
                            "stage": "m1_retrieval",
                            "event": "start",
                            "current": current_index,
                            "total": total_executions,
                            "query_id": query.query_id,
                            "axis_id": query.axis_id,
                            "query_text": query.query_text,
                            "provider": provider.provider_name,
                        }
                    )
                started = time.perf_counter()
                try:
                    rows = provider.search(
                        query,
                        limit=self.results_per_query,
                    )
                    raw_works.extend(rows)
                    executions.append(
                        CatalogQueryExecution(
                            query_id=query.query_id,
                            axis_id=query.axis_id,
                            provider=(
                                provider.provider_name
                            ),
                            success=True,
                            result_count=len(rows),
                            elapsed_seconds=(
                                time.perf_counter()
                                - started
                            ),
                        )
                    )
                except Exception as exc:
                    executions.append(
                        CatalogQueryExecution(
                            query_id=query.query_id,
                            axis_id=query.axis_id,
                            provider=(
                                provider.provider_name
                            ),
                            success=False,
                            result_count=0,
                            elapsed_seconds=(
                                time.perf_counter()
                                - started
                            ),
                            error=(
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                        )
                    )
                finally:
                    completed_executions += 1
                    if self.progress_callback is not None:
                        last = executions[-1]
                        self.progress_callback(
                            {
                                "stage": "m1_retrieval",
                                "event": "complete",
                                "current": completed_executions,
                                "total": total_executions,
                                "query_id": query.query_id,
                                "axis_id": query.axis_id,
                                "query_text": query.query_text,
                                "provider": provider.provider_name,
                                "success": bool(last.success),
                                "result_count": int(last.result_count),
                                "elapsed_seconds": float(last.elapsed_seconds),
                                "error": last.error,
                            }
                        )

        searched_at = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        )
        canonical, supplementary_collapsed = (
            canonicalize_catalog_works(raw_works)
        )
        body = {
            "schema_version": (
                "literature-catalog-packet-v1"
            ),
            "catalog_id": _stable_id(
                "literature_catalog",
                profile_id,
                searched_at,
                *[
                    row.work_id
                    for row in canonical
                ],
            ),
            "acquisition_profile_id": profile_id,
            "searched_at_utc": searched_at,
            "providers_requested": [
                provider.provider_name
                for provider in self.providers
            ],
            "queries": [
                row.model_dump(mode="json")
                for row in queries
            ],
            "works": [
                row.model_dump(mode="json")
                for row in canonical
            ],
            "executions": [
                row.model_dump(mode="json")
                for row in executions
            ],
            "raw_work_count": len(raw_works),
            "canonical_work_count": len(
                canonical
            ),
            "deduplicated_work_count": (
                len(raw_works) - len(canonical)
            ),
            "supplementary_records_collapsed": (
                supplementary_collapsed
            ),
            "epistemic_usage": (
                "candidate_source_only_"
                "not_positive_premise"
            ),
        }
        packet = LiteratureCatalogPacket(
            **body,
            catalog_sha256=_sha256_json(body),
        )
        return CatalogRetrievalOutcome(
            packet=packet
        )
