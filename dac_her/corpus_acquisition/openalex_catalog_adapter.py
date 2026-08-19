from __future__ import annotations

import hashlib
import os
from typing import Any

from dac_her.literature_catalog import normalize_doi
from dac_her.literature_catalog_contracts import CatalogQuery, CatalogWork
from dac_her.literature_discovery.providers import (
    LiteratureSearchRequest,
    OpenAlexProvider,
)


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


class OpenAlexCatalogProvider:
    """Adapter from the existing OpenAlex discovery provider to neutral M1.

    Only a location explicitly returned as ``pdf_url`` is copied into
    ``CatalogWork.open_access_url``. Landing pages are kept as bibliographic
    URLs instead of being promoted to automatic direct-PDF candidates.
    """

    provider_name = "openalex"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        mailto: str | None = None,
        api_key_env: str = "OPENALEX_API_KEY",
        mailto_env: str = "OPENALEX_MAILTO",
        fallback_mailto_env: str = "CROSSREF_MAILTO",
        timeout: float = 30.0,
        provider: OpenAlexProvider | None = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.getenv(api_key_env)
        resolved_mailto = mailto
        if resolved_mailto is None:
            resolved_mailto = os.getenv(mailto_env) or os.getenv(fallback_mailto_env)
        self.provider = provider or OpenAlexProvider(
            api_key=resolved_key,
            mailto=resolved_mailto,
            timeout=timeout,
        )

    def search(
        self,
        query: CatalogQuery,
        *,
        limit: int,
    ) -> list[CatalogWork]:
        records = self.provider.search(
            LiteratureSearchRequest(
                query=query.query_text,
                mechanism_bucket=query.axis_id,
                limit=max(1, int(limit)),
            )
        )
        rows: list[CatalogWork] = []
        for record in records:
            metadata = _dict(record.metadata)
            primary = _dict(metadata.get("primary_location"))
            best_oa = _dict(metadata.get("best_oa_location"))
            oa = _dict(metadata.get("open_access"))

            # Deliberately prefer explicit PDF URLs only.  OpenAlex oa_url may
            # be a landing page and must not be mislabeled as a direct PDF.
            pdf_url = _first_text(
                best_oa.get("pdf_url"),
                primary.get("pdf_url"),
            )
            landing_url = _first_text(
                primary.get("landing_page_url"),
                best_oa.get("landing_page_url"),
                oa.get("oa_url"),
            )

            provider_id = None
            for ref in record.provider_references:
                if ref.provider == self.provider_name:
                    provider_id = ref.provider_id
                    break
            if not provider_id:
                provider_id = str(metadata.get("openalex_id") or record.paper_id)
            public_openalex_url = None
            normalized_provider_id = str(provider_id).strip()
            if normalized_provider_id.startswith("https://openalex.org/"):
                public_openalex_url = normalized_provider_id
            elif normalized_provider_id.startswith("W"):
                public_openalex_url = f"https://openalex.org/{normalized_provider_id}"

            work_type = str(metadata.get("work_type") or "").strip()
            citation_count = metadata.get("cited_by_count")
            try:
                citation_count = (
                    int(citation_count)
                    if citation_count is not None
                    else None
                )
            except (TypeError, ValueError):
                citation_count = None

            doi = normalize_doi(record.doi)
            rows.append(
                CatalogWork(
                    work_id=_stable_id(
                        "catalog_work",
                        doi or normalized_provider_id or record.title,
                    ),
                    title=record.title,
                    year=record.year,
                    publication_date=(
                        str(metadata.get("publication_date"))
                        if metadata.get("publication_date")
                        else None
                    ),
                    doi=doi,
                    url=landing_url or public_openalex_url,
                    open_access_url=pdf_url,
                    abstract=record.abstract,
                    authors=[],
                    venue=record.venue,
                    citation_count=citation_count,
                    publication_types=[work_type] if work_type else [],
                    providers=[self.provider_name],
                    provider_ids={self.provider_name: normalized_provider_id},
                    retrieval_query_ids=[query.query_id],
                    retrieval_axis_ids=[query.axis_id],
                )
            )
        return rows
