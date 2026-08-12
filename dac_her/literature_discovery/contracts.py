from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SourceDepth = Literal["metadata", "abstract", "fulltext"]
_SOURCE_DEPTH_RANK: dict[SourceDepth, int] = {
    "metadata": 0,
    "abstract": 1,
    "fulltext": 2,
}


def normalize_doi(value: str | None) -> str | None:
    """Return a canonical DOI string without URL/prefix decoration."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.I)
    text = re.sub(r"^doi:\s*", "", text, flags=re.I)
    text = text.strip().lower()
    return text or None


def literature_paper_id(
    *,
    doi: str | None,
    provider: str,
    provider_id: str,
) -> str:
    """Build a stable internal ID, preferring DOI identity when available."""
    normalized_doi = normalize_doi(doi)
    if normalized_doi:
        identity = f"doi:{normalized_doi}"
    else:
        provider_name = str(provider).strip().lower()
        provider_key = str(provider_id).strip()
        if not provider_name or not provider_key:
            raise ValueError("provider and provider_id are required when DOI is absent")
        identity = f"provider:{provider_name}:{provider_key}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"literature:{digest}"


@dataclass(frozen=True, order=True)
class ProviderReference:
    provider: str
    provider_id: str

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must not be empty")
        if not self.provider_id.strip():
            raise ValueError("provider_id must not be empty")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProviderReference":
        return cls(
            provider=str(value["provider"]),
            provider_id=str(value["provider_id"]),
        )


@dataclass(frozen=True)
class LiteratureRecord:
    paper_id: str
    title: str
    abstract: str | None
    doi: str | None
    year: int | None
    venue: str | None
    provider_references: tuple[ProviderReference, ...]
    discovery_queries: tuple[str, ...] = ()
    mechanism_buckets: tuple[str, ...] = ()
    source_depth: SourceDepth = "metadata"
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            raise ValueError("paper_id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.provider_references:
            raise ValueError("at least one provider reference is required")
        if self.source_depth not in _SOURCE_DEPTH_RANK:
            raise ValueError(f"unsupported source_depth: {self.source_depth!r}")
        if self.year is not None and self.year < 0:
            raise ValueError("year must be non-negative")

    @classmethod
    def from_provider_result(
        cls,
        *,
        provider: str,
        provider_id: str,
        title: str,
        abstract: str | None = None,
        doi: str | None = None,
        year: int | None = None,
        venue: str | None = None,
        discovery_query: str | None = None,
        mechanism_bucket: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "LiteratureRecord":
        normalized_doi = normalize_doi(doi)
        source_depth: SourceDepth = "abstract" if (abstract or "").strip() else "metadata"
        return cls(
            paper_id=literature_paper_id(
                doi=normalized_doi,
                provider=provider,
                provider_id=provider_id,
            ),
            title=title.strip(),
            abstract=(abstract.strip() if abstract and abstract.strip() else None),
            doi=normalized_doi,
            year=year,
            venue=(venue.strip() if venue and venue.strip() else None),
            provider_references=(
                ProviderReference(provider=provider.strip(), provider_id=provider_id.strip()),
            ),
            discovery_queries=(
                (discovery_query.strip(),)
                if discovery_query and discovery_query.strip()
                else ()
            ),
            mechanism_buckets=(
                (mechanism_bucket.strip(),)
                if mechanism_bucket and mechanism_bucket.strip()
                else ()
            ),
            source_depth=source_depth,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "doi": self.doi,
            "year": self.year,
            "venue": self.venue,
            "provider_references": [item.to_dict() for item in self.provider_references],
            "discovery_queries": list(self.discovery_queries),
            "mechanism_buckets": list(self.mechanism_buckets),
            "source_depth": self.source_depth,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LiteratureRecord":
        return cls(
            paper_id=str(value["paper_id"]),
            title=str(value["title"]),
            abstract=(str(value["abstract"]) if value.get("abstract") else None),
            doi=normalize_doi(value.get("doi")),
            year=(int(value["year"]) if value.get("year") is not None else None),
            venue=(str(value["venue"]) if value.get("venue") else None),
            provider_references=tuple(
                ProviderReference.from_dict(item)
                for item in value.get("provider_references", [])
            ),
            discovery_queries=tuple(str(item) for item in value.get("discovery_queries", [])),
            mechanism_buckets=tuple(str(item) for item in value.get("mechanism_buckets", [])),
            source_depth=str(value.get("source_depth", "metadata")),  # type: ignore[arg-type]
            metadata=dict(value.get("metadata") or {}),
        )


def merge_literature_records(
    current: LiteratureRecord,
    incoming: LiteratureRecord,
) -> LiteratureRecord:
    """Merge repeated discoveries of the same internal paper identity."""
    if current.paper_id != incoming.paper_id:
        raise ValueError("cannot merge records with different paper_id values")

    richer, other = (
        (incoming, current)
        if _SOURCE_DEPTH_RANK[incoming.source_depth] > _SOURCE_DEPTH_RANK[current.source_depth]
        else (current, incoming)
    )
    provider_refs = tuple(sorted(set(current.provider_references) | set(incoming.provider_references)))
    queries = tuple(sorted(set(current.discovery_queries) | set(incoming.discovery_queries)))
    buckets = tuple(sorted(set(current.mechanism_buckets) | set(incoming.mechanism_buckets)))
    metadata = dict(current.metadata)
    metadata.update(incoming.metadata)

    return LiteratureRecord(
        paper_id=current.paper_id,
        title=(richer.title or other.title),
        abstract=(richer.abstract or other.abstract),
        doi=(current.doi or incoming.doi),
        year=(richer.year if richer.year is not None else other.year),
        venue=(richer.venue or other.venue),
        provider_references=provider_refs,
        discovery_queries=queries,
        mechanism_buckets=buckets,
        source_depth=richer.source_depth,
        metadata=metadata,
    )
