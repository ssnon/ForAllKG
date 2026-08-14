from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CatalogQuery(StrictModel):
    query_id: str
    profile_id: str
    axis_id: str
    query_text: str


class CatalogWork(StrictModel):
    work_id: str
    title: str
    year: int | None = None
    publication_date: str | None = None
    doi: str | None = None
    url: str | None = None
    open_access_url: str | None = None
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    citation_count: int | None = None
    publication_types: list[str] = Field(default_factory=list)
    providers: list[str] = Field(default_factory=list)
    provider_ids: dict[str, str] = Field(default_factory=dict)
    retrieval_query_ids: list[str] = Field(default_factory=list)
    retrieval_axis_ids: list[str] = Field(default_factory=list)


class CatalogQueryExecution(StrictModel):
    query_id: str
    axis_id: str
    provider: str
    success: bool
    result_count: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


class LiteratureCatalogPacket(StrictModel):
    schema_version: Literal["literature-catalog-packet-v1"] = (
        "literature-catalog-packet-v1"
    )
    catalog_id: str
    catalog_sha256: str
    acquisition_profile_id: str
    searched_at_utc: str
    providers_requested: list[str] = Field(default_factory=list)
    queries: list[CatalogQuery] = Field(default_factory=list)
    works: list[CatalogWork] = Field(default_factory=list)
    executions: list[CatalogQueryExecution] = Field(default_factory=list)
    raw_work_count: int = 0
    canonical_work_count: int = 0
    deduplicated_work_count: int = 0
    supplementary_records_collapsed: int = 0
    epistemic_usage: Literal[
        "candidate_source_only_not_positive_premise"
    ] = "candidate_source_only_not_positive_premise"
