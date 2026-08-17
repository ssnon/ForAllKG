from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.fresh_c_acquisition import (
    FRESH_C_BLIND_ORDER_NAMESPACE,
    FreshCIdentityRecord,
    canonical_json,
    project_catalog_identity,
    rank_fresh_identities,
    sha256_file,
    sha256_json,
    validate_historical_exclusion_ledger,
)
from dac_her.literature_catalog_contracts import (
    CatalogQuery,
    CatalogQueryExecution,
    CatalogWork,
    LiteratureCatalogPacket,
)

LIVE_DISCOVERY_SEMANTICS_ID = "sers_fresh_c_live_metadata_discovery_v1"
LIVE_DISCOVERY_PROTOCOL_PREFIX = "sers_fresh_c_live_discovery_protocol_v1"
NEUTRAL_AXIS_ID = "fresh_c_broad_domain_identity_only"
PROFILE_ID = "sers_fresh_c_broad_domain_v1"

EXPECTED_C01B_LOCK_ID = (
    "sers_fresh_c_activation_readiness_lock_v1:73f30b6430c9e8bc0aec"
)
EXPECTED_C01B_LOCK_SHA256 = (
    "0157539364af62597032493ec15405ec6f67336c78c497c3623d1f5db79f8eb2"
)
EXPECTED_C01B_FREEZE_COMMIT = (
    "e9e495b5b103f3e6c6945ece00141758c5035f83"
)
EXPECTED_HISTORICAL_LEDGER_ID = (
    "sers_fresh_c_historical_exclusion_ledger_v1:c131d242e751a66fb411"
)
EXPECTED_HISTORICAL_LEDGER_SHA256 = (
    "448140e98c847aaade5d321d84db74643804c858be4b253092e1cb63134bfdd2"
)
EXPECTED_HISTORICAL_IDENTITY_COUNT = 560

EXPECTED_PROVIDERS = ["semantic_scholar", "crossref"]
EXPECTED_BROAD_QUERIES = [
    "surface enhanced Raman spectroscopy gold silver",
    "SERS gold silver",
    "surface enhanced Raman spectroscopy Au Ag",
    "SERS Au Ag",
]
RESULTS_PER_QUERY = 100
EXPECTED_PROVIDER_QUERY_EXECUTIONS = 8
MAX_RAW_METADATA_ROWS = 800
TARGET_ACQUIRED_PAPERS = 25

DEFAULT_PROTOCOL_PATH = Path(
    "dac_her/sers_fresh_c_live_discovery_protocol_v1.json"
)
DEFAULT_C01B_DIR = Path(
    "evaluation/sers_fresh_c/c0_1b_activation_readiness_v1"
)
DEFAULT_DISCOVERY_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c0_1c_live_discovery_protocol_freeze_v1"
)
DEFAULT_DISCOVERY_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c0_1c_live_discovery_run_v1"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LiveDiscoveryProtocol(StrictModel):
    schema_version: Literal["sers-fresh-c-live-discovery-protocol-v1"]
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantics_id: Literal["sers_fresh_c_live_metadata_discovery_v1"]
    stage: Literal["C0.1C"]

    c0_1b_lock_id: Literal[
        "sers_fresh_c_activation_readiness_lock_v1:73f30b6430c9e8bc0aec"
    ]
    c0_1b_lock_sha256: Literal[
        "0157539364af62597032493ec15405ec6f67336c78c497c3623d1f5db79f8eb2"
    ]
    c0_1b_freeze_commit: Literal[
        "e9e495b5b103f3e6c6945ece00141758c5035f83"
    ]
    historical_ledger_id: Literal[
        "sers_fresh_c_historical_exclusion_ledger_v1:c131d242e751a66fb411"
    ]
    historical_ledger_sha256: Literal[
        "448140e98c847aaade5d321d84db74643804c858be4b253092e1cb63134bfdd2"
    ]
    historical_identity_count: Literal[560]

    providers: list[str]
    broad_queries: list[str]
    neutral_axis_id: Literal["fresh_c_broad_domain_identity_only"]
    results_per_query: Literal[100]
    expected_provider_query_executions: Literal[8]
    max_raw_metadata_rows: Literal[800]
    target_acquired_papers: Literal[25]

    all_provider_query_executions_must_succeed: Literal[True]
    raw_catalog_packet_persisted: Literal[False]
    title_persisted_in_selection_artifacts: Literal[False]
    abstract_persisted_in_selection_artifacts: Literal[False]
    citation_count_persisted_in_selection_artifacts: Literal[False]
    access_locator_metadata_only: Literal[True]
    human_inspection_of_identity_or_locator_artifacts_before_c1_allowed: Literal[False]
    historical_exclusion_before_blind_order: Literal[True]
    blind_order_namespace: Literal["sers_fresh_c_blind_identity_order_v1"]
    full_fresh_identity_queue_frozen: Literal[True]
    queue_truncated_to_target: Literal[False]
    scientific_fields_used_for_ordering: Literal[False]
    llm_calls: Literal[0]

    explicit_live_discovery_confirmation_required: Literal[True]
    discovery_started_marker_before_first_network_call: Literal[True]
    same_epoch_rerun_after_start_allowed: Literal[False]
    failure_authorizes_query_or_selection_tuning: Literal[False]
    failed_epoch_consumes_fresh_reserve_c: Literal[False]
    fresh_reserve_c_consumption_occurs_in_c0_1c: Literal[False]
    automatic_c0_1d_transition_allowed: Literal[False]
    stop_after_success: Literal[True]

    @model_validator(mode="after")
    def _exact_contract(self) -> "LiveDiscoveryProtocol":
        if self.providers != EXPECTED_PROVIDERS:
            raise ValueError("C0.1C provider set drifted.")
        if self.broad_queries != EXPECTED_BROAD_QUERIES:
            raise ValueError("C0.1C broad queries drifted.")
        return self


class AccessLocatorRecord(StrictModel):
    canonical_id: str
    identity_method: Literal["doi_family", "normalized_title_sha256"]
    catalog_work_ids: list[str]
    doi_candidates: list[str]
    urls: list[str]
    open_access_urls: list[str]
    providers: list[str]
    provider_ids: dict[str, list[str]]

    @model_validator(mode="after")
    def _sorted_unique(self) -> "AccessLocatorRecord":
        for field in (
            "catalog_work_ids",
            "doi_candidates",
            "urls",
            "open_access_urls",
            "providers",
        ):
            values = getattr(self, field)
            if values != sorted(set(values)):
                raise ValueError(f"Access locator {field} must be sorted/unique.")
        for key, values in self.provider_ids.items():
            if values != sorted(set(values)):
                raise ValueError(
                    f"Access locator provider_ids[{key!r}] must be sorted/unique."
                )
        return self


class QueryExecutionSummary(StrictModel):
    query_id: str
    provider: str
    success: bool
    result_count: int = Field(ge=0)
    error_type: str | None = None


class BlindQueueRecord(StrictModel):
    canonical_id: str
    identity_method: Literal["doi_family", "normalized_title_sha256"]
    score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rank: int = Field(ge=1)


class LiveDiscoveryManifest(StrictModel):
    schema_version: Literal["sers-fresh-c-live-discovery-run-v1"]
    run_id: str
    run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    discovery_freeze_id: str
    discovery_freeze_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    c0_1b_lock_id: str
    c0_1b_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    historical_ledger_id: str
    historical_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    historical_identity_count: int

    searched_at_utc: str
    providers: list[str]
    broad_query_count: int
    results_per_query: int
    expected_provider_query_executions: int
    observed_provider_query_executions: int
    successful_provider_query_executions: int
    query_executions: list[QueryExecutionSummary]

    raw_work_count: int = Field(ge=0)
    catalog_canonical_work_count: int = Field(ge=0)
    projected_unique_identity_count: int = Field(ge=0)
    identity_duplicate_merge_count: int = Field(ge=0)
    ambiguous_identity_excluded_count: int = Field(ge=0)
    historical_excluded_identity_count: int = Field(ge=0)
    fresh_identity_queue_count: int = Field(ge=0)
    target_acquired_papers: int

    blind_queue_path: str
    blind_queue_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    access_locator_path: str
    access_locator_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    raw_catalog_packet_persisted: Literal[False]
    scientific_metadata_fields_persisted: Literal[False]
    title_persisted: Literal[False]
    abstract_persisted: Literal[False]
    citation_count_persisted: Literal[False]
    scientific_fields_used_for_ordering: Literal[False]
    llm_calls: Literal[0]
    network_used: Literal[True]
    exact_http_request_count_known: Literal[False]
    fresh_reserve_c_consumed: Literal[False]
    semantic_read_performed: Literal[False]
    automatic_c0_1d_transition_authorized: Literal[False]
    stop: Literal[True]


class DiscoveryProjection:
    def __init__(
        self,
        *,
        identity_records: list[FreshCIdentityRecord],
        locators: list[AccessLocatorRecord],
        duplicate_merge_count: int,
        ambiguous_identity_excluded_count: int,
    ) -> None:
        self.identity_records = identity_records
        self.locators = locators
        self.duplicate_merge_count = duplicate_merge_count
        self.ambiguous_identity_excluded_count = ambiguous_identity_excluded_count


def _payload_sha(payload: Mapping[str, Any], sha_field: str) -> str:
    value = dict(payload)
    value.pop(sha_field, None)
    return sha256_json(value)


def _protocol_identity_sha(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("protocol_id", None)
    value.pop("protocol_sha256", None)
    return sha256_json(value)


def expected_protocol_id(payload: Mapping[str, Any]) -> str:
    return (
        LIVE_DISCOVERY_PROTOCOL_PREFIX
        + ":"
        + _protocol_identity_sha(payload)[:20]
    )


def load_and_validate_protocol(path: Path) -> LiveDiscoveryProtocol:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("C0.1C protocol must be a JSON object.")
    protocol = LiveDiscoveryProtocol.model_validate(raw)
    if protocol.protocol_id != expected_protocol_id(raw):
        raise ValueError("C0.1C protocol ID mismatch.")
    if protocol.protocol_sha256 != _payload_sha(raw, "protocol_sha256"):
        raise ValueError("C0.1C protocol SHA mismatch.")
    return protocol


def stable_query_id(index: int, query_text: str) -> str:
    raw = (
        LIVE_DISCOVERY_SEMANTICS_ID
        + "\0"
        + str(index)
        + "\0"
        + query_text
    ).encode("utf-8")
    return "fresh_c_query:" + hashlib.sha256(raw).hexdigest()[:20]


def make_catalog_queries(protocol: LiveDiscoveryProtocol) -> list[CatalogQuery]:
    return [
        CatalogQuery(
            query_id=stable_query_id(index, text),
            profile_id=PROFILE_ID,
            axis_id=NEUTRAL_AXIS_ID,
            query_text=text,
        )
        for index, text in enumerate(protocol.broad_queries, start=1)
    ]


def _clean_optional(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _merge_provider_ids(
    target: dict[str, set[str]],
    provider_ids: Mapping[str, str],
) -> None:
    for provider, value in provider_ids.items():
        clean = _clean_optional(value)
        if clean:
            target.setdefault(str(provider), set()).add(clean)


def project_packet_to_identity_only(packet: LiteratureCatalogPacket) -> DiscoveryProjection:
    grouped: dict[str, dict[str, Any]] = {}
    duplicate_merge_count = 0
    ambiguous_identity_excluded_count = 0

    for work in packet.works:
        try:
            identity = project_catalog_identity(work)
        except ValueError:
            # Candidate-level fail-closed: an ambiguous bibliographic identity
            # may not enter the Fresh-C queue, but does not authorize semantic
            # inspection, manual substitution, or query adaptation.
            ambiguous_identity_excluded_count += 1
            continue
        row = grouped.get(identity.canonical_id)
        if row is None:
            row = {
                "canonical_id": identity.canonical_id,
                "identity_method": identity.identity_method,
                "catalog_work_ids": set(),
                "doi_candidates": set(),
                "urls": set(),
                "open_access_urls": set(),
                "providers": set(),
                "provider_ids": {},
            }
            grouped[identity.canonical_id] = row
        else:
            duplicate_merge_count += 1
            if row["identity_method"] != identity.identity_method:
                raise ValueError(
                    "Canonical identity collision across identity methods."
                )

        row["catalog_work_ids"].add(identity.catalog_work_id)
        doi = _clean_optional(work.doi)
        if doi:
            row["doi_candidates"].add(doi.casefold())
        url = _clean_optional(work.url)
        if url:
            row["urls"].add(url)
        oa = _clean_optional(work.open_access_url)
        if oa:
            row["open_access_urls"].add(oa)
        row["providers"].update(str(value) for value in work.providers)
        _merge_provider_ids(row["provider_ids"], work.provider_ids)

    identity_records: list[FreshCIdentityRecord] = []
    locators: list[AccessLocatorRecord] = []
    for canonical_id in sorted(grouped):
        row = grouped[canonical_id]
        catalog_work_ids = sorted(row["catalog_work_ids"])
        if not catalog_work_ids:
            raise ValueError("Projected identity lacks catalog work ID.")
        identity_records.append(
            FreshCIdentityRecord(
                canonical_id=canonical_id,
                catalog_work_id=catalog_work_ids[0],
                identity_method=row["identity_method"],
            )
        )
        locators.append(
            AccessLocatorRecord(
                canonical_id=canonical_id,
                identity_method=row["identity_method"],
                catalog_work_ids=catalog_work_ids,
                doi_candidates=sorted(row["doi_candidates"]),
                urls=sorted(row["urls"]),
                open_access_urls=sorted(row["open_access_urls"]),
                providers=sorted(row["providers"]),
                provider_ids={
                    key: sorted(values)
                    for key, values in sorted(row["provider_ids"].items())
                },
            )
        )
    return DiscoveryProjection(
        identity_records=identity_records,
        locators=locators,
        duplicate_merge_count=duplicate_merge_count,
        ambiguous_identity_excluded_count=ambiguous_identity_excluded_count,
    )


def build_fresh_queue(
    *,
    projection: DiscoveryProjection,
    historical_ledger: Mapping[str, Any],
) -> tuple[list[BlindQueueRecord], list[AccessLocatorRecord], int]:
    ledger = validate_historical_exclusion_ledger(historical_ledger)
    historical = set(ledger.canonical_ids)
    historical_excluded = sum(
        1
        for row in projection.identity_records
        if row.canonical_id in historical
    )

    ranked = rank_fresh_identities(
        candidates=projection.identity_records,
        historical_ledger=ledger,
        namespace=FRESH_C_BLIND_ORDER_NAMESPACE,
    )
    queue = [
        BlindQueueRecord(
            canonical_id=row.canonical_id,
            identity_method=row.identity_method,
            score_sha256=row.score_sha256,
            rank=row.rank,
        )
        for row in ranked
    ]
    fresh_ids = {row.canonical_id for row in queue}
    fresh_locators = [
        row
        for row in projection.locators
        if row.canonical_id in fresh_ids
    ]
    fresh_locators.sort(key=lambda row: row.canonical_id)
    return queue, fresh_locators, historical_excluded


def summarize_query_executions(
    executions: Sequence[CatalogQueryExecution],
) -> list[QueryExecutionSummary]:
    rows: list[QueryExecutionSummary] = []
    for row in executions:
        error_type = None
        if not row.success and row.error:
            error_type = str(row.error).split(":", 1)[0].strip() or "Error"
        rows.append(
            QueryExecutionSummary(
                query_id=row.query_id,
                provider=row.provider,
                success=bool(row.success),
                result_count=int(row.result_count),
                error_type=error_type,
            )
        )
    return rows


def assert_complete_execution(
    *,
    protocol: LiveDiscoveryProtocol,
    executions: Sequence[CatalogQueryExecution],
) -> None:
    if len(executions) != protocol.expected_provider_query_executions:
        raise RuntimeError(
            "C0.1C provider-query execution count incomplete: "
            f"{len(executions)} != {protocol.expected_provider_query_executions}"
        )
    queries = make_catalog_queries(protocol)
    expected_pairs = {
        (query.query_id, provider)
        for query in queries
        for provider in protocol.providers
    }
    observed_pairs = {(row.query_id, row.provider) for row in executions}
    if observed_pairs != expected_pairs or len(observed_pairs) != len(executions):
        raise RuntimeError(
            "C0.1C provider-query execution identity set drifted."
        )
    if any(int(row.result_count) > protocol.results_per_query for row in executions):
        raise RuntimeError(
            "C0.1C provider-query result count exceeded frozen per-query budget."
        )
    failures = [row for row in executions if not row.success]
    if failures:
        labels = [f"{row.provider}:{row.query_id}" for row in failures]
        raise RuntimeError(
            "C0.1C requires every frozen provider-query execution to succeed; "
            "failed=" + ",".join(labels)
        )


def make_blind_queue_payload(
    *,
    protocol: LiveDiscoveryProtocol,
    queue: Sequence[BlindQueueRecord],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "sers-fresh-c-blind-selection-queue-v1",
        "semantics_id": LIVE_DISCOVERY_SEMANTICS_ID,
        "protocol_id": protocol.protocol_id,
        "blind_order_namespace": FRESH_C_BLIND_ORDER_NAMESPACE,
        "queue_is_full_fresh_identity_order": True,
        "queue_truncated_to_target": False,
        "target_acquired_papers": TARGET_ACQUIRED_PAPERS,
        "queue_count": len(queue),
        "records": [row.model_dump(mode="json") for row in queue],
        "scientific_fields_used_for_ordering": False,
        "title_persisted": False,
        "abstract_persisted": False,
        "citation_count_persisted": False,
        "llm_calls": 0,
        "fresh_reserve_c_consumed": False,
    }
    body["queue_sha256"] = _payload_sha(body, "queue_sha256")
    return body


def make_access_locator_payload(
    *,
    protocol: LiveDiscoveryProtocol,
    locators: Sequence[AccessLocatorRecord],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "sers-fresh-c-access-locator-manifest-v1",
        "protocol_id": protocol.protocol_id,
        "locator_count": len(locators),
        "records": [row.model_dump(mode="json") for row in locators],
        "allowed_metadata": [
            "canonical_id",
            "identity_method",
            "catalog_work_ids",
            "doi_candidates",
            "urls",
            "open_access_urls",
            "providers",
            "provider_ids",
        ],
        "title_persisted": False,
        "abstract_persisted": False,
        "citation_count_persisted": False,
        "scientific_interpretation_performed": False,
        "llm_calls": 0,
        "fresh_reserve_c_consumed": False,
    }
    body["locator_sha256"] = _payload_sha(body, "locator_sha256")
    return body
