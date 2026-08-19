from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.literature_catalog import doi_family, normalize_title
from dac_her.literature_catalog_contracts import CatalogWork


FRESH_C_IDENTITY_SEMANTICS_ID = (
    "sers_fresh_c_identity_only_acquisition_v1"
)
FRESH_C_BLIND_ORDER_NAMESPACE = (
    "sers_fresh_c_blind_identity_order_v1"
)
FRESH_C_PROTOCOL_SCHEMA_VERSION = (
    "sers-fresh-c-acquisition-protocol-preregistration-v1"
)
FRESH_C_PROTOCOL_ID_PREFIX = (
    "sers_fresh_c_acquisition_protocol_preregistration_v1"
)
FRESH_C_HISTORICAL_LEDGER_SEMANTICS_ID = (
    "sers_fresh_c_historical_exclusion_ledger_v1"
)

TITLE_FALLBACK_MIN_NORMALIZED_LENGTH = 20

ALLOWED_PRECONSUMPTION_OPERATIONS = frozenset(
    {
        "metadata_transport_without_scientific_inspection",
        "canonical_identity_projection",
        "historical_identity_exclusion",
        "blind_identity_ordering",
        "oa_resolution",
        "pdf_byte_download",
        "pdf_magic_validation",
        "sha256_hashing",
        "provenance_write",
        "content_seal",
    }
)

FORBIDDEN_PRECONSUMPTION_OPERATIONS = frozenset(
    {
        "title_semantic_scoring",
        "abstract_semantic_scoring",
        "citation_scoring",
        "axis_semantic_scoring",
        "hypothesis_specific_selection",
        "pdf_text_extraction",
        "scientific_text_read",
        "llm_extraction",
        "entity_relation_extraction",
        "graph_projection",
        "trend_extraction",
        "hypothesis_evaluation",
    }
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FreshCIdentityRecord(StrictModel):
    """Opaque identity-only record accepted by the blind ordering stage."""

    canonical_id: str
    catalog_work_id: str
    identity_method: Literal[
        "doi_family",
        "normalized_title_sha256",
    ]

    @model_validator(mode="after")
    def _validate_identity(self) -> "FreshCIdentityRecord":
        if self.identity_method == "doi_family":
            if not self.canonical_id.startswith("doi:"):
                raise ValueError(
                    "DOI-family identities must use the doi: prefix."
                )
        else:
            prefix = "title_sha256:"
            if not self.canonical_id.startswith(prefix):
                raise ValueError(
                    "Title-fallback identities must use title_sha256:."
                )
            digest = self.canonical_id[len(prefix) :]
            if (
                len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)
            ):
                raise ValueError("Invalid title fallback SHA256.")
        if not self.catalog_work_id.strip():
            raise ValueError("catalog_work_id must not be empty.")
        return self


class HistoricalLedgerSource(StrictModel):
    source_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_identity_count: int = Field(ge=0)


class HistoricalExclusionLedger(StrictModel):
    schema_version: Literal[
        "sers-fresh-c-historical-exclusion-ledger-v1"
    ] = "sers-fresh-c-historical-exclusion-ledger-v1"
    semantics_id: Literal[
        "sers_fresh_c_historical_exclusion_ledger_v1"
    ] = FRESH_C_HISTORICAL_LEDGER_SEMANTICS_ID
    ledger_id: str
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_ids: list[str]
    sources: list[HistoricalLedgerSource]
    completeness_scope: Literal[
        "all_pre_fresh_c_scientific_exposure"
    ] = "all_pre_fresh_c_scientific_exposure"
    completeness_asserted: Literal[True] = True
    frozen_before_fresh_c_live_discovery: Literal[True] = True
    scientific_fields_retained: Literal[False] = False
    llm_calls: Literal[0] = 0

    @model_validator(mode="after")
    def _ordered_unique(self) -> "HistoricalExclusionLedger":
        if self.canonical_ids != sorted(set(self.canonical_ids)):
            raise ValueError(
                "Historical canonical_ids must be sorted and unique."
            )
        return self


class BlindRankedIdentity(StrictModel):
    canonical_id: str
    catalog_work_id: str
    identity_method: Literal[
        "doi_family",
        "normalized_title_sha256",
    ]
    score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rank: int = Field(ge=1)


class FreshCPreConsumptionSemanticState(StrictModel):
    fresh_reserve_c_consumed: Literal[False] = False
    semantic_read_performed: Literal[False] = False
    pdf_text_extraction_performed: Literal[False] = False
    llm_extraction_performed: Literal[False] = False
    entity_relation_extraction_performed: Literal[False] = False
    graph_projection_performed: Literal[False] = False
    trend_extraction_performed: Literal[False] = False
    hypothesis_evaluation_performed: Literal[False] = False


class IdentityPolicy(StrictModel):
    canonical_identity_priority: Literal[
        "doi_family_then_normalized_title_sha256"
    ]
    doi_prefix: Literal["doi:"]
    title_fallback_prefix: Literal["title_sha256:"]
    title_fallback_min_normalized_length: Literal[20]
    title_cleartext_retained_in_selector_input: Literal[False]
    abstract_retained_in_selector_input: Literal[False]
    citation_count_retained_in_selector_input: Literal[False]
    acquisition_axes_retained_in_selector_input: Literal[False]
    hypothesis_fields_retained_in_selector_input: Literal[False]


class DiscoveryScopePolicy(StrictModel):
    domain_profile_id: Literal["sers_au_ag"]
    scope_semantics: Literal[
        "broad_domain_only_no_hypothesis_or_gap_terms"
    ]
    providers: list[Literal["semantic_scholar", "crossref"]]
    broad_queries: list[str]
    axis_queries_allowed: Literal[False]
    hypothesis_terms_allowed: Literal[False]
    novelty_gap_terms_allowed: Literal[False]
    title_or_abstract_scoring_allowed: Literal[False]
    citation_scoring_allowed: Literal[False]
    results_per_query_defined_in_preregistration: Literal[False]
    search_depth_must_be_frozen_before_live_discovery: Literal[True]


class BlindOrderingPolicy(StrictModel):
    namespace: Literal["sers_fresh_c_blind_identity_order_v1"]
    algorithm: Literal[
        "sort_ascending_sha256_namespace_nul_canonical_id"
    ]
    tie_breaker: Literal["canonical_id"]
    ordering_input_fields: list[Literal["canonical_id"]]
    scientific_fields_used: Literal[False]
    llm_calls: Literal[0]
    caller_supplied_scoring_basis_allowed: Literal[False]
    target_count_defined_in_preregistration: Literal[False]
    target_count_must_be_frozen_before_live_discovery: Literal[True]


class HistoricalExclusionPolicy(StrictModel):
    ledger_required_before_live_discovery: Literal[True]
    completeness_scope: Literal[
        "all_pre_fresh_c_scientific_exposure"
    ]
    exclusion_key: Literal["canonical_id"]
    missing_or_incomplete_ledger_behavior: Literal["fail_closed"]
    ambiguous_identity_behavior: Literal["fail_closed"]
    historical_overlap_behavior: Literal["exclude"]


class AccessFailurePolicy(StrictModel):
    oa_only_automatic_acquisition: Literal[True]
    paywall_bypass_allowed: Literal[False]
    access_availability_used_for_blind_score: Literal[False]
    unresolved_or_download_failed_behavior: Literal[
        "record_inaccessible_then_continue_frozen_blind_order"
    ]
    replacement_basis: Literal["next_identity_in_frozen_blind_order"]
    manual_replacement_allowed: Literal[False]
    scientific_content_based_replacement_allowed: Literal[False]
    failed_candidate_rank_is_reassigned: Literal[False]
    retry_policy_source: Literal[
        "configs/acquisition/source_access_default_v1.yaml"
    ]


class ReusePolicy(StrictModel):
    reused_components: list[str]
    scientific_candidate_selector_reused: Literal[False]
    forbidden_selector_path: Literal[
        "dac_her/corpus_acquisition/candidate_selection.py"
    ]
    paywall_bypass_allowed: Literal[False]
    positive_scientific_evidence_promotion_allowed: Literal[False]


class PreConsumptionPolicy(StrictModel):
    allowed_operations: list[str]
    forbidden_operations: list[str]
    scientific_content_interpretation_allowed: Literal[False]
    semantic_pdf_read_allowed: Literal[False]
    hypothesis_specific_selection_allowed: Literal[False]
    consumption_must_precede_first_scientific_transformation: Literal[True]


class PreregistrationSafety(StrictModel):
    protocol_preregistration_only: Literal[True]
    fresh_c_stage_activated: Literal[False]
    activation_preconditions_satisfied_at_preregistration: Literal[False]
    live_discovery_started: Literal[False]
    live_selection_started: Literal[False]
    live_acquisition_started: Literal[False]
    content_sealed: Literal[False]
    fresh_reserve_c_consumed: Literal[False]
    semantic_read_performed: Literal[False]
    network_calls: Literal[0]
    llm_calls: Literal[0]
    automatic_next_stage_authorized: Literal[False]
    stop_after_preregistration_freeze: Literal[True]


class FreshCAcquisitionProtocol(StrictModel):
    schema_version: Literal[
        "sers-fresh-c-acquisition-protocol-preregistration-v1"
    ]
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantics_id: Literal["sers_fresh_c_identity_only_acquisition_v1"]
    stage: Literal["C0.1A"]
    status: Literal["PREREGISTRATION_ONLY"]
    activation_preconditions_required: list[str]
    discovery_scope_policy: DiscoveryScopePolicy
    identity_policy: IdentityPolicy
    blind_ordering_policy: BlindOrderingPolicy
    historical_exclusion_policy: HistoricalExclusionPolicy
    access_failure_policy: AccessFailurePolicy
    reuse_policy: ReusePolicy
    preconsumption_policy: PreConsumptionPolicy
    safety: PreregistrationSafety

    @model_validator(mode="after")
    def _exact_semantics(self) -> "FreshCAcquisitionProtocol":
        expected_preconditions = [
            "I0_FROZEN",
            "C0_0_EXISTING_RESERVE_PROVENANCE_AUDIT_PASS",
        ]
        if self.activation_preconditions_required != expected_preconditions:
            raise ValueError("Fresh-C activation preconditions drifted.")
        expected_providers = ["semantic_scholar", "crossref"]
        expected_queries = [
            "surface enhanced Raman spectroscopy gold silver",
            "SERS gold silver",
            "surface enhanced Raman spectroscopy Au Ag",
            "SERS Au Ag",
        ]
        if self.discovery_scope_policy.providers != expected_providers:
            raise ValueError("Fresh-C discovery provider set drifted.")
        if self.discovery_scope_policy.broad_queries != expected_queries:
            raise ValueError("Fresh-C broad discovery queries drifted.")
        if self.blind_ordering_policy.ordering_input_fields != [
            "canonical_id"
        ]:
            raise ValueError(
                "Fresh-C blind ordering may consume canonical_id only."
            )
        if set(self.preconsumption_policy.allowed_operations) != set(
            ALLOWED_PRECONSUMPTION_OPERATIONS
        ):
            raise ValueError(
                "Allowed pre-consumption operation set drifted."
            )
        if set(self.preconsumption_policy.forbidden_operations) != set(
            FORBIDDEN_PRECONSUMPTION_OPERATIONS
        ):
            raise ValueError(
                "Forbidden pre-consumption operation set drifted."
            )
        forbidden_selector = self.reuse_policy.forbidden_selector_path
        if forbidden_selector in self.reuse_policy.reused_components:
            raise ValueError(
                "Scientific candidate selector may not be reused."
            )
        return self


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha(
    payload: Mapping[str, Any],
    sha_field: str,
) -> str:
    value = dict(payload)
    value.pop(sha_field, None)
    return sha256_json(value)


def _protocol_identity_sha(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("protocol_id", None)
    value.pop("protocol_sha256", None)
    return sha256_json(value)


def expected_protocol_id(payload: Mapping[str, Any]) -> str:
    digest = _protocol_identity_sha(payload)
    return f"{FRESH_C_PROTOCOL_ID_PREFIX}:{digest[:20]}"


def canonical_identity_from_fields(
    *,
    doi: str | None,
    title: str | None,
) -> tuple[str, Literal["doi_family", "normalized_title_sha256"]]:
    """Project bibliographic metadata to a selector-safe canonical identity.

    DOI family is preferred.  If DOI is unavailable, the normalized title is
    used only to compute a SHA256 fallback and is not returned.
    """

    family = doi_family(doi)
    if family:
        return f"doi:{family}", "doi_family"

    normalized = normalize_title(str(title or ""))
    if len(normalized) < TITLE_FALLBACK_MIN_NORMALIZED_LENGTH:
        raise ValueError(
            "Cannot form stable Fresh-C identity: DOI missing and "
            "normalized title is too short."
        )
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"title_sha256:{digest}", "normalized_title_sha256"


def project_catalog_identity(work: CatalogWork) -> FreshCIdentityRecord:
    """Discard scientific metadata before the blind ordering boundary."""

    canonical_id, method = canonical_identity_from_fields(
        doi=work.doi,
        title=work.title,
    )
    return FreshCIdentityRecord(
        canonical_id=canonical_id,
        catalog_work_id=work.work_id,
        identity_method=method,
    )


def _blind_score(
    canonical_id: str,
    *,
    namespace: str = FRESH_C_BLIND_ORDER_NAMESPACE,
) -> str:
    raw = (namespace + "\0" + canonical_id).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_historical_exclusion_ledger(
    *,
    canonical_ids: Iterable[str],
    sources: Sequence[HistoricalLedgerSource],
) -> HistoricalExclusionLedger:
    ids = sorted(str(value) for value in canonical_ids)
    if not ids:
        raise ValueError(
            "Historical exclusion ledger must not be empty."
        )
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Historical exclusion ledger contains duplicate identities."
        )
    source_rows = [
        row.model_dump(mode="json")
        if hasattr(row, "model_dump")
        else dict(row)
        for row in sources
    ]
    if not source_rows:
        raise ValueError(
            "Historical exclusion ledger requires provenance sources."
        )

    body: dict[str, Any] = {
        "schema_version": (
            "sers-fresh-c-historical-exclusion-ledger-v1"
        ),
        "semantics_id": FRESH_C_HISTORICAL_LEDGER_SEMANTICS_ID,
        "canonical_ids": ids,
        "sources": source_rows,
        "completeness_scope": (
            "all_pre_fresh_c_scientific_exposure"
        ),
        "completeness_asserted": True,
        "frozen_before_fresh_c_live_discovery": True,
        "scientific_fields_retained": False,
        "llm_calls": 0,
    }
    identity_sha = sha256_json(body)
    body["ledger_id"] = (
        "sers_fresh_c_historical_exclusion_ledger_v1:"
        + identity_sha[:20]
    )
    body["ledger_sha256"] = _payload_sha(
        body,
        "ledger_sha256",
    )
    return HistoricalExclusionLedger.model_validate(body)


def validate_historical_exclusion_ledger(
    ledger: HistoricalExclusionLedger | Mapping[str, Any],
) -> HistoricalExclusionLedger:
    row = (
        ledger
        if isinstance(ledger, HistoricalExclusionLedger)
        else HistoricalExclusionLedger.model_validate(ledger)
    )
    payload = row.model_dump(mode="json")
    expected_sha = _payload_sha(payload, "ledger_sha256")
    if row.ledger_sha256 != expected_sha:
        raise ValueError("Historical exclusion ledger SHA drifted.")
    return row


def rank_fresh_identities(
    *,
    candidates: Sequence[FreshCIdentityRecord],
    historical_ledger: HistoricalExclusionLedger | Mapping[str, Any],
    namespace: str = FRESH_C_BLIND_ORDER_NAMESPACE,
) -> list[BlindRankedIdentity]:
    """Blindly order non-historical candidates.

    There is intentionally no target-count/limit parameter here.  C0.1A does
    not choose a holdout size; a later activation protocol must freeze that
    value before live discovery.
    """

    if namespace != FRESH_C_BLIND_ORDER_NAMESPACE:
        raise ValueError(
            "Caller-supplied blind-order namespace is not authorized."
        )
    ledger = validate_historical_exclusion_ledger(
        historical_ledger
    )
    historical = set(ledger.canonical_ids)

    seen: set[str] = set()
    fresh: list[FreshCIdentityRecord] = []
    for row in candidates:
        if row.canonical_id in seen:
            raise ValueError(
                "Fresh-C candidate canonical identity duplicated: "
                f"{row.canonical_id}"
            )
        seen.add(row.canonical_id)
        if row.canonical_id in historical:
            continue
        fresh.append(row)

    ranked = sorted(
        (
            (
                _blind_score(
                    row.canonical_id,
                    namespace=namespace,
                ),
                row.canonical_id,
                row,
            )
            for row in fresh
        ),
        key=lambda item: (item[0], item[1]),
    )
    return [
        BlindRankedIdentity(
            canonical_id=row.canonical_id,
            catalog_work_id=row.catalog_work_id,
            identity_method=row.identity_method,
            score_sha256=score,
            rank=index,
        )
        for index, (score, _canonical_id, row) in enumerate(
            ranked,
            start=1,
        )
    ]


def assert_preconsumption_operation_allowed(operation: str) -> None:
    if operation in FORBIDDEN_PRECONSUMPTION_OPERATIONS:
        raise PermissionError(
            "Fresh-C semantic consumption boundary blocks "
            f"pre-consumption operation: {operation}"
        )
    if operation not in ALLOWED_PRECONSUMPTION_OPERATIONS:
        raise PermissionError(
            "Fresh-C pre-consumption operation is not explicitly "
            f"allowlisted: {operation}"
        )


def assert_semantically_unread(
    state: FreshCPreConsumptionSemanticState | Mapping[str, Any],
) -> FreshCPreConsumptionSemanticState:
    return (
        state
        if isinstance(state, FreshCPreConsumptionSemanticState)
        else FreshCPreConsumptionSemanticState.model_validate(state)
    )


def load_and_validate_protocol(
    path: Path,
) -> FreshCAcquisitionProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Fresh-C protocol must be a JSON object.")

    protocol = FreshCAcquisitionProtocol.model_validate(payload)
    expected_id = expected_protocol_id(payload)
    if protocol.protocol_id != expected_id:
        raise ValueError(
            "Fresh-C protocol ID mismatch: "
            f"{protocol.protocol_id!r} != {expected_id!r}"
        )
    expected_sha = _payload_sha(payload, "protocol_sha256")
    if protocol.protocol_sha256 != expected_sha:
        raise ValueError(
            "Fresh-C protocol SHA256 mismatch: "
            f"{protocol.protocol_sha256!r} != {expected_sha!r}"
        )
    return protocol
