from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.corpus_acquisition.access_contracts import SourceAcquisitionPolicy
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_json

C01D_SEMANTICS_ID = "sers_fresh_c_blind_oa_content_acquisition_v1"
C01D_PROTOCOL_PREFIX = "sers_fresh_c_blind_oa_content_acquisition_protocol_v1"

EXPECTED_V24_PROTOCOL_ID = "sers_fresh_c_openalex_crossref_substitution_protocol_v2_4:5e09cf20c06210a742a5"
EXPECTED_V24_PROTOCOL_SHA256 = "99389342726c2bc2deab5fde0b2aacf3b52ea9b4a6f6981578f2b73008220fcb"
EXPECTED_V24_FREEZE_ID = "sers_fresh_c_openalex_crossref_substitution_freeze_v1:1ad72017775ef79662dc"
EXPECTED_V24_FREEZE_MANIFEST_SHA256 = "7ad250b56946e4954b55e7e0b7de01e22a9675890a581697a4fcd3f7bd9b14be"
EXPECTED_V24_RUN_ID = "sers_fresh_c_openalex_crossref_run_v2_4:220ee42f03afe8aaa1e1"
EXPECTED_V24_RUN_SHA256 = "813e0869c8c4b82971669de65bbcd7f335328d05e417f9c5efecb638e3c8e151"
EXPECTED_V24_QUEUE_COUNT = 599

V24_RUN_DIR = Path("evaluation/sers_fresh_c/c0_1c_v2_4_recovery_run_v1")
V24_RUN_MANIFEST_PATH = V24_RUN_DIR / "run_manifest.json"
V24_QUEUE_PATH = V24_RUN_DIR / "blind_selection_queue.json"
V24_LOCATOR_PATH = V24_RUN_DIR / "access_locator_manifest.json"
V24_DIAGNOSTICS_PATH = V24_RUN_DIR / "TRANSPORT_DIAGNOSTICS.json"
V24_STARTED_PATH = V24_RUN_DIR / "DISCOVERY_RECOVERY_STARTED.json"
V24_COMPLETE_PATH = V24_RUN_DIR / "DISCOVERY_RECOVERY_COMPLETE.json"

DEFAULT_PROTOCOL_PATH = Path(
    "dac_her/sers_fresh_c_content_acquisition_v1_protocol.json"
)
DEFAULT_PROTOCOL_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c0_1d_content_acquisition_protocol_freeze_v1"
)
DEFAULT_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c0_1d_content_acquisition_run_v1"
)
DEFAULT_RESULT_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c0_1d_content_acquisition_result_freeze_v1"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContentAcquisitionProtocol(StrictModel):
    schema_version: Literal["sers-fresh-c-content-acquisition-protocol-v1"]
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantics_id: Literal["sers_fresh_c_blind_oa_content_acquisition_v1"]
    stage: Literal["C0.1D"]

    upstream_v24_protocol_id: Literal[
        "sers_fresh_c_openalex_crossref_substitution_protocol_v2_4:5e09cf20c06210a742a5"
    ]
    upstream_v24_protocol_sha256: Literal[
        "99389342726c2bc2deab5fde0b2aacf3b52ea9b4a6f6981578f2b73008220fcb"
    ]
    upstream_v24_freeze_id: Literal[
        "sers_fresh_c_openalex_crossref_substitution_freeze_v1:1ad72017775ef79662dc"
    ]
    upstream_v24_freeze_manifest_sha256: Literal[
        "7ad250b56946e4954b55e7e0b7de01e22a9675890a581697a4fcd3f7bd9b14be"
    ]
    upstream_v24_run_id: Literal[
        "sers_fresh_c_openalex_crossref_run_v2_4:220ee42f03afe8aaa1e1"
    ]
    upstream_v24_run_sha256: Literal[
        "813e0869c8c4b82971669de65bbcd7f335328d05e417f9c5efecb638e3c8e151"
    ]
    upstream_blind_queue_count: Literal[599]
    upstream_provider_query_executions_successful: Literal[8]
    upstream_semantic_read_performed: Literal[False]
    upstream_fresh_reserve_c_consumed: Literal[False]

    target_successful_pdf_count: Literal[25]
    maximum_identity_attempts: Literal[599]
    selection_rule: Literal[
        "first_25_successful_verified_oa_pdfs_in_frozen_blind_order"
    ]
    replacement_rule: Literal[
        "oa_failure_advances_to_next_frozen_identity_without_manual_replacement"
    ]
    manual_candidate_replacement_allowed: Literal[False]
    hypothesis_aware_selection_allowed: Literal[False]
    title_abstract_scoring_allowed: Literal[False]
    scientific_metadata_inspection_allowed: Literal[False]

    use_unpaywall: Literal[True]
    use_openalex: Literal[True]
    use_catalog_open_access_url: Literal[True]
    unpaywall_email_env: Literal["UNPAYWALL_EMAIL"]
    openalex_api_key_env: Literal["OPENALEX_API_KEY"]
    openalex_mailto_env: Literal["OPENALEX_MAILTO"]
    fallback_email_env: Literal["CROSSREF_MAILTO"]
    credential_values_persisted: Literal[False]

    request_timeout_seconds: Literal[45.0]
    retries: Literal[2]
    retry_backoff_seconds: Literal[1.0]
    resolver_delay_seconds: Literal[0.1]
    max_artifact_bytes: Literal[104857600]
    require_pdf_magic: Literal[True]
    try_all_direct_pdf_locations: Literal[True]
    send_landing_page_referer: Literal[True]
    auto_download_main: Literal[True]
    paywall_bypass_allowed: Literal[False]

    pdf_text_extraction_allowed: Literal[False]
    pdf_semantic_read_allowed: Literal[False]
    llm_calls: Literal[0]
    scientific_reassessment_allowed: Literal[False]
    positive_evidence_promotion_allowed: Literal[False]

    reserve_c_identity_selection_finalized_on_success: Literal[True]
    reserve_c_content_sealed_on_success: Literal[True]
    fresh_reserve_c_consumed_on_success: Literal[False]
    automatic_c1_transition_allowed: Literal[False]
    stop_after_success: Literal[True]

    @model_validator(mode="after")
    def _check_exact_contract(self) -> "ContentAcquisitionProtocol":
        if self.maximum_identity_attempts != self.upstream_blind_queue_count:
            raise ValueError("C0.1D must allow scanning exactly the full frozen queue.")
        return self


def _payload_sha(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def _protocol_identity_sha(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("protocol_id", None)
    value.pop("protocol_sha256", None)
    return sha256_json(value)


def expected_protocol_id(payload: Mapping[str, Any]) -> str:
    return C01D_PROTOCOL_PREFIX + ":" + _protocol_identity_sha(payload)[:20]


def load_and_validate_protocol(path: Path) -> ContentAcquisitionProtocol:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("C0.1D protocol must be a JSON object.")
    protocol = ContentAcquisitionProtocol.model_validate(raw)
    if protocol.protocol_id != expected_protocol_id(raw):
        raise ValueError("C0.1D protocol ID mismatch.")
    if protocol.protocol_sha256 != _payload_sha(raw, "protocol_sha256"):
        raise ValueError("C0.1D protocol SHA mismatch.")
    return protocol


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_credentials() -> None:
    if not os.getenv("OPENALEX_API_KEY"):
        raise RuntimeError("OPENALEX_API_KEY is required for C0.1D.")
    if not (
        os.getenv("UNPAYWALL_EMAIL")
        or os.getenv("CROSSREF_MAILTO")
    ):
        raise RuntimeError(
            "UNPAYWALL_EMAIL or CROSSREF_MAILTO is required for C0.1D."
        )


def source_policy(protocol: ContentAcquisitionProtocol) -> SourceAcquisitionPolicy:
    return SourceAcquisitionPolicy(
        policy_id="sers_fresh_c_c0_1d_oa_pdf_policy_v1",
        unpaywall_email_env=protocol.unpaywall_email_env,
        fallback_email_env=protocol.fallback_email_env,
        openalex_api_key_env=protocol.openalex_api_key_env,
        openalex_mailto_env=protocol.openalex_mailto_env,
        openalex_require_api_key=True,
        use_unpaywall=True,
        use_openalex=True,
        use_catalog_open_access_url=True,
        request_timeout_seconds=protocol.request_timeout_seconds,
        retries=protocol.retries,
        retry_backoff_seconds=protocol.retry_backoff_seconds,
        resolver_delay_seconds=protocol.resolver_delay_seconds,
        max_artifact_bytes=protocol.max_artifact_bytes,
        require_pdf_magic=True,
        try_all_direct_pdf_locations=True,
        send_landing_page_referer=True,
        auto_download_main=True,
        allow_catalog_oa_fallback=True,
        supplementary_discovery="deferred_to_m3_1",
    )


def canonical_id_to_doi(canonical_id: str) -> str | None:
    text = str(canonical_id or "").strip()
    if text.lower().startswith("doi:"):
        doi = text[4:].strip().lower()
        return doi or None
    return None


def locator_record_to_minimal_work(record: Mapping[str, Any]):
    from dac_her.literature_catalog_contracts import CatalogWork

    canonical_id = str(record.get("canonical_id") or "").strip()
    if not canonical_id:
        raise ValueError("Locator record missing canonical_id.")

    doi = str(record.get("doi") or "").strip().lower() or canonical_id_to_doi(
        canonical_id
    )
    url = str(record.get("url") or "").strip() or None
    open_access_url = str(record.get("open_access_url") or "").strip() or None
    provider_ids = record.get("provider_ids")
    if not isinstance(provider_ids, dict):
        provider_ids = {}

    # Do not read or propagate scientific title text in C0.1D.
    return CatalogWork(
        work_id=canonical_id,
        title=f"sealed_identity:{canonical_id}",
        doi=doi,
        url=url,
        open_access_url=open_access_url,
        authors=[],
        publication_types=[],
        providers=sorted(str(key) for key in provider_ids),
        provider_ids={
            str(key): str(value)
            for key, value in provider_ids.items()
            if value is not None
        },
        retrieval_query_ids=[],
        retrieval_axis_ids=[],
    )


def validate_upstream_v24(root: Path) -> dict[str, Any]:
    run = load_json_object(root / V24_RUN_MANIFEST_PATH)
    queue = load_json_object(root / V24_QUEUE_PATH)
    locators = load_json_object(root / V24_LOCATOR_PATH)
    complete = load_json_object(root / V24_COMPLETE_PATH)

    if run.get("run_id") != EXPECTED_V24_RUN_ID:
        raise ValueError("C0.1D upstream v2.4 run ID drifted.")
    if run.get("run_sha256") != EXPECTED_V24_RUN_SHA256:
        raise ValueError("C0.1D upstream v2.4 run SHA drifted.")
    if run.get("successful_provider_query_executions") != 8:
        raise ValueError("C0.1D requires v2.4 8/8 provider success.")
    if run.get("fresh_identity_queue_count") != EXPECTED_V24_QUEUE_COUNT:
        raise ValueError("C0.1D upstream queue count drifted.")
    if run.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("Upstream v2.4 already consumed Fresh C.")
    if run.get("semantic_read_performed") is not False:
        raise ValueError("Upstream v2.4 performed semantic read.")
    if complete.get("run_id") != EXPECTED_V24_RUN_ID:
        raise ValueError("v2.4 COMPLETE run ID drifted.")

    queue_records = queue.get("records") or []
    locator_records = locators.get("records") or []
    if len(queue_records) != EXPECTED_V24_QUEUE_COUNT:
        raise ValueError("C0.1D blind queue length drifted.")
    if len(locator_records) != EXPECTED_V24_QUEUE_COUNT:
        raise ValueError("C0.1D locator count drifted.")

    queue_ids = [str(row.get("canonical_id") or "") for row in queue_records]
    locator_map = {
        str(row.get("canonical_id") or ""): row
        for row in locator_records
    }
    if len(locator_map) != EXPECTED_V24_QUEUE_COUNT:
        raise ValueError("C0.1D locator canonical IDs are not unique.")
    if set(queue_ids) != set(locator_map):
        raise ValueError("C0.1D queue/locator identity sets differ.")
    if len(queue_ids) != len(set(queue_ids)):
        raise ValueError("C0.1D blind queue contains duplicates.")
    for index, row in enumerate(queue_records, start=1):
        if row.get("rank") != index:
            raise ValueError("C0.1D blind queue rank drifted.")

    return {
        "run": run,
        "queue": queue,
        "locators": locators,
        "queue_ids": queue_ids,
        "locator_map": locator_map,
    }


def seal_payload(selected: list[dict[str, Any]]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "sers-fresh-c-content-seal-v1",
        "selection_rule": (
            "first_25_successful_verified_oa_pdfs_in_frozen_blind_order"
        ),
        "selected_count": len(selected),
        "records": selected,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "pdf_text_extraction_performed": False,
        "llm_calls": 0,
    }
    body["content_seal_sha256"] = _payload_sha(
        body, "content_seal_sha256"
    )
    return body
