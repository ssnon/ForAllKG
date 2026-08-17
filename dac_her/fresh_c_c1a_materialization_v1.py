from __future__ import annotations

import hashlib
import importlib.metadata
import json
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.fresh_c_acquisition import sha256_file, sha256_json

C1A_SEMANTICS_ID = "sers_fresh_c_c1a_irreversible_local_text_materialization_v1"
C1A_PROTOCOL_PREFIX = "sers_fresh_c_c1a_materialization_protocol_v1"

EXPECTED_C01D_PROTOCOL_FREEZE_ID = "sers_fresh_c_content_acquisition_protocol_freeze_v1:5fc48f32b139793ee86b"
EXPECTED_C01D_PROTOCOL_FREEZE_SHA256 = "d4fb7533f04ce226d47244f65d4f0a1269622234a028c7366728b19e1722a682"
EXPECTED_C01D_RUN_ID = "sers_fresh_c_content_acquisition_run_v1:310a3fbc7c0311cd0ca1"
EXPECTED_C01D_RUN_SHA256 = "7fc1ed5a5c5cc10c77951f973b2452cf0f70f665f500bd269295d7c178e5f0c7"
EXPECTED_C01D_CONTENT_SEAL_SHA256 = "55b240222446fa30c81faf2df3841245f97d821d4e0b8e7f4d8f92127a9f9d81"
EXPECTED_C01D_RESULT_FREEZE_ID = "sers_fresh_c_content_acquisition_result_freeze_v1:afc55cfdc78819827cde"
EXPECTED_C01D_RESULT_FREEZE_SHA256 = "c0686755c472dd936f5e58a3bea9599eb32b259b884a2fe51d65b427976fbf84"

EXPECTED_PDFTEXT_VERSION = "0.6.3"
EXPECTED_PYPDFIUM2_VERSION = "4.30.0"
EXPECTED_SELECTED_COUNT = 25

C01D_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c0_1d_content_acquisition_run_v1"
)
C01D_SELECTED_PATH = C01D_RUN_DIR / "selected_reserve_c.json"
C01D_RUN_MANIFEST_PATH = C01D_RUN_DIR / "run_manifest.json"
C01D_CONTENT_SEAL_PATH = C01D_RUN_DIR / "content_seal.json"
C01D_RESULT_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c0_1d_content_acquisition_result_freeze_v1"
)
C01D_RESULT_FREEZE_MANIFEST = C01D_RESULT_FREEZE_DIR / "freeze_manifest.json"

DEFAULT_PROTOCOL_PATH = Path(
    "dac_her/sers_fresh_c_c1a_materialization_v1_protocol.json"
)
DEFAULT_PROTOCOL_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c1a_materialization_protocol_freeze_v1"
)
DEFAULT_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c1a_materialization_run_v1"
)
DEFAULT_RESULT_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c1a_materialization_result_freeze_v1"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class C1AMaterializationProtocol(StrictModel):
    schema_version: Literal["sers-fresh-c-c1a-materialization-protocol-v1"]
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantics_id: Literal[
        "sers_fresh_c_c1a_irreversible_local_text_materialization_v1"
    ]
    stage: Literal["C1A"]

    c01d_protocol_freeze_id: Literal[
        "sers_fresh_c_content_acquisition_protocol_freeze_v1:5fc48f32b139793ee86b"
    ]
    c01d_protocol_freeze_sha256: Literal[
        "d4fb7533f04ce226d47244f65d4f0a1269622234a028c7366728b19e1722a682"
    ]
    c01d_run_id: Literal[
        "sers_fresh_c_content_acquisition_run_v1:310a3fbc7c0311cd0ca1"
    ]
    c01d_run_sha256: Literal[
        "7fc1ed5a5c5cc10c77951f973b2452cf0f70f665f500bd269295d7c178e5f0c7"
    ]
    c01d_content_seal_sha256: Literal[
        "55b240222446fa30c81faf2df3841245f97d821d4e0b8e7f4d8f92127a9f9d81"
    ]
    c01d_result_freeze_id: Literal[
        "sers_fresh_c_content_acquisition_result_freeze_v1:afc55cfdc78819827cde"
    ]
    c01d_result_freeze_sha256: Literal[
        "c0686755c472dd936f5e58a3bea9599eb32b259b884a2fe51d65b427976fbf84"
    ]
    selected_pdf_count: Literal[25]

    consumption_marker_before_first_text_extraction: Literal[True]
    fresh_reserve_c_consumed_at_marker_write: Literal[True]
    consumption_irreversible: Literal[True]
    same_epoch_rerun_after_marker_allowed: Literal[False]
    failure_restores_freshness: Literal[False]

    materializer: Literal["pdftext_plain_text_v0_6_3"]
    pdftext_version: Literal["0.6.3"]
    pypdfium2_version: Literal["4.30.0"]
    extraction_sort_reading_order: Literal[True]
    extraction_keep_hyphens: Literal[False]
    extraction_workers: Literal[0]
    all_25_pdfs_must_be_materialized: Literal[True]
    empty_extraction_is_failure: Literal[True]
    identity_replacement_allowed_after_c01d: Literal[False]

    network_allowed_during_materialization: Literal[False]
    socket_network_guard_required: Literal[True]
    external_literature_lookup_allowed: Literal[False]
    llm_calls: Literal[0]
    ocr_performed: Literal[False]
    scientific_reviewer_read_performed: Literal[False]
    scientific_adjudication_performed: Literal[False]
    hypothesis_state_mutation_allowed: Literal[False]
    positive_evidence_promotion_allowed: Literal[False]

    materialized_text_persisted: Literal[True]
    page_boundaries_persisted: Literal[True]
    source_pdf_sha256_reverified: Literal[True]
    text_sha256_required: Literal[True]
    automatic_c1b_transition_allowed: Literal[False]
    stop_after_success: Literal[True]

    @model_validator(mode="after")
    def _check_contract(self) -> "C1AMaterializationProtocol":
        if self.selected_pdf_count != EXPECTED_SELECTED_COUNT:
            raise ValueError("C1A selected count drifted.")
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
    return C1A_PROTOCOL_PREFIX + ":" + _protocol_identity_sha(payload)[:20]


def load_and_validate_protocol(path: Path) -> C1AMaterializationProtocol:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("C1A protocol must be a JSON object.")
    protocol = C1AMaterializationProtocol.model_validate(raw)
    if protocol.protocol_id != expected_protocol_id(raw):
        raise ValueError("C1A protocol ID mismatch.")
    if protocol.protocol_sha256 != _payload_sha(raw, "protocol_sha256"):
        raise ValueError("C1A protocol SHA mismatch.")
    return protocol


def load_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return raw


def package_versions() -> dict[str, str]:
    return {
        "pdftext": importlib.metadata.version("pdftext"),
        "pypdfium2": importlib.metadata.version("pypdfium2"),
    }


def validate_package_versions() -> dict[str, str]:
    versions = package_versions()
    if versions["pdftext"] != EXPECTED_PDFTEXT_VERSION:
        raise RuntimeError(
            "C1A pdftext version mismatch: "
            f"{versions['pdftext']} != {EXPECTED_PDFTEXT_VERSION}"
        )
    if versions["pypdfium2"] != EXPECTED_PYPDFIUM2_VERSION:
        raise RuntimeError(
            "C1A pypdfium2 version mismatch: "
            f"{versions['pypdfium2']} != {EXPECTED_PYPDFIUM2_VERSION}"
        )
    return versions


def validate_c01d_closed_frozen(root: Path) -> dict[str, Any]:
    selected = load_json_object(root / C01D_SELECTED_PATH)
    run = load_json_object(root / C01D_RUN_MANIFEST_PATH)
    seal = load_json_object(root / C01D_CONTENT_SEAL_PATH)
    freeze = load_json_object(root / C01D_RESULT_FREEZE_MANIFEST)

    if run.get("run_id") != EXPECTED_C01D_RUN_ID:
        raise ValueError("C1A C0.1D run ID drifted.")
    if run.get("run_sha256") != EXPECTED_C01D_RUN_SHA256:
        raise ValueError("C1A C0.1D run SHA drifted.")
    if seal.get("content_seal_sha256") != EXPECTED_C01D_CONTENT_SEAL_SHA256:
        raise ValueError("C1A C0.1D content seal drifted.")
    if freeze.get("freeze_id") != EXPECTED_C01D_RESULT_FREEZE_ID:
        raise ValueError("C1A C0.1D result freeze ID drifted.")
    if freeze.get("manifest_sha256") != EXPECTED_C01D_RESULT_FREEZE_SHA256:
        raise ValueError("C1A C0.1D result freeze SHA drifted.")
    if freeze.get("selected_verified_pdf_count") != EXPECTED_SELECTED_COUNT:
        raise ValueError("C1A C0.1D selected count drifted.")
    if freeze.get("reserve_c_identity_selection_finalized") is not True:
        raise ValueError("C1A requires finalized Reserve-C identities.")
    if freeze.get("reserve_c_content_sealed") is not True:
        raise ValueError("C1A requires sealed Reserve-C PDF content.")
    if freeze.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("C1A expected Fresh C to be unconsumed before marker.")
    if freeze.get("semantic_read_performed") is not False:
        raise ValueError("C1A expected no prior semantic read.")

    records = selected.get("records") or []
    if len(records) != EXPECTED_SELECTED_COUNT:
        raise ValueError("C1A selected_reserve_c records must be exactly 25.")

    seen_ids: set[str] = set()
    normalized = []
    frozen_hashes = freeze.get("selected_pdf_sha256") or {}
    for index, row in enumerate(records, start=1):
        if row.get("reserve_index") != index:
            raise ValueError("C1A reserve index drifted.")
        canonical_id = str(row.get("canonical_id") or "").strip()
        if not canonical_id or canonical_id in seen_ids:
            raise ValueError("C1A canonical ID missing or duplicate.")
        seen_ids.add(canonical_id)
        path = root / str(row.get("local_path") or "")
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_sha = str(row.get("artifact_sha256") or "")
        if frozen_hashes.get(canonical_id) != expected_sha:
            raise ValueError("C1A C0.1D freeze PDF hash map drifted.")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError("C1A sealed PDF SHA256 drifted.")
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError("C1A sealed source lost PDF magic.")
        normalized.append({
            "reserve_index": index,
            "canonical_id": canonical_id,
            "source_path": str(path.relative_to(root)),
            "source_pdf_sha256": expected_sha,
        })

    return {
        "selected": selected,
        "run": run,
        "seal": seal,
        "freeze": freeze,
        "records": normalized,
    }


@contextmanager
def network_disabled() -> Iterator[None]:
    """Process-local fail-closed guard against socket-based network access."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo

    def blocked(*args: Any, **kwargs: Any):
        raise RuntimeError("C1A_NETWORK_DISABLED")

    try:
        socket.socket.connect = blocked  # type: ignore[assignment]
        socket.socket.connect_ex = blocked  # type: ignore[assignment]
        socket.create_connection = blocked  # type: ignore[assignment]
        socket.getaddrinfo = blocked  # type: ignore[assignment]
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[assignment]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[assignment]
        socket.create_connection = original_create_connection  # type: ignore[assignment]
        socket.getaddrinfo = original_getaddrinfo  # type: ignore[assignment]


def materialize_pdf_pages(source_path: Path) -> list[str]:
    # pdftext 0.6.3 is a local pypdfium2-based text extractor.
    from pdftext.extraction import paginated_plain_text_output

    with network_disabled():
        pages = paginated_plain_text_output(
            str(source_path),
            sort=True,
            hyphens=False,
            page_range=None,
            flatten_pdf=False,
            workers=None,
        )
    if not isinstance(pages, list):
        raise RuntimeError("C1A pdftext returned non-list page output.")
    normalized = [str(page or "").strip() for page in pages]
    if not any(normalized):
        raise RuntimeError("C1A_EMPTY_TEXT_EXTRACTION")
    return normalized


def render_page_bounded_text(pages: list[str]) -> str:
    chunks = []
    for index, page in enumerate(pages, start=1):
        chunks.append(f"[[PAGE {index}]]\n{page}")
    return "\n\n".join(chunks).strip() + "\n"
