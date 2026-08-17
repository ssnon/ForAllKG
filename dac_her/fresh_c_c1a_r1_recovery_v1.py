from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from dac_her.fresh_c_acquisition import sha256_file, sha256_json
from dac_her.fresh_c_c1a_materialization_v1 import (
    C01D_RESULT_FREEZE_MANIFEST,
    C01D_SELECTED_PATH,
    DEFAULT_PROTOCOL_FREEZE_DIR as C1A_PROTOCOL_FREEZE_DIR,
    DEFAULT_RUN_DIR as C1A_FAILED_RUN_DIR,
    load_json_object,
    validate_c01d_closed_frozen,
)

C1AR1_SEMANTICS_ID = "sers_fresh_c_c1a_r1_post_consumption_structural_recovery_v1"
C1AR1_PROTOCOL_PREFIX = "sers_fresh_c_c1a_r1_recovery_protocol_v1"

EXPECTED_C01D_RESULT_FREEZE_ID = "sers_fresh_c_content_acquisition_result_freeze_v1:afc55cfdc78819827cde"
EXPECTED_C01D_RESULT_FREEZE_SHA256 = "c0686755c472dd936f5e58a3bea9599eb32b259b884a2fe51d65b427976fbf84"
EXPECTED_C01D_CONTENT_SEAL_SHA256 = "55b240222446fa30c81faf2df3841245f97d821d4e0b8e7f4d8f92127a9f9d81"
EXPECTED_C1A_PROTOCOL_ID = "sers_fresh_c_c1a_materialization_protocol_v1:fa8658ea777fe70ab355"
EXPECTED_C1A_PROTOCOL_SHA256 = "c3656c32937c700467bb7b4e7b8f5be828da85445be632d3a2b0e560b4eafe06"
EXPECTED_C1A_FREEZE_ID = "sers_fresh_c_c1a_materialization_protocol_freeze_v1:8ba12585eb5b9bc620af"
EXPECTED_C1A_FREEZE_SHA256 = "5802f2757a352ebbd20f5834dafecf20157a46bf39e88ce7107d540572a6deba"
EXPECTED_FAILED_INDEX = 14
EXPECTED_FAILED_CANONICAL_ID = "doi:10.1021/acs.jpcc.8b01309"
EXPECTED_MATERIALIZED_BEFORE_FAILURE = 13
EXPECTED_SELECTED_COUNT = 25
EXPECTED_PDFMINER_SIX_VERSION = "20260107"

DEFAULT_PROTOCOL_PATH = Path(
    "dac_her/sers_fresh_c_c1a_r1_recovery_v1_protocol.json"
)
DEFAULT_PROTOCOL_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c1a_r1_recovery_protocol_freeze_v1"
)
DEFAULT_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c1a_r1_recovery_run_v1"
)
DEFAULT_RESULT_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c1a_r1_recovery_result_freeze_v1"
)

FAILED_CONSUMPTION_MARKER = (
    C1A_FAILED_RUN_DIR / "RESERVE_C_CONSUMPTION_STARTED.json"
)
FAILED_MARKER = C1A_FAILED_RUN_DIR / "C1A_MATERIALIZATION_FAILED.json"
FAILED_AUDIT = C1A_FAILED_RUN_DIR / "FAILED_EPOCH_AUDIT.json"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class C1AR1RecoveryProtocol(StrictModel):
    schema_version: Literal["sers-fresh-c-c1a-r1-recovery-protocol-v1"]
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantics_id: Literal[
        "sers_fresh_c_c1a_r1_post_consumption_structural_recovery_v1"
    ]
    stage: Literal["C1A-R1"]

    c01d_result_freeze_id: Literal[
        "sers_fresh_c_content_acquisition_result_freeze_v1:afc55cfdc78819827cde"
    ]
    c01d_result_freeze_sha256: Literal[
        "c0686755c472dd936f5e58a3bea9599eb32b259b884a2fe51d65b427976fbf84"
    ]
    c01d_content_seal_sha256: Literal[
        "55b240222446fa30c81faf2df3841245f97d821d4e0b8e7f4d8f92127a9f9d81"
    ]
    c1a_protocol_id: Literal[
        "sers_fresh_c_c1a_materialization_protocol_v1:fa8658ea777fe70ab355"
    ]
    c1a_protocol_sha256: Literal[
        "c3656c32937c700467bb7b4e7b8f5be828da85445be632d3a2b0e560b4eafe06"
    ]
    c1a_protocol_freeze_id: Literal[
        "sers_fresh_c_c1a_materialization_protocol_freeze_v1:8ba12585eb5b9bc620af"
    ]
    c1a_protocol_freeze_sha256: Literal[
        "5802f2757a352ebbd20f5834dafecf20157a46bf39e88ce7107d540572a6deba"
    ]

    source_identity_count: Literal[25]
    source_identity_set_must_remain_exact: Literal[True]
    source_pdf_sha256_set_must_remain_exact: Literal[True]
    identity_replacement_allowed: Literal[False]
    redownload_allowed: Literal[False]
    prior_failed_outputs_reused: Literal[False]

    fresh_reserve_c_already_consumed: Literal[True]
    consumption_irreversible: Literal[True]
    original_c1a_same_epoch_rerun_allowed: Literal[False]
    recovery_same_epoch_rerun_after_start_allowed: Literal[False]

    primary_extractor: Literal["pdfminer_six_full_page_text_v1"]
    pdfminer_six_version: Literal["20260107"]
    structural_repair_trigger: Literal[
        "primary_structural_failure_or_zero_page_or_zero_text"
    ]
    structural_repair_tool: Literal["mutool_clean"]
    structural_repair_binary_sha256_frozen_before_execution: Literal[True]
    structural_repair_version_frozen_before_execution: Literal[True]
    structural_repair_derivative_persisted: Literal[True]
    original_pdf_overwrite_allowed: Literal[False]
    repaired_derivative_sha256_required: Literal[True]
    repaired_derivative_must_be_reparsed_by_primary_extractor: Literal[True]

    require_page_traversal: Literal[True]
    require_at_least_one_page: Literal[True]
    require_nonzero_document_text: Literal[True]
    page_boundaries_persisted: Literal[True]
    all_25_sources_must_reach_materialized_status: Literal[True]

    direct_positive_evidence_from_materialized_text_allowed_later: Literal[True]
    negative_absence_inference_from_any_single_paper_allowed: Literal[False]
    repaired_derivative_completeness_claim_allowed: Literal[False]
    scientific_reviewer_read_performed_in_recovery: Literal[False]
    scientific_adjudication_performed_in_recovery: Literal[False]
    hypothesis_state_mutation_allowed: Literal[False]
    positive_evidence_promotion_performed_in_recovery: Literal[False]

    external_literature_lookup_allowed: Literal[False]
    network_calls_allowed: Literal[False]
    llm_calls: Literal[0]
    ocr_performed: Literal[False]
    automatic_c1b_transition_allowed: Literal[False]
    stop_after_success: Literal[True]


def _payload_sha(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def _identity_sha(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("protocol_id", None)
    value.pop("protocol_sha256", None)
    return sha256_json(value)


def expected_protocol_id(payload: Mapping[str, Any]) -> str:
    return C1AR1_PROTOCOL_PREFIX + ":" + _identity_sha(payload)[:20]


def load_and_validate_protocol(path: Path) -> C1AR1RecoveryProtocol:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("C1A-R1 protocol must be a JSON object.")
    protocol = C1AR1RecoveryProtocol.model_validate(raw)
    if protocol.protocol_id != expected_protocol_id(raw):
        raise ValueError("C1A-R1 protocol ID mismatch.")
    if protocol.protocol_sha256 != _payload_sha(raw, "protocol_sha256"):
        raise ValueError("C1A-R1 protocol SHA mismatch.")
    return protocol


def validate_failed_c1a_state(root: Path) -> dict[str, Any]:
    validate_c01d_closed_frozen(root)

    c1a_freeze = load_json_object(
        root / C1A_PROTOCOL_FREEZE_DIR / "freeze_manifest.json"
    )
    if c1a_freeze.get("protocol_id") != EXPECTED_C1A_PROTOCOL_ID:
        raise ValueError("C1A-R1 parent C1A protocol ID drifted.")
    if c1a_freeze.get("protocol_sha256") != EXPECTED_C1A_PROTOCOL_SHA256:
        raise ValueError("C1A-R1 parent C1A protocol SHA drifted.")
    if c1a_freeze.get("freeze_id") != EXPECTED_C1A_FREEZE_ID:
        raise ValueError("C1A-R1 parent C1A freeze ID drifted.")
    if c1a_freeze.get("manifest_sha256") != EXPECTED_C1A_FREEZE_SHA256:
        raise ValueError("C1A-R1 parent C1A freeze SHA drifted.")

    marker = load_json_object(root / FAILED_CONSUMPTION_MARKER)
    failed = load_json_object(root / FAILED_MARKER)
    audit = load_json_object(root / FAILED_AUDIT)

    if marker.get("fresh_reserve_c_consumed") is not True:
        raise ValueError("C1A-R1 requires already-consumed Fresh C.")
    if marker.get("consumption_irreversible") is not True:
        raise ValueError("C1A-R1 requires irreversible consumption.")
    if marker.get("same_epoch_rerun_allowed") is not False:
        raise ValueError("Original C1A same epoch must remain non-rerunnable.")

    if failed.get("fresh_reserve_c_consumed") is not True:
        raise ValueError("C1A failure marker consumption state drifted.")
    if failed.get("materialized_pdf_count_before_failure") != EXPECTED_MATERIALIZED_BEFORE_FAILURE:
        raise ValueError("C1A failed materialized count drifted.")
    if failed.get("identity_replacement_allowed") is not False:
        raise ValueError("C1A failure marker identity policy drifted.")

    if audit.get("failed_reserve_index") != EXPECTED_FAILED_INDEX:
        raise ValueError("C1A failed reserve index drifted.")
    if audit.get("failed_canonical_id") != EXPECTED_FAILED_CANONICAL_ID:
        raise ValueError("C1A failed canonical identity drifted.")
    if audit.get("scientific_reviewer_read_performed") is not False:
        raise ValueError("C1A failed epoch unexpectedly scientific-read.")
    if audit.get("scientific_adjudication_performed") is not False:
        raise ValueError("C1A failed epoch unexpectedly adjudicated.")

    return {
        "consumption_marker": marker,
        "failure_marker": failed,
        "failure_audit": audit,
    }


def pdfminer_version() -> str:
    return importlib.metadata.version("pdfminer.six")


def validate_pdfminer_version() -> str:
    version = pdfminer_version()
    if version != EXPECTED_PDFMINER_SIX_VERSION:
        raise RuntimeError(
            "C1A-R1 pdfminer.six version mismatch: "
            f"{version} != {EXPECTED_PDFMINER_SIX_VERSION}"
        )
    return version


def mutool_executable() -> Path:
    command = shutil.which("mutool")
    if command is None:
        raise RuntimeError("C1A-R1 mutool command not found.")
    return Path(command).resolve()


def mutool_version(binary: Path) -> str:
    completed = subprocess.run(
        [str(binary), "-v"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    text = completed.stdout.strip()
    if completed.returncode != 0 or not text:
        raise RuntimeError(
            "C1A-R1 failed to read mutool version: "
            f"exit={completed.returncode}"
        )
    return text


def mutool_fingerprint() -> dict[str, str]:
    binary = mutool_executable()
    return {
        "path": str(binary),
        "sha256": sha256_file(binary),
        "version_output": mutool_version(binary),
    }


def _page_text(layout: Any) -> str:
    from pdfminer.layout import LTTextContainer

    chunks: list[str] = []
    for element in layout:
        if isinstance(element, LTTextContainer):
            chunks.append(element.get_text())
    return "".join(chunks).strip()


def extract_pdfminer_pages(path: Path) -> list[str]:
    from pdfminer.high_level import extract_pages

    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    pages = [_page_text(layout) for layout in extract_pages(str(path))]
    if not pages:
        raise RuntimeError("C1AR1_ZERO_PAGE_EXTRACTION")
    if not any("".join(page.split()) for page in pages):
        raise RuntimeError("C1AR1_ZERO_DOCUMENT_TEXT")
    return pages


def render_page_bounded_text(pages: list[str]) -> str:
    return (
        "\n\n".join(
            f"[[PAGE {index}]]\n{page}"
            for index, page in enumerate(pages, start=1)
        ).strip()
        + "\n"
    )


def repair_with_mutool(
    *,
    binary: Path,
    source: Path,
    derivative: Path,
    log_path: Path,
) -> dict[str, Any]:
    derivative.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(binary), "clean", str(source), str(derivative)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            "C1AR1_MUTool_REPAIR_FAILED:"
            f"exit={completed.returncode}"
        )
    if not derivative.is_file() or derivative.stat().st_size == 0:
        raise RuntimeError("C1AR1_MUTool_REPAIR_PRODUCED_NO_DERIVATIVE")
    data = derivative.read_bytes()
    if not data.startswith(b"%PDF-"):
        raise RuntimeError("C1AR1_REPAIRED_DERIVATIVE_LOST_PDF_MAGIC")
    return {
        "exit_code": completed.returncode,
        "derivative_sha256": sha256_file(derivative),
        "derivative_byte_count": derivative.stat().st_size,
        "eof_present": b"%%EOF" in data,
        "startxref_present": b"startxref" in data,
        "log_sha256": sha256_file(log_path),
    }
