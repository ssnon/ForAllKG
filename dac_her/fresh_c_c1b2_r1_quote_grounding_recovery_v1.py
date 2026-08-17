from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from dac_her.fresh_c_c1b1_reviewer_contract_v1 import (
    FreshCFinalAdjudication,
    FreshCPaperReview,
    canonical_json_sha256,
)
from dac_her.fresh_c_c1b2_scientific_adjudication_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR as PARENT_PROTOCOL_FREEZE_DIR,
    DEFAULT_RUN_DIR as PARENT_RUN_DIR,
    EXPECTED_BASE_URL,
    EXPECTED_MODEL,
    load_object,
    openai_strict_transport_schema,
    validate_corpus_metadata,
    validate_frozen_lineage,
    validate_protocol as validate_parent_protocol,
)

STAGE = "C1B.2-R1"
SEMANTICS_ID = "sers_fresh_c_c1b2_r1_quote_grounding_recovery_v1"
PROTOCOL_PREFIX = "sers_fresh_c_c1b2_r1_recovery_protocol_v1"

EXPECTED_PARENT_PROTOCOL_ID = "sers_fresh_c_c1b2_scientific_protocol_v1:231b259d4eedb766c4a2"
EXPECTED_PARENT_PROTOCOL_SHA256 = "e4557ad77ffa11f2d954e95e0c140dbfc56d9ea0aa7b89d6c64ac31402fb2e70"
EXPECTED_PARENT_FREEZE_ID = "sers_fresh_c_c1b2_scientific_protocol_freeze_v1:cd9065ffee576865bd09"
EXPECTED_PARENT_FREEZE_SHA256 = "01bde9481335febe4ddec8a18405a31736e400700e10cbdb3e6b240f6e740202"
EXPECTED_PARENT_SOURCE_COMMIT = "49f9be5e4d4f6863f21e42bcd6bbde7467240439"

EXPECTED_FAILURE_TYPE = "ValueError"
EXPECTED_FAILURE_FRAGMENT = "Verbatim quote is not an exact whitespace-normalized substring"

RECOVERY_SCHEMA_ADAPTER_ID = "c1b2_r1_verbatim_quote_null_only_v1"
EXPECTED_PAPER_RECOVERY_SCHEMA_SHA256 = "fa46acf10252d27f4c7b7b1e97d1584da70e0230fdbda015e9bcfbee48c24b48"
EXPECTED_FINAL_RECOVERY_SCHEMA_SHA256 = "7e475579c8f3ecbca7ff36a6d5ef0d1b545d990a3c9d0398b95df0e991da0856"

DEFAULT_PROTOCOL_PATH = Path("dac_her/sers_fresh_c_c1b2_r1_recovery_protocol_v1.json")
DEFAULT_SCHEMA_QUALIFICATION_DIR = Path(
    "evaluation/sers_fresh_c/c1b2_r1_quote_null_schema_qualification_v1"
)
DEFAULT_PROTOCOL_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c1b2_r1_recovery_protocol_freeze_v1"
)
DEFAULT_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c1b2_r1_scientific_recovery_run_v1"
)
DEFAULT_RESULT_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c1b2_r1_scientific_recovery_result_freeze_v1"
)


def paper_recovery_transport_schema() -> dict[str, Any]:
    schema = copy.deepcopy(openai_strict_transport_schema(FreshCPaperReview))
    locator = schema["$defs"]["EvidenceLocator"]
    if "verbatim_quote" not in locator["properties"]:
        raise ValueError("EvidenceLocator.verbatim_quote schema path drifted")
    if "verbatim_quote" not in locator["required"]:
        raise ValueError("Parent strict schema lost required verbatim_quote key")
    locator["properties"]["verbatim_quote"] = {
        "type": "null",
        "title": "Verbatim Quote",
    }
    return schema


def final_recovery_transport_schema() -> dict[str, Any]:
    return openai_strict_transport_schema(FreshCFinalAdjudication)


def protocol_expected_id(payload: dict[str, Any]) -> str:
    tmp = dict(payload)
    tmp.pop("protocol_id", None)
    tmp.pop("protocol_sha256", None)
    return PROTOCOL_PREFIX + ":" + canonical_json_sha256(tmp)[:20]


def validate_protocol(path: Path) -> dict[str, Any]:
    p = load_object(path)
    if p.get("schema_version") != "sers-fresh-c-c1b2-r1-recovery-protocol-v1":
        raise ValueError("C1B.2-R1 protocol schema mismatch")
    if p.get("stage") != STAGE or p.get("semantics_id") != SEMANTICS_ID:
        raise ValueError("C1B.2-R1 stage/semantics mismatch")
    if p.get("protocol_id") != protocol_expected_id(p):
        raise ValueError("C1B.2-R1 protocol ID mismatch")
    tmp = dict(p)
    stored = tmp.pop("protocol_sha256", None)
    if stored != canonical_json_sha256(tmp):
        raise ValueError("C1B.2-R1 protocol SHA mismatch")

    exact = {
        "parent_protocol_id": EXPECTED_PARENT_PROTOCOL_ID,
        "parent_protocol_sha256": EXPECTED_PARENT_PROTOCOL_SHA256,
        "parent_protocol_freeze_id": EXPECTED_PARENT_FREEZE_ID,
        "parent_protocol_freeze_sha256": EXPECTED_PARENT_FREEZE_SHA256,
        "parent_source_code_commit": EXPECTED_PARENT_SOURCE_COMMIT,
        "recovery_reason": "VERBATIM_QUOTE_GROUNDING_VALIDATION_FAILURE",
        "recovery_schema_adapter_id": RECOVERY_SCHEMA_ADAPTER_ID,
        "paper_recovery_transport_schema_sha256": EXPECTED_PAPER_RECOVERY_SCHEMA_SHA256,
        "final_recovery_transport_schema_sha256": EXPECTED_FINAL_RECOVERY_SCHEMA_SHA256,
        "raw_reviewer_models_changed": False,
        "scientific_system_prompts_changed": False,
        "scientific_target_boundaries_changed": False,
        "relation_label_vocabulary_changed": False,
        "verdict_lattice_changed": False,
        "verbatim_quote_evidence_enabled": False,
        "verbatim_quote_transport_value": None,
        "paper_review_order": list(range(1, 26)),
        "paper_review_calls": 25,
        "final_adjudication_calls": 1,
        "maximum_recovery_llm_calls": 26,
        "maximum_recovery_network_calls": 26,
        "reuse_failed_parent_response_allowed": False,
        "external_literature_lookup_allowed": False,
        "hypothesis_rewrite_allowed": False,
        "hypothesis_upgrade_allowed": False,
        "h2_resurrection_allowed": False,
        "same_recovery_epoch_rerun_allowed_after_start": False,
        "failure_restores_freshness": False,
        "failure_authorizes_tuning_on_fresh_c": False,
        "recovery_result_may_claim_new_fresh_reserve": False,
        "operator_confirmation_required": True,
        "automatic_post_recovery_transition_allowed": False,
        "stop_after_result_freeze": True,
    }
    for key, expected in exact.items():
        if p.get(key) != expected:
            raise ValueError(f"C1B.2-R1 protocol field drifted: {key}")
    if canonical_json_sha256(paper_recovery_transport_schema()) != EXPECTED_PAPER_RECOVERY_SCHEMA_SHA256:
        raise ValueError("C1B.2-R1 paper recovery schema drifted")
    if canonical_json_sha256(final_recovery_transport_schema()) != EXPECTED_FINAL_RECOVERY_SCHEMA_SHA256:
        raise ValueError("C1B.2-R1 final recovery schema drifted")
    return p


def validate_runtime_env() -> dict[str, Any]:
    base_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    if base_url != EXPECTED_BASE_URL:
        raise RuntimeError("C1B.2-R1 OPENAI_BASE_URL drifted")
    model = os.getenv("FRESH_C_C1B_REVIEWER_MODEL", "").strip()
    if model != EXPECTED_MODEL:
        raise RuntimeError("C1B.2-R1 reviewer model drifted")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not present")
    return {"base_url": base_url, "reviewer_model": model, "credential_present": True}


def validate_parent_failure_state(root: Path) -> dict[str, Any]:
    parent = validate_parent_protocol(
        root / "dac_her/sers_fresh_c_c1b2_scientific_protocol_v1.json"
    )
    if parent["protocol_id"] != EXPECTED_PARENT_PROTOCOL_ID:
        raise ValueError("Parent protocol ID drifted")
    if parent["protocol_sha256"] != EXPECTED_PARENT_PROTOCOL_SHA256:
        raise ValueError("Parent protocol SHA drifted")

    freeze = load_object(root / PARENT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json")
    if freeze.get("freeze_id") != EXPECTED_PARENT_FREEZE_ID:
        raise ValueError("Parent freeze ID drifted")
    if freeze.get("manifest_sha256") != EXPECTED_PARENT_FREEZE_SHA256:
        raise ValueError("Parent freeze SHA drifted")
    if freeze.get("source_code_commit") != EXPECTED_PARENT_SOURCE_COMMIT:
        raise ValueError("Parent source commit drifted")

    run_dir = root / PARENT_RUN_DIR
    started = load_object(run_dir / "C1B2_SCIENTIFIC_READ_STARTED.json")
    failed = load_object(run_dir / "C1B2_SCIENTIFIC_ADJUDICATION_FAILED.json")

    if started.get("protocol_id") != EXPECTED_PARENT_PROTOCOL_ID:
        raise ValueError("Parent start-marker protocol drifted")
    if started.get("same_epoch_rerun_allowed") is not False:
        raise ValueError("Parent failed epoch incorrectly permits rerun")
    if failed.get("error_type") != EXPECTED_FAILURE_TYPE:
        raise ValueError("Recovery authorized only for frozen quote-grounding ValueError")
    if EXPECTED_FAILURE_FRAGMENT not in failed.get("error_summary", ""):
        raise ValueError("Parent failure reason is not quote-grounding validation")
    if failed.get("completed_reserve_indexes") != []:
        raise ValueError("Recovery requires zero completed parent reviews")
    if failed.get("completed_paper_reviews") != 0:
        raise ValueError("Recovery requires zero completed parent reviews")
    if failed.get("scientific_llm_call_attempts") != 1:
        raise ValueError("Parent LLM attempt count drifted")
    if failed.get("scientific_network_call_attempts") != 1:
        raise ValueError("Parent network attempt count drifted")
    if failed.get("fresh_c_scientific_text_read_performed") is not True:
        raise ValueError("Parent failure lost scientific-read state")
    if failed.get("same_epoch_rerun_allowed") is not False:
        raise ValueError("Parent failure incorrectly permits same-epoch rerun")
    if failed.get("failure_restores_freshness") is not False:
        raise ValueError("Parent failure cannot restore freshness")
    if failed.get("failure_authorizes_tuning_on_fresh_c") is not False:
        raise ValueError("Parent failure cannot authorize scientific tuning")

    for forbidden in (
        "run_manifest.json",
        "C1B2_SCIENTIFIC_ADJUDICATION_COMPLETE.json",
        "final_adjudication.json",
    ):
        if (run_dir / forbidden).exists():
            raise ValueError(f"Failed parent unexpectedly contains {forbidden}")
    review_dir = run_dir / "paper_reviews"
    if review_dir.exists() and any(review_dir.glob("*.json")):
        raise ValueError("Failed parent unexpectedly persisted a paper review")

    validate_frozen_lineage(root)
    records = validate_corpus_metadata(root, parse_pages=False)
    if len(records) != 25:
        raise ValueError("Recovery source set is not exact frozen 25")
    return {"started": started, "failed": failed, "records": records}


def validate_schema_qualification(q: dict[str, Any]) -> None:
    if q.get("schema_version") != "sers-fresh-c-c1b2-r1-schema-qualification-v1":
        raise ValueError("Recovery qualification schema mismatch")
    if q.get("recovery_schema_adapter_id") != RECOVERY_SCHEMA_ADAPTER_ID:
        raise ValueError("Recovery qualification adapter drifted")
    if q.get("paper_recovery_transport_schema_sha256") != EXPECTED_PAPER_RECOVERY_SCHEMA_SHA256:
        raise ValueError("Recovery qualification paper-schema SHA drifted")
    if q.get("paper_schema_passed") is not True:
        raise ValueError("Recovery paper schema did not pass")
    if q.get("verbatim_quote_returned_null") is not True:
        raise ValueError("Recovery quote-null constraint did not pass")
    if q.get("requested_model") != EXPECTED_MODEL or q.get("served_model") != EXPECTED_MODEL:
        raise ValueError("Recovery qualification model binding drifted")
    if q.get("network_calls") != 1 or q.get("llm_calls") != 1:
        raise ValueError("Recovery qualification call counts drifted")
    for key in (
        "fresh_c_scientific_text_used",
        "scientific_hypothesis_text_used",
        "scientific_adjudication_performed",
    ):
        if q.get(key) is not False:
            raise ValueError(f"Recovery qualification safety drifted: {key}")
