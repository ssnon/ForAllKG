import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from typing import Literal

STAGE = "C1B.1-R1"
SEMANTICS_ID = "sers_fresh_c_c1b1_r1_transport_qualification_v1"
PROTOCOL_PREFIX = "sers_fresh_c_c1b1_r1_transport_protocol_v1"

PARENT_FREEZE = Path(
    "evaluation/sers_fresh_c/"
    "c1b1_reviewer_protocol_freeze_v1/freeze_manifest.json"
)
EXPECTED_PARENT_FREEZE_ID = (
    "sers_fresh_c_c1b1_reviewer_protocol_freeze_v1:"
    "30fc8ea1d36ec3503c21"
)
EXPECTED_PARENT_FREEZE_SHA256 = (
    "31ba6d47570935bcb9809bebe8b727b65bd4aab5831106ad96f5eb8b3cb650be"
)
EXPECTED_PARENT_SOURCE_COMMIT = "ba102a604fd18799659aa06b839ea4f25a268459"
EXPECTED_PARENT_PROTOCOL_ID = (
    "sers_fresh_c_c1b1_reviewer_protocol_v1:b71d51b3ceb72234681f"
)
EXPECTED_PARENT_PROTOCOL_SHA256 = (
    "11778cdf4926cb4481df54713b7b56059edaf381c6151cabe6b97569305a1ba4"
)

EXPECTED_BASE_URL = "https://openrouter.ai/api/v1"
EXPECTED_MODEL = "openai/gpt-5.6-luna"
CREDENTIAL_ENV = "OPENAI_API_KEY"
BASE_URL_ENV = "OPENAI_BASE_URL"
MODEL_ENV = "FRESH_C_C1B_REVIEWER_MODEL"

DEFAULT_PROTOCOL_PATH = Path(
    "dac_her/sers_fresh_c_c1b1_r1_transport_protocol_v1.json"
)
DEFAULT_PROTOCOL_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c1b1_r1_transport_protocol_freeze_v1"
)
DEFAULT_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c1b1_r1_transport_qualification_run_v1"
)
DEFAULT_RESULT_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c1b1_r1_transport_result_freeze_v1"
)

def canonical_json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

def load_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value

def normalized_base_url(value):
    return value.rstrip("/")

def validate_parent_freeze(root):
    p = load_object(Path(root) / PARENT_FREEZE)
    expected = {
        "freeze_id": EXPECTED_PARENT_FREEZE_ID,
        "manifest_sha256": EXPECTED_PARENT_FREEZE_SHA256,
        "protocol_id": EXPECTED_PARENT_PROTOCOL_ID,
        "protocol_sha256": EXPECTED_PARENT_PROTOCOL_SHA256,
        "source_code_commit": EXPECTED_PARENT_SOURCE_COMMIT,
        "reviewer_model": EXPECTED_MODEL,
        "reviewer_backend": "openai_chat_completions_json_schema_v1",
        "fresh_c_scientific_text_semantic_read_performed": False,
        "scientific_adjudication_performed": False,
        "c1b2_authorized": False,
        "stop": True,
    }
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError(f"Parent C1B.1 freeze drifted: {key}")
    return p

def protocol_expected_id(payload):
    tmp = dict(payload)
    tmp.pop("protocol_id", None)
    tmp.pop("protocol_sha256", None)
    return PROTOCOL_PREFIX + ":" + canonical_json_sha256(tmp)[:20]

def validate_protocol(path):
    p = load_object(path)
    if p.get("schema_version") != "sers-fresh-c-c1b1-r1-transport-protocol-v1":
        raise ValueError("C1B.1-R1 protocol schema mismatch")
    if p.get("stage") != STAGE or p.get("semantics_id") != SEMANTICS_ID:
        raise ValueError("C1B.1-R1 stage/semantics mismatch")
    if p.get("protocol_id") != protocol_expected_id(p):
        raise ValueError("C1B.1-R1 protocol ID mismatch")
    tmp = dict(p)
    stored = tmp.pop("protocol_sha256")
    if stored != canonical_json_sha256(tmp):
        raise ValueError("C1B.1-R1 protocol SHA mismatch")

    exact = {
        "parent_freeze_id": EXPECTED_PARENT_FREEZE_ID,
        "parent_freeze_sha256": EXPECTED_PARENT_FREEZE_SHA256,
        "parent_protocol_id": EXPECTED_PARENT_PROTOCOL_ID,
        "parent_protocol_sha256": EXPECTED_PARENT_PROTOCOL_SHA256,
        "parent_model_binding_preserved": True,
        "parent_prompt_and_schema_semantics_preserved": True,
        "corrected_transport_semantics": "openrouter_openai_compatible_chat_completions_json_schema_v1",
        "base_url": EXPECTED_BASE_URL,
        "base_url_env": BASE_URL_ENV,
        "credential_env": CREDENTIAL_ENV,
        "reviewer_model": EXPECTED_MODEL,
        "reviewer_model_env": MODEL_ENV,
        "catalog_membership_required": True,
        "response_format_parameter_required": True,
        "temperature_parameter_policy": "OMIT_UNSUPPORTED_PARAMETER",
        "deterministic_seed": 0,
        "reasoning_effort": "medium",
        "reasoning_exclude": True,
        "provider_only": ["openai"],
        "provider_allow_fallbacks": False,
        "provider_require_parameters": True,
        "provider_data_collection": "deny",
        "synthetic_structured_output_qualification_calls": 1,
        "catalog_metadata_network_calls": 1,
        "maximum_network_calls_during_qualification": 2,
        "maximum_llm_calls_during_qualification": 1,
        "fresh_c_text_allowed_in_qualification": False,
        "scientific_hypothesis_text_allowed_in_qualification": False,
        "external_literature_allowed_in_qualification": False,
        "scientific_adjudication_allowed_in_qualification": False,
        "automatic_c1b2_transition_allowed": False,
        "stop_after_result_freeze": True,
    }
    for key, value in exact.items():
        if p.get(key) != value:
            raise ValueError(f"C1B.1-R1 protocol drifted: {key}")
    return p

def validate_runtime_env():
    base_url = normalized_base_url(os.getenv(BASE_URL_ENV, ""))
    if base_url != EXPECTED_BASE_URL:
        raise RuntimeError(
            f"{BASE_URL_ENV} must equal {EXPECTED_BASE_URL!r}; got {base_url!r}"
        )
    model = os.getenv(MODEL_ENV, "").strip()
    if model != EXPECTED_MODEL:
        raise RuntimeError(
            f"{MODEL_ENV} must equal {EXPECTED_MODEL!r}; got {model!r}"
        )
    if not os.getenv(CREDENTIAL_ENV):
        raise RuntimeError(f"{CREDENTIAL_ENV} is not present")
    return {
        "base_url": base_url,
        "reviewer_model": model,
        "credential_present": True,
    }

class SyntheticTransportProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["PASS"]
    scientific_content_used: Literal[False]
    fresh_c_content_used: Literal[False]
