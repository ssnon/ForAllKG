import subprocess
from pathlib import Path
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b1_r1_transport_qualification_v1 import (
    DEFAULT_RUN_DIR,
    canonical_json_sha256,
    load_object,
)

def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

def main():
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    result = load_object(root / DEFAULT_RUN_DIR / "qualification_result.json")
    complete = load_object(root / DEFAULT_RUN_DIR / "C1B1_R1_QUALIFICATION_COMPLETE.json")
    tmp = dict(result)
    result_id = tmp.pop("result_id")
    stored = tmp.pop("result_sha256")
    if stored != canonical_json_sha256(tmp):
        raise ValueError("Transport qualification result SHA drifted")
    if result_id != (
        "sers_fresh_c_c1b1_r1_transport_qualification_result_v1:" + stored[:20]
    ):
        raise ValueError("Transport qualification result ID drifted")
    if complete["result_id"] != result_id or complete["result_sha256"] != stored:
        raise ValueError("Qualification COMPLETE mismatch")
    for key in (
        "catalog_membership_verified",
        "response_format_parameter_verified_or_advertised",
        "structured_json_schema_call_passed",
    ):
        if result.get(key) is not True:
            raise ValueError(f"Qualification did not verify: {key}")
    if result.get("catalog_metadata_network_calls") != 1:
        raise ValueError("Catalog call count drifted")
    if result.get("synthetic_llm_calls") != 1:
        raise ValueError("Synthetic LLM count drifted")
    if result.get("total_network_calls") != 2:
        raise ValueError("Total network count drifted")
    if result.get("temperature_parameter_sent") is not False:
        raise ValueError("Unsupported temperature parameter was sent")
    if result.get("deterministic_seed") != 0:
        raise ValueError("Deterministic seed drifted")
    if result.get("reasoning_effort") != "medium":
        raise ValueError("Reasoning effort drifted")
    if result.get("reasoning_exclude") is not True:
        raise ValueError("Reasoning exclusion drifted")
    if result.get("provider_only") != ["openai"]:
        raise ValueError("Upstream provider binding drifted")
    if result.get("provider_allow_fallbacks") is not False:
        raise ValueError("Provider fallback policy drifted")
    if result.get("provider_require_parameters") is not True:
        raise ValueError("Provider parameter policy drifted")
    for key in (
        "fresh_c_scientific_text_semantic_read_performed",
        "scientific_hypothesis_text_used",
        "external_literature_used",
        "scientific_adjudication_performed",
        "c1b2_authorized",
    ):
        if result.get(key) is not False:
            raise ValueError(f"Safety field drifted: {key}")
    if result.get("stop") is not True:
        raise ValueError("STOP drifted")
    print("Fresh-C C1B.1-R1 transport qualification result verifier")
    print(f"Result ID: {result_id}")
    print(f"Result SHA256: {stored}")
    print(f"Requested model: {result['requested_model']}")
    print(f"Served model: {result['served_model']}")
    print("Catalog membership verified: True")
    print("Structured JSON-schema call passed: True")
    print("Fresh-C scientific text read: False")
    print("Scientific adjudication: False")
    print("C1B.2 authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
