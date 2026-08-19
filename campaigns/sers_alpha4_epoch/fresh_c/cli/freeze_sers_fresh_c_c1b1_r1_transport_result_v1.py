import json
import subprocess
from pathlib import Path
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b1_r1_transport_qualification_v1 import (
    DEFAULT_RESULT_FREEZE_DIR,
    DEFAULT_RUN_DIR,
    canonical_json_sha256,
    load_object,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_c1b1_r1_transport_result_v1 import main as verify_result

def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

def _atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)

def main():
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_result()
    result = load_object(root / DEFAULT_RUN_DIR / "qualification_result.json")
    body = {
        "schema_version": "sers-fresh-c-c1b1-r1-transport-result-freeze-v1",
        "result_id": result["result_id"],
        "result_sha256": result["result_sha256"],
        "parent_freeze_id": result["parent_freeze_id"],
        "corrected_transport_semantics": result["corrected_transport_semantics"],
        "base_url": result["base_url"],
        "requested_model": result["requested_model"],
        "served_model": result["served_model"],
        "temperature_parameter_sent": result["temperature_parameter_sent"],
        "deterministic_seed": result["deterministic_seed"],
        "reasoning_effort": result["reasoning_effort"],
        "reasoning_exclude": result["reasoning_exclude"],
        "provider_only": result["provider_only"],
        "provider_allow_fallbacks": result["provider_allow_fallbacks"],
        "provider_require_parameters": result["provider_require_parameters"],
        "provider_data_collection": result["provider_data_collection"],
        "catalog_membership_verified": True,
        "structured_json_schema_call_passed": True,
        "catalog_metadata_network_calls": 1,
        "synthetic_llm_calls": 1,
        "fresh_c_scientific_text_semantic_read_performed": False,
        "scientific_adjudication_performed": False,
        "c1b2_authorized": False,
        "stop": True,
    }
    ident = canonical_json_sha256(body)
    body["freeze_id"] = (
        "sers_fresh_c_c1b1_r1_transport_result_freeze_v1:" + ident[:20]
    )
    tmp = dict(body)
    body["manifest_sha256"] = canonical_json_sha256(tmp)
    output = root / DEFAULT_RESULT_FREEZE_DIR
    if output.exists():
        raise FileExistsError("Transport result freeze directory exists")
    _atomic(output / "freeze_manifest.json", body)
    _atomic(output / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "c1b2_authorized": False,
        "fresh_c_scientific_text_semantic_read_performed": False,
        "stop": True,
    })
    print("Fresh-C C1B.1-R1 transport qualification result freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Requested model: {body['requested_model']}")
    print(f"Served model: {body['served_model']}")
    print("Structured JSON-schema call passed: True")
    print("Fresh-C scientific text read: False")
    print("Scientific adjudication: False")
    print("C1B.2 authorized: False")
    print("STOP: True")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
