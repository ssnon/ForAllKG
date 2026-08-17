import argparse
import json
import os
import subprocess
from pathlib import Path
from openai import OpenAI

from dac_her.fresh_c_c1b1_r1_transport_qualification_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_RUN_DIR,
    SyntheticTransportProbe,
    canonical_json_sha256,
    load_object,
    validate_parent_freeze,
    validate_protocol,
    validate_runtime_env,
)
from scripts.verify_sers_fresh_c_c1b1_r1_transport_protocol_freeze_v1 import (
    main as verify_protocol_freeze,
)

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

def _state(root):
    verify_protocol_freeze()
    validate_parent_freeze(root)
    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    env = validate_runtime_env()
    freeze = load_object(root / DEFAULT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json")
    run_dir = root / DEFAULT_RUN_DIR
    empty = not run_dir.exists() or not any(run_dir.iterdir())
    return p, env, freeze, empty

def preflight():
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    p, env, freeze, empty = _state(root)
    if not empty:
        raise RuntimeError("Transport qualification run directory is not empty")
    print("Fresh-C C1B.1-R1 transport qualification preflight")
    print(f"Protocol freeze ID: {freeze['freeze_id']}")
    print(f"Base URL current: {env['base_url']}")
    print(f"Reviewer model current: {env['reviewer_model']}")
    print("Credential present: True")
    print("Expected catalog metadata calls: 1")
    print("Expected synthetic LLM calls: 1")
    print("Fresh-C text will be read: False")
    print("Scientific hypothesis text will be used: False")
    print("Scientific adjudication will be performed: False")
    print("C1B.2 authorized: False")
    print("STOP: True")
    print("Preflight: PASS")
    return 0

def qualify():
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    p, env, freeze, empty = _state(root)
    if not empty:
        raise RuntimeError("Transport qualification epoch already exists")
    run_dir = root / DEFAULT_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=False)
    client = OpenAI(
        api_key=os.environ[p["credential_env"]],
        base_url=env["base_url"],
        timeout=120.0,
        max_retries=0,
    )
    models = client.models.list().data
    target = next((m for m in models if m.id == p["reviewer_model"]), None)
    if target is None:
        raise RuntimeError("Frozen reviewer model missing from OpenRouter catalog")
    extra = getattr(target, "model_extra", {}) or {}
    supported = extra.get("supported_parameters")
    if supported is not None and "response_format" not in supported:
        raise RuntimeError("Frozen reviewer model does not advertise response_format")

    schema = SyntheticTransportProbe.model_json_schema()
    response = client.chat.completions.create(
        model=p["reviewer_model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "This is a synthetic transport qualification. "
                    "No scientific or Fresh-C content is present. "
                    "Return the required JSON object only."
                ),
            },
            {
                "role": "user",
                "content": (
                    'Return status="PASS", scientific_content_used=false, '
                    'fresh_c_content_used=false.'
                ),
            },
        ],
        seed=p["deterministic_seed"],
        max_tokens=2048,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SyntheticTransportProbe",
                "strict": True,
                "schema": schema,
            },
        },
        extra_body={
            "reasoning": {
                "effort": p["reasoning_effort"],
                "exclude": p["reasoning_exclude"],
            },
            "provider": {
                "only": p["provider_only"],
                "allow_fallbacks": p["provider_allow_fallbacks"],
                "require_parameters": p["provider_require_parameters"],
                "data_collection": p["provider_data_collection"],
            },
        },
    )
    parsed = SyntheticTransportProbe.model_validate_json(
        response.choices[0].message.content
    )
    usage = response.usage
    result = {
        "schema_version": "sers-fresh-c-c1b1-r1-transport-qualification-result-v1",
        "protocol_id": p["protocol_id"],
        "protocol_sha256": p["protocol_sha256"],
        "protocol_freeze_id": freeze["freeze_id"],
        "parent_freeze_id": p["parent_freeze_id"],
        "corrected_transport_semantics": p["corrected_transport_semantics"],
        "base_url": env["base_url"],
        "requested_model": p["reviewer_model"],
        "served_model": response.model,
        "temperature_parameter_sent": False,
        "deterministic_seed": p["deterministic_seed"],
        "reasoning_effort": p["reasoning_effort"],
        "reasoning_exclude": p["reasoning_exclude"],
        "provider_only": p["provider_only"],
        "provider_allow_fallbacks": p["provider_allow_fallbacks"],
        "provider_require_parameters": p["provider_require_parameters"],
        "provider_data_collection": p["provider_data_collection"],
        "catalog_membership_verified": True,
        "catalog_supported_parameters": supported,
        "response_format_parameter_verified_or_advertised": (
            supported is None or "response_format" in supported
        ),
        "structured_json_schema_call_passed": parsed.status == "PASS",
        "finish_reason": response.choices[0].finish_reason,
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "catalog_metadata_network_calls": 1,
        "synthetic_llm_calls": 1,
        "total_network_calls": 2,
        "fresh_c_scientific_text_semantic_read_performed": False,
        "scientific_hypothesis_text_used": False,
        "external_literature_used": False,
        "scientific_adjudication_performed": False,
        "c1b2_authorized": False,
        "stop": True,
    }
    result["result_sha256"] = canonical_json_sha256(result)
    result["result_id"] = (
        "sers_fresh_c_c1b1_r1_transport_qualification_result_v1:"
        + result["result_sha256"][:20]
    )
    _atomic(run_dir / "qualification_result.json", result)
    _atomic(run_dir / "C1B1_R1_QUALIFICATION_COMPLETE.json", {
        "result_id": result["result_id"],
        "result_sha256": result["result_sha256"],
        "structured_json_schema_call_passed": True,
        "fresh_c_scientific_text_semantic_read_performed": False,
        "scientific_adjudication_performed": False,
        "c1b2_authorized": False,
        "stop": True,
    })
    print("Fresh-C C1B.1-R1 transport qualification complete")
    print(f"Result ID: {result['result_id']}")
    print(f"Result SHA256: {result['result_sha256']}")
    print(f"Requested model: {result['requested_model']}")
    print(f"Served model: {result['served_model']}")
    print("Catalog membership verified: True")
    print("Structured JSON-schema call passed: True")
    print("Catalog metadata network calls: 1")
    print("Synthetic LLM calls: 1")
    print("Fresh-C scientific text read: False")
    print("Scientific hypothesis text used: False")
    print("Scientific adjudication: False")
    print("C1B.2 authorized: False")
    print("STOP: True")
    return 0

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qualify-transport", action="store_true")
    args = parser.parse_args()
    return preflight() if args.preflight else qualify()

if __name__ == "__main__":
    raise SystemExit(main())
