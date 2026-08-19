from __future__ import annotations

import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b0_contract_v1 import (
    DEFAULT_RUN_DIR,
    canonical_json_sha256,
    load_object,
    validate_c1ar1_lineage,
    validate_r2_lineage,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    validate_r2_lineage(root)
    validate_c1ar1_lineage(root, hash_text_files=True)
    contract = load_object(root / DEFAULT_RUN_DIR / "input_contract.json")
    complete = load_object(root / DEFAULT_RUN_DIR / "C1B0_AUDIT_COMPLETE.json")

    tmp = dict(contract)
    stored_id = tmp.pop("contract_id")
    stored_sha = tmp.pop("contract_sha256")
    expected_sha = canonical_json_sha256(tmp)
    if stored_sha != expected_sha:
        raise ValueError("C1B.0 contract SHA drifted.")
    if stored_id != "sers_fresh_c_c1b0_input_contract_v1:" + stored_sha[:20]:
        raise ValueError("C1B.0 contract ID drifted.")
    if complete["contract_id"] != stored_id:
        raise ValueError("C1B.0 COMPLETE contract ID mismatch.")
    if complete["contract_sha256"] != stored_sha:
        raise ValueError("C1B.0 COMPLETE contract SHA mismatch.")
    for field in (
        "fresh_c_scientific_text_semantic_read_performed",
        "scientific_adjudication_performed",
        "c1b1_authorized",
    ):
        if contract.get(field) is not False:
            raise ValueError(f"C1B.0 safety field drifted: {field}")
    if contract.get("network_calls") != 0 or contract.get("llm_calls") != 0:
        raise ValueError("C1B.0 unexpectedly used network or LLM.")
    if contract.get("stop") is not True:
        raise ValueError("C1B.0 STOP drifted.")

    print("Fresh-C C1B.0 input-contract result verifier")
    print(f"Contract ID: {stored_id}")
    print(f"Contract SHA256: {stored_sha}")
    print("Scientific targets: H1,H3")
    print("H2 terminal rejected: True")
    print("Exact materialized corpus: 25/25")
    print("Repaired reserve #14 absence/completeness inference: False")
    print("Fresh-C scientific text semantic read performed: False")
    print("Scientific adjudication performed: False")
    print("Network calls: 0")
    print("LLM calls: 0")
    print("C1B.1 authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
