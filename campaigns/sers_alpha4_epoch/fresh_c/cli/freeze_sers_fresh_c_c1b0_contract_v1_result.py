from __future__ import annotations

import json
import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b0_contract_v1 import (
    DEFAULT_RESULT_FREEZE_DIR,
    DEFAULT_RUN_DIR,
    canonical_json_sha256,
    load_object,
    sha256_file,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_c1b0_contract_v1_result import main as verify_result


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_result()
    contract_path = root / DEFAULT_RUN_DIR / "input_contract.json"
    contract = load_object(contract_path)
    body = {
        "schema_version": "sers-fresh-c-c1b0-contract-result-freeze-v1",
        "contract_id": contract["contract_id"],
        "contract_sha256": contract["contract_sha256"],
        "contract_file_sha256": sha256_file(contract_path),
        "r2_report_id": contract["r2"]["r2_report_id"],
        "r2_report_sha256": contract["r2"]["r2_report_sha256"],
        "r2_freeze_id": contract["r2"]["r2_freeze_id"],
        "c1ar1_corpus_sha256": (
            "cffb7eab1465258b61ea28d64b1a703cb5a2b0cb940da0342bc7c1929db89e19"
        ),
        "source_identity_count": 25,
        "scientific_target_hypothesis_ids": contract["r2"][
            "scientific_target_hypothesis_ids"
        ],
        "terminal_rejected_hypothesis_ids": contract["r2"][
            "terminal_rejected_hypothesis_ids"
        ],
        "primary_remaining_candidate_hypothesis_id": contract["r2"][
            "primary_remaining_candidate_hypothesis_id"
        ],
        "fresh_c_scientific_text_semantic_read_performed": False,
        "scientific_adjudication_performed": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "c1b1_authorized": False,
        "stop": True,
    }
    ident = canonical_json_sha256(body)
    body["freeze_id"] = (
        "sers_fresh_c_c1b0_contract_result_freeze_v1:" + ident[:20]
    )
    tmp = dict(body)
    body["manifest_sha256"] = canonical_json_sha256(tmp)
    output = root / DEFAULT_RESULT_FREEZE_DIR
    if output.exists():
        raise FileExistsError("C1B.0 result freeze directory exists.")
    _atomic(output / "freeze_manifest.json", body)
    _atomic(output / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "c1b1_authorized": False,
        "fresh_c_scientific_text_semantic_read_performed": False,
        "scientific_adjudication_performed": False,
        "stop": True,
    })
    print("Fresh-C C1B.0 input-contract result freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Contract ID: {body['contract_id']}")
    print("Exact materialized corpus: 25/25")
    print("Fresh-C scientific text semantic read performed: False")
    print("Scientific adjudication performed: False")
    print("C1B.1 authorized: False")
    print("Network calls during freeze: 0")
    print("LLM calls during freeze: 0")
    print("STOP: True")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
