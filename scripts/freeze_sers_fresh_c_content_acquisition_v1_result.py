from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from dac_her.fresh_c_acquisition import sha256_file, sha256_json
from dac_her.fresh_c_content_acquisition_v1 import (
    DEFAULT_RESULT_FREEZE_DIR,
    DEFAULT_RUN_DIR,
    load_json_object,
)
from scripts.verify_sers_fresh_c_content_acquisition_v1_result import (
    main as verify_result,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _payload_sha(payload, field):
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def _atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULT_FREEZE_DIR,
    )
    args = parser.parse_args()
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_result()

    run_dir = root / DEFAULT_RUN_DIR
    manifest = load_json_object(run_dir / "run_manifest.json")
    selected = load_json_object(run_dir / "selected_reserve_c.json")
    seal = load_json_object(run_dir / "content_seal.json")

    pdf_hashes = {
        row["canonical_id"]: row["artifact_sha256"]
        for row in selected["records"]
    }
    body = {
        "schema_version": "sers-fresh-c-content-acquisition-result-freeze-v1",
        "source_run_id": manifest["run_id"],
        "source_run_sha256": manifest["run_sha256"],
        "selected_verified_pdf_count": 25,
        "content_seal_sha256": seal["content_seal_sha256"],
        "selected_pdf_sha256": pdf_hashes,
        "reserve_c_identity_selection_finalized": True,
        "reserve_c_content_sealed": True,
        "semantic_read_performed": False,
        "pdf_text_extraction_performed": False,
        "fresh_reserve_c_consumed": False,
        "c1_authorized": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "stop": True,
    }
    ident = sha256_json(body)
    body["freeze_id"] = (
        "sers_fresh_c_content_acquisition_result_freeze_v1:"
        + ident[:20]
    )
    body["manifest_sha256"] = _payload_sha(body, "manifest_sha256")

    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if output.exists():
        raise FileExistsError("C0.1D result freeze directory already exists.")
    _atomic(output / "freeze_manifest.json", body)
    _atomic(output / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "reserve_c_identity_selection_finalized": True,
        "reserve_c_content_sealed": True,
        "fresh_reserve_c_consumed": False,
        "c1_authorized": False,
        "stop": True,
    })

    print("Fresh-C C0.1D result freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source run ID: {body['source_run_id']}")
    print("Selected verified OA PDFs: 25")
    print(f"Content seal SHA256: {body['content_seal_sha256']}")
    print("Reserve-C identity selection finalized: True")
    print("Reserve-C content sealed: True")
    print("Semantic read performed: False")
    print("Fresh Reserve C consumed: False")
    print("C1 authorized: False")
    print("STOP: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
