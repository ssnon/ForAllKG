from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from dac_her.fresh_c_acquisition import sha256_file, sha256_json
from dac_her.fresh_c_content_acquisition_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    V24_COMPLETE_PATH,
    V24_DIAGNOSTICS_PATH,
    V24_LOCATOR_PATH,
    V24_QUEUE_PATH,
    V24_RUN_MANIFEST_PATH,
    load_and_validate_protocol,
    validate_upstream_v24,
)

CRITICAL_COMPONENTS = (
    "dac_her/fresh_c_content_acquisition_v1.py",
    "dac_her/sers_fresh_c_content_acquisition_v1_protocol.json",
    "scripts/verify_sers_fresh_c_content_acquisition_v1_protocol.py",
    "scripts/freeze_sers_fresh_c_content_acquisition_v1_protocol.py",
    "scripts/verify_sers_fresh_c_content_acquisition_v1_protocol_freeze.py",
    "scripts/run_sers_fresh_c_content_acquisition_v1.py",
    "scripts/verify_sers_fresh_c_content_acquisition_v1_result.py",
    "scripts/freeze_sers_fresh_c_content_acquisition_v1_result.py",
    "scripts/verify_sers_fresh_c_content_acquisition_v1_result_freeze.py",
    "tests/test_sers_fresh_c_content_acquisition_v1.py",
    "dac_her/corpus_acquisition/access_contracts.py",
    "dac_her/corpus_acquisition/oa_resolution.py",
    "dac_her/corpus_acquisition/openalex_access.py",
    "dac_her/corpus_acquisition/access_priority.py",
    "dac_her/corpus_acquisition/artifact_acquisition.py",
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


def _tracked_sha(root: Path, relative: Path, commit: str) -> str:
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative.as_posix()}"],
        cwd=root,
    )
    expected = hashlib.sha256(committed).hexdigest()
    if sha256_file(root / relative) != expected:
        raise RuntimeError(f"Upstream artifact drifted: {relative}")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROTOCOL_FREEZE_DIR,
    )
    args = parser.parse_args()

    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if subprocess.run(["git", "diff", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Tracked worktree dirty; refuse C0.1D protocol freeze.")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Index dirty; refuse C0.1D protocol freeze.")

    validate_upstream_v24(root)
    p = load_and_validate_protocol(
        args.protocol if args.protocol.is_absolute() else root / args.protocol
    )
    source_commit = _git(root, "rev-parse", "HEAD")

    upstream_hashes = {}
    for relative in (
        V24_RUN_MANIFEST_PATH,
        V24_QUEUE_PATH,
        V24_LOCATOR_PATH,
        V24_DIAGNOSTICS_PATH,
        V24_COMPLETE_PATH,
    ):
        upstream_hashes[str(relative)] = _tracked_sha(
            root, relative, source_commit
        )

    component_hashes = {}
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"],
            cwd=root,
        )
        expected = hashlib.sha256(committed).hexdigest()
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"C0.1D component drifted: {relative}")
        component_hashes[relative] = expected

    body = {
        "schema_version": "sers-fresh-c-content-acquisition-protocol-freeze-v1",
        "protocol_id": p.protocol_id,
        "protocol_sha256": p.protocol_sha256,
        "source_code_commit": source_commit,
        "upstream_v24_run_id": p.upstream_v24_run_id,
        "upstream_v24_run_sha256": p.upstream_v24_run_sha256,
        "upstream_artifact_sha256": upstream_hashes,
        "critical_component_sha256": component_hashes,
        "blind_queue_count": p.upstream_blind_queue_count,
        "target_verified_pdf_count": p.target_successful_pdf_count,
        "live_acquisition_ready": True,
        "live_acquisition_authorized": False,
        "live_acquisition_started": False,
        "scientific_metadata_inspection_allowed": False,
        "pdf_text_extraction_allowed": False,
        "paywall_bypass_allowed": False,
        "fresh_reserve_c_consumed": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "automatic_c1_transition_authorized": False,
        "stop": True,
    }
    ident = sha256_json(body)
    body["freeze_id"] = (
        "sers_fresh_c_content_acquisition_protocol_freeze_v1:"
        + ident[:20]
    )
    body["manifest_sha256"] = _payload_sha(body, "manifest_sha256")

    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if output.exists():
        raise FileExistsError("C0.1D protocol freeze dir already exists.")
    _atomic(output / "freeze_manifest.json", body)
    _atomic(output / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "live_acquisition_ready": True,
        "live_acquisition_authorized": False,
        "fresh_reserve_c_consumed": False,
        "stop": True,
    })

    print("Fresh-C C0.1D blind OA acquisition protocol freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print(f"Blind queue count: {p.upstream_blind_queue_count}")
    print(f"Target verified PDFs: {p.target_successful_pdf_count}")
    print("Scientific metadata inspection allowed: False")
    print("PDF text extraction allowed: False")
    print("Paywall bypass allowed: False")
    print("Network calls during freeze: 0")
    print("Fresh Reserve C consumed: False")
    print("STOP: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
