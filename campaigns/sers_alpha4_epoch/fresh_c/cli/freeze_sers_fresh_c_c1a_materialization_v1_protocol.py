from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1a_materialization_v1 import (
    C01D_CONTENT_SEAL_PATH,
    C01D_RESULT_FREEZE_MANIFEST,
    C01D_RUN_MANIFEST_PATH,
    C01D_SELECTED_PATH,
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    load_and_validate_protocol,
    validate_c01d_closed_frozen,
)

CRITICAL_COMPONENTS = (
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_c1a_materialization_v1.py",
    "dac_her/sers_fresh_c_c1a_materialization_v1_protocol.json",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1a_materialization_v1_protocol.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/freeze_sers_fresh_c_c1a_materialization_v1_protocol.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1a_materialization_v1_protocol_freeze.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/run_sers_fresh_c_c1a_materialization_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1a_materialization_v1_result.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/freeze_sers_fresh_c_c1a_materialization_v1_result.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1a_materialization_v1_result_freeze.py",
    "tests/test_sers_fresh_c_c1a_materialization_v1.py",
    "requirements.txt",
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
        ["git", "show", f"{commit}:{relative.as_posix()}"], cwd=root
    )
    expected = hashlib.sha256(committed).hexdigest()
    if sha256_file(root / relative) != expected:
        raise RuntimeError(f"C1A frozen parent artifact drifted: {relative}")
    return expected


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if subprocess.run(["git", "diff", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Tracked worktree dirty; refuse C1A protocol freeze.")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Index dirty; refuse C1A protocol freeze.")

    upstream = validate_c01d_closed_frozen(root)
    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    source_commit = _git(root, "rev-parse", "HEAD")

    parent_hashes = {}
    for relative in (
        C01D_SELECTED_PATH,
        C01D_RUN_MANIFEST_PATH,
        C01D_CONTENT_SEAL_PATH,
        C01D_RESULT_FREEZE_MANIFEST,
    ):
        parent_hashes[str(relative)] = _tracked_sha(root, relative, source_commit)

    component_hashes = {}
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root
        )
        expected = hashlib.sha256(committed).hexdigest()
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"C1A component drifted: {relative}")
        component_hashes[relative] = expected

    body = {
        "schema_version": "sers-fresh-c-c1a-materialization-protocol-freeze-v1",
        "protocol_id": p.protocol_id,
        "protocol_sha256": p.protocol_sha256,
        "source_code_commit": source_commit,
        "c01d_result_freeze_id": p.c01d_result_freeze_id,
        "c01d_result_freeze_sha256": p.c01d_result_freeze_sha256,
        "c01d_content_seal_sha256": p.c01d_content_seal_sha256,
        "parent_artifact_sha256": parent_hashes,
        "critical_component_sha256": component_hashes,
        "selected_pdf_count": 25,
        "consumption_marker_before_first_text_extraction": True,
        "consumption_irreversible": True,
        "materializer": p.materializer,
        "pdftext_version": p.pdftext_version,
        "pypdfium2_version": p.pypdfium2_version,
        "network_allowed_during_materialization": False,
        "scientific_reviewer_read_performed": False,
        "scientific_adjudication_performed": False,
        "fresh_reserve_c_consumed": False,
        "live_materialization_ready": True,
        "live_materialization_authorized": False,
        "live_materialization_started": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "automatic_c1b_transition_authorized": False,
        "stop": True,
    }
    ident = sha256_json(body)
    body["freeze_id"] = (
        "sers_fresh_c_c1a_materialization_protocol_freeze_v1:" + ident[:20]
    )
    body["manifest_sha256"] = _payload_sha(body, "manifest_sha256")

    output = root / DEFAULT_PROTOCOL_FREEZE_DIR
    if output.exists():
        raise FileExistsError("C1A protocol freeze directory already exists.")
    _atomic(output / "freeze_manifest.json", body)
    _atomic(output / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "live_materialization_ready": True,
        "live_materialization_authorized": False,
        "fresh_reserve_c_consumed": False,
        "stop": True,
    })

    print("Fresh-C C1A irreversible materialization protocol freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("Selected sealed PDFs: 25")
    print("Consumption marker before first text extraction: True")
    print("Materializer: pdftext 0.6.3 + pypdfium2 4.30.0")
    print("Network allowed during materialization: False")
    print("Scientific reviewer read performed: False")
    print("Scientific adjudication performed: False")
    print("Fresh Reserve C consumed: False")
    print("Network calls during freeze: 0")
    print("LLM calls during freeze: 0")
    print("STOP: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
