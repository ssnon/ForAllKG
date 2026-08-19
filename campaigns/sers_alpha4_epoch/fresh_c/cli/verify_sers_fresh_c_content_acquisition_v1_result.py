from __future__ import annotations

import json
import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_content_acquisition_v1 import (
    DEFAULT_RUN_DIR,
    load_json_object,
    validate_upstream_v24,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_content_acquisition_v1_protocol_freeze import (
    main as verify_protocol_freeze,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_protocol_freeze()
    validate_upstream_v24(root)
    run_dir = root / DEFAULT_RUN_DIR
    if (run_dir / "CONTENT_ACQUISITION_FAILED.json").exists():
        raise RuntimeError("C0.1D failed epoch exists; success verification forbidden.")

    manifest = load_json_object(run_dir / "run_manifest.json")
    selected = load_json_object(run_dir / "selected_reserve_c.json")
    seal = load_json_object(run_dir / "content_seal.json")
    complete = load_json_object(run_dir / "CONTENT_ACQUISITION_COMPLETE.json")

    if manifest.get("selected_verified_pdf_count") != 25:
        raise ValueError("C0.1D selected count is not 25.")
    if selected.get("selected_count") != 25:
        raise ValueError("C0.1D selected manifest count is not 25.")
    if seal.get("selected_count") != 25:
        raise ValueError("C0.1D seal count is not 25.")

    records = selected.get("records") or []
    if len(records) != 25:
        raise ValueError("C0.1D selected records length drifted.")
    blind_ranks = [row["blind_rank"] for row in records]
    if blind_ranks != sorted(blind_ranks):
        raise ValueError("C0.1D selected blind ranks are not ordered.")
    if len({row["canonical_id"] for row in records}) != 25:
        raise ValueError("C0.1D selected identities are not unique.")

    for index, row in enumerate(records, start=1):
        if row.get("reserve_index") != index:
            raise ValueError("C0.1D reserve index drifted.")
        path = root / row["local_path"]
        if not path.exists():
            raise FileNotFoundError(path)
        if sha256_file(path) != row["artifact_sha256"]:
            raise ValueError("C0.1D sealed PDF SHA drifted.")
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError("C0.1D sealed artifact lost PDF magic.")

    for field in (
        "manual_candidate_replacement_performed",
        "hypothesis_aware_selection_performed",
        "scientific_metadata_inspection_performed",
        "pdf_text_extraction_performed",
        "semantic_read_performed",
        "paywall_bypass_attempted",
        "positive_evidence_promotion_performed",
        "fresh_reserve_c_consumed",
        "automatic_c1_transition_authorized",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"C0.1D safety field drifted: {field}")

    for filename, field in (
        ("selected_reserve_c.json", "selected_reserve_c_file_sha256"),
        ("content_seal.json", "content_seal_file_sha256"),
        ("identity_attempts.jsonl", "identity_attempts_file_sha256"),
    ):
        if sha256_file(run_dir / filename) != manifest[field]:
            raise ValueError(f"C0.1D artifact SHA drifted: {filename}")

    if complete["run_id"] != manifest["run_id"]:
        raise ValueError("C0.1D COMPLETE run ID mismatch.")
    if complete["run_sha256"] != manifest["run_sha256"]:
        raise ValueError("C0.1D COMPLETE run SHA mismatch.")

    print("Fresh-C C0.1D blind OA content-acquisition result verifier")
    print(f"Run ID: {manifest['run_id']}")
    print(f"Run SHA256: {manifest['run_sha256']}")
    print(f"Attempted blind identities: {manifest['attempted_identity_count']}")
    print("Selected verified OA PDFs: 25")
    print(f"Content seal SHA256: {manifest['content_seal_sha256']}")
    print("Scientific metadata inspection performed: False")
    print("PDF text extraction performed: False")
    print("Semantic read performed: False")
    print("Fresh Reserve C consumed: False")
    print("LLM calls: 0")
    print("Automatic C1 transition authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
