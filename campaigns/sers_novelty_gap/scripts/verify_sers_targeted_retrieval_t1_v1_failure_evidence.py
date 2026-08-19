from __future__ import annotations

import hashlib
import json
from pathlib import Path

from campaigns.sers_novelty_gap.sers_targeted_retrieval_t1_live_recovery_v2 import (
    ROOT,
    V1_FAILURE_MANIFEST,
    V1_RUN_ROOT,
    validate_v1_failure_evidence,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def main() -> int:
    if not V1_FAILURE_MANIFEST.is_file():
        print("T1 v1 failure evidence verification: FAIL")
        print(" - failure manifest missing")
        return 2

    manifest = json.loads(
        V1_FAILURE_MANIFEST.read_text(encoding="utf-8")
    )
    body = dict(manifest)
    observed_id = body.pop("failure_freeze_id", None)
    observed_sha = body.pop("manifest_sha256", None)
    expected_sha = _sha256_json(body)
    expected_id = (
        "sers_targeted_retrieval_t1_v1_failure_freeze:"
        + expected_sha[:20]
    )
    issues: list[str] = []
    if observed_sha != expected_sha:
        issues.append("failure manifest SHA mismatch")
    if observed_id != expected_id:
        issues.append("failure freeze ID mismatch")

    for rel, expected in manifest.get("files", {}).items():
        path = V1_RUN_ROOT / rel
        if not path.is_file():
            issues.append(f"missing v1 evidence file: {rel}")
        elif _sha256(path) != expected:
            issues.append(f"v1 evidence SHA mismatch: {rel}")

    try:
        diagnosis = validate_v1_failure_evidence()
        if (
            diagnosis["recovered_audit_sha256"]
            != manifest.get("recovered_gap1_audit_sha256")
        ):
            issues.append("recovered gap_01 audit SHA mismatch")
        if diagnosis["structural_pass"] is not True:
            issues.append("recovered gap_01 audit is not structural PASS")
    except Exception as exc:
        issues.append(
            f"offline v1 recovery validation failed: "
            f"{type(exc).__name__}: {exc}"
        )
        diagnosis = None

    if issues:
        print("T1 v1 failure evidence verification: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Network calls:", 0)
        print("LLM calls:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    assert diagnosis is not None
    print("T1 v1 failure evidence verification: PASS")
    print("Failure Freeze ID:", observed_id)
    print("Manifest SHA256:", observed_sha)
    print("V1 source HEAD:", diagnosis["v1_source_git_head"])
    print("V1 exception:", diagnosis["v1_exception_type"])
    print("Recovered hypothesis:", diagnosis["hypothesis_id"])
    print("Recovered queries:", diagnosis["delta_query_count"])
    print(
        "Recovered provider executions:",
        diagnosis["successful_execution_count"],
        "success /",
        diagnosis["failed_execution_count"],
        "failed",
    )
    print(
        "Recovered delta works:",
        diagnosis["delta_canonical_work_count"],
        "canonical /",
        diagnosis["delta_abstract_work_count"],
        "with abstract",
    )
    print(
        "Every recovered query operational:",
        diagnosis["every_query_operational"],
    )
    print("Recovered structural PASS:", diagnosis["structural_pass"])
    print("V1 gap_01 network replay authorized:", False)
    print("Network calls during verification:", 0)
    print("LLM calls:", 0)
    print("Fresh Reserve C consumed:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
