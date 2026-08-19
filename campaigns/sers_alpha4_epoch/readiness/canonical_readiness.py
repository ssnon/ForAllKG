from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx

from dac_her.measurement_merge_invariants import (
    MEASUREMENT_MERGE_INVARIANT_ID,
    measurement_value_payload_issues,
)


CANONICAL_READINESS_SEMANTICS_ID = (
    "canonical_readiness_gate_v1_alpha4c5f1"
)


class CanonicalReadinessError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".alpha4c5f1.tmp")
    tmp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def snapshot_optional(root: Path, path: Path) -> dict[str, Any]:
    try:
        relative = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        relative = str(path)
    if not path.exists():
        return {
            "path": relative,
            "present": False,
            "sha256": "",
        }
    return {
        "path": relative,
        "present": True,
        "sha256": sha256_file(path),
    }


def canonical_graph_snapshot(
    graph_path: Path,
    *,
    expected_domain_profile_id: str,
    expected_measurement_merge_invariant_id: str = (
        MEASUREMENT_MERGE_INVARIANT_ID
    ),
    include_issue_details: bool = False,
) -> dict[str, Any]:
    """Return structural readiness only; no scientific interpretation.

    `include_issue_details` is deliberately opt-in. It is suitable for an
    already-consumed/seen debugging fixture, but should remain False for a
    genuinely blind reserve pre-consumption gate.
    """
    if not graph_path.exists():
        return {
            "canonical_present": False,
            "canonical_path": str(graph_path),
            "canonical_sha256": "",
            "canonical_nodes": None,
            "canonical_edges": None,
            "domain_profile_id": "",
            "measurement_merge_invariant_id": "",
            "measurement_xor_issue_count": None,
            "measurement_xor_issues": [] if include_issue_details else None,
            "ready": False,
            "readiness_issues": ["canonical_missing"],
        }

    graph = nx.read_graphml(graph_path, force_multigraph=True)
    domain_profile_id = str(
        graph.graph.get("domain_profile_id", "")
    ).strip()
    invariant_id = str(
        graph.graph.get("measurement_merge_invariant_id", "")
    ).strip()
    xor_issues = measurement_value_payload_issues(graph)

    readiness_issues: list[str] = []
    if domain_profile_id != expected_domain_profile_id:
        readiness_issues.append("domain_profile_mismatch")
    if invariant_id != expected_measurement_merge_invariant_id:
        readiness_issues.append("measurement_merge_invariant_mismatch")
    if xor_issues:
        readiness_issues.append("measurement_numeric_text_xor_violation")

    payload = {
        "canonical_present": True,
        "canonical_path": str(graph_path),
        "canonical_sha256": sha256_file(graph_path),
        "canonical_nodes": graph.number_of_nodes(),
        "canonical_edges": graph.number_of_edges(),
        "domain_profile_id": domain_profile_id,
        "measurement_merge_invariant_id": invariant_id,
        "measurement_xor_issue_count": len(xor_issues),
        "measurement_xor_issues": (
            xor_issues if include_issue_details else None
        ),
        "ready": not readiness_issues,
        "readiness_issues": readiness_issues,
    }
    return payload


def make_readiness_lock(
    *,
    root: Path,
    paper_ids: Iterable[str],
    expected_domain_profile_id: str,
    paper_records: Mapping[str, Mapping[str, Any]],
    source_label: str,
) -> dict[str, Any]:
    ordered = [str(value) for value in paper_ids]
    if not ordered or len(set(ordered)) != len(ordered):
        raise CanonicalReadinessError(
            "Readiness lock requires a non-empty unique paper list."
        )
    if set(paper_records) != set(ordered):
        raise CanonicalReadinessError(
            "Readiness paper records do not exactly match paper_ids."
        )

    records: dict[str, Any] = {}
    for paper_id in ordered:
        row = dict(paper_records[paper_id])
        canonical = row.get("canonical")
        if not isinstance(canonical, Mapping):
            raise CanonicalReadinessError(
                f"{paper_id}: canonical readiness record missing."
            )
        if canonical.get("ready") is not True:
            raise CanonicalReadinessError(
                f"{paper_id}: canonical is not ready: "
                f"{canonical.get('readiness_issues')!r}"
            )
        if canonical.get("measurement_xor_issue_count") != 0:
            raise CanonicalReadinessError(
                f"{paper_id}: nonzero Measurement XOR count."
            )
        if (
            canonical.get("measurement_merge_invariant_id")
            != MEASUREMENT_MERGE_INVARIANT_ID
        ):
            raise CanonicalReadinessError(
                f"{paper_id}: unexpected Measurement merge invariant."
            )
        if canonical.get("domain_profile_id") != expected_domain_profile_id:
            raise CanonicalReadinessError(
                f"{paper_id}: unexpected domain profile."
            )
        # Issue details must never be persisted in a blind readiness lock.
        canonical = dict(canonical)
        canonical.pop("measurement_xor_issues", None)
        row["canonical"] = canonical
        records[paper_id] = row

    payload: dict[str, Any] = {
        "semantics_id": CANONICAL_READINESS_SEMANTICS_ID,
        "created_at": now_iso(),
        "source_label": source_label,
        "expected_domain_profile_id": expected_domain_profile_id,
        "expected_measurement_merge_invariant_id": (
            MEASUREMENT_MERGE_INVARIANT_ID
        ),
        "paper_ids": ordered,
        "paper_count": len(ordered),
        "paper_records": records,
        "all_ready": True,
        "scientific_values_disclosed": False,
        "root": str(root),
    }
    payload["lock_sha256"] = sha256_json(payload)
    return payload


def _resolve_lock_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def verify_readiness_lock(
    *,
    root: Path,
    lock: Mapping[str, Any],
    expected_paper_ids: Iterable[str],
    expected_domain_profile_id: str,
) -> list[str]:
    issues: list[str] = []
    expected = [str(value) for value in expected_paper_ids]

    if lock.get("semantics_id") != CANONICAL_READINESS_SEMANTICS_ID:
        issues.append("readiness semantics ID mismatch")
    if lock.get("paper_ids") != expected:
        issues.append("readiness paper list/order mismatch")
    if lock.get("paper_count") != len(expected):
        issues.append("readiness paper count mismatch")
    if lock.get("all_ready") is not True:
        issues.append("readiness lock is not all_ready")
    if lock.get("scientific_values_disclosed") is not False:
        issues.append("readiness lock disclosed scientific values")
    if lock.get("expected_domain_profile_id") != expected_domain_profile_id:
        issues.append("readiness expected domain mismatch")
    if (
        lock.get("expected_measurement_merge_invariant_id")
        != MEASUREMENT_MERGE_INVARIANT_ID
    ):
        issues.append("readiness expected invariant mismatch")

    raw_records = lock.get("paper_records")
    if not isinstance(raw_records, Mapping):
        issues.append("readiness paper_records missing")
        return issues
    if set(raw_records) != set(expected):
        issues.append("readiness paper_records set mismatch")
        return issues

    for paper_id in expected:
        record = raw_records.get(paper_id)
        if not isinstance(record, Mapping):
            issues.append(f"{paper_id}: readiness record invalid")
            continue
        canonical = record.get("canonical")
        if not isinstance(canonical, Mapping):
            issues.append(f"{paper_id}: canonical record missing")
            continue
        raw_path = str(canonical.get("canonical_path") or "")
        if not raw_path:
            issues.append(f"{paper_id}: canonical path missing")
            continue
        path = _resolve_lock_path(root, raw_path)
        if not path.exists():
            issues.append(f"{paper_id}: locked canonical disappeared")
            continue
        observed_sha = sha256_file(path)
        if observed_sha != canonical.get("canonical_sha256"):
            issues.append(f"{paper_id}: canonical SHA256 drifted")
            continue

        observed = canonical_graph_snapshot(
            path,
            expected_domain_profile_id=expected_domain_profile_id,
            include_issue_details=False,
        )
        if observed.get("ready") is not True:
            issues.append(
                f"{paper_id}: canonical no longer ready: "
                f"{observed.get('readiness_issues')!r}"
            )
        if observed.get("measurement_xor_issue_count") != 0:
            issues.append(f"{paper_id}: Measurement XOR drifted")

        decision = record.get("resolution_decisions")
        if isinstance(decision, Mapping):
            decision_path_raw = str(decision.get("path") or "")
            if decision_path_raw:
                decision_path = _resolve_lock_path(root, decision_path_raw)
                present = decision_path.exists()
                if present != bool(decision.get("present")):
                    issues.append(
                        f"{paper_id}: resolution decisions presence drifted"
                    )
                elif present and (
                    sha256_file(decision_path) != decision.get("sha256")
                ):
                    issues.append(
                        f"{paper_id}: resolution decisions SHA256 drifted"
                    )

    expected_lock_sha = lock.get("lock_sha256")
    if not isinstance(expected_lock_sha, str) or not expected_lock_sha:
        issues.append("readiness lock SHA256 missing")
    else:
        payload = dict(lock)
        payload.pop("lock_sha256", None)
        if sha256_json(payload) != expected_lock_sha:
            issues.append("readiness lock payload SHA256 mismatch")

    return issues


def load_and_verify_readiness_lock(
    *,
    root: Path,
    lock_path: Path,
    expected_paper_ids: Iterable[str],
    expected_domain_profile_id: str,
) -> dict[str, Any]:
    if not lock_path.exists():
        raise CanonicalReadinessError(
            f"Canonical readiness lock missing: {lock_path}"
        )
    value = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CanonicalReadinessError(
            f"Canonical readiness lock must be a JSON object: {lock_path}"
        )
    issues = verify_readiness_lock(
        root=root,
        lock=value,
        expected_paper_ids=expected_paper_ids,
        expected_domain_profile_id=expected_domain_profile_id,
    )
    if issues:
        raise CanonicalReadinessError(
            "Canonical readiness verification failed:\n- "
            + "\n- ".join(issues)
        )
    return value


def guarded_write_consumption_marker(
    *,
    root: Path,
    lock_path: Path,
    marker_path: Path,
    expected_paper_ids: Iterable[str],
    expected_domain_profile_id: str,
    marker_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a reserve-consumption marker only after a current lock passes.

    The lock is revalidated immediately before the marker write, so canonical
    bytes, Measurement XOR, domain identity, and resolution-decision hashes
    cannot silently drift between preparation and reserve consumption.
    """
    if marker_path.exists():
        raise CanonicalReadinessError(
            f"Consumption marker already exists: {marker_path}"
        )
    lock = load_and_verify_readiness_lock(
        root=root,
        lock_path=lock_path,
        expected_paper_ids=expected_paper_ids,
        expected_domain_profile_id=expected_domain_profile_id,
    )
    payload = dict(marker_payload)
    payload.update(
        {
            "canonical_readiness_semantics_id": (
                CANONICAL_READINESS_SEMANTICS_ID
            ),
            "canonical_readiness_lock_path": str(lock_path),
            "canonical_readiness_lock_sha256": lock["lock_sha256"],
            "canonical_readiness_verified_immediately_before_consumption": (
                True
            ),
            "canonical_readiness_verified_at": now_iso(),
        }
    )
    atomic_json(marker_path, payload)
    return payload
