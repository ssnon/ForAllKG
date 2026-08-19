from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from campaigns.sers_alpha4_epoch.holdout.alpha4c4d2_holdout_support import (
    DATA_ROOT,
    ROOT,
    manual_decisions,
    read_jsonl,
)
from campaigns.sers_alpha4_epoch.alpha4.alpha4c5f2_strict_source import (
    STRICT_SOURCE_LAYOUT_SEMANTICS_ID,
    resolve_strict_source_attempt_aware,
    verify_strict_source_attempt_aware_unchanged,
)
from campaigns.sers_alpha4_epoch.readiness.canonical_readiness import (
    CanonicalReadinessError,
    atomic_json,
    canonical_graph_snapshot,
    make_readiness_lock,
    snapshot_optional,
)
from dac_her.measurement_merge_invariants import (
    MEASUREMENT_MERGE_INVARIANT_ID,
)


ALPHA4C5F21_SERS_READINESS_ID = (
    "sers_canonical_readiness_v1_alpha4c5f21_attempt_layout"
)
SERS_DOMAIN_PROFILE_ID = "sers_au_ag"
SERS_CONFIG = "configs/papers_sers_au_ag.yaml"

REFREEZE_ELIGIBLE_REASONS = frozenset(
    {
        "canonical_missing",
        "measurement_merge_invariant_mismatch",
        "measurement_numeric_text_xor_violation",
    }
)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _canonical_path(paper_id: str) -> Path:
    return (
        DATA_ROOT
        / "extracted"
        / paper_id
        / f"{paper_id}.graphml"
    )


def _decisions_path(paper_id: str) -> Path:
    return (
        DATA_ROOT
        / "extracted"
        / paper_id
        / "resolution"
        / "decisions.jsonl"
    )


def _snapshot(
    paper_id: str,
    *,
    detailed: bool,
) -> dict[str, Any]:
    row = canonical_graph_snapshot(
        _canonical_path(paper_id),
        expected_domain_profile_id=SERS_DOMAIN_PROFILE_ID,
        expected_measurement_merge_invariant_id=(
            MEASUREMENT_MERGE_INVARIANT_ID
        ),
        include_issue_details=detailed,
    )
    row["canonical_path"] = _relative(
        _canonical_path(paper_id)
    )
    return row


def audit_sers_alpha4c5f2_canonical_readiness(
    paper_id: str,
    *,
    detailed: bool = False,
) -> dict[str, Any]:
    source = resolve_strict_source_attempt_aware(paper_id)
    canonical = _snapshot(paper_id, detailed=detailed)
    return {
        "paper_id": paper_id,
        "readiness_id": ALPHA4C5F21_SERS_READINESS_ID,
        "strict_source_layout_semantics_id": (
            STRICT_SOURCE_LAYOUT_SEMANTICS_ID
        ),
        "strict_attempt_layout": source["attempt_layout"],
        "strict_quality_status": source[
            "extraction_quality"
        ]["graph_materialization_status"],
        "positive_evidence_queries_allowed": source[
            "extraction_quality"
        ]["positive_evidence_queries_allowed"],
        "active_payload_complete_flag": source[
            "active_payload_complete_flag"
        ],
        "requires_allow_incomplete": source[
            "requires_allow_incomplete"
        ],
        "canonical": canonical,
        "ready": canonical["ready"] is True,
        "refreeze_eligible": (
            bool(canonical["readiness_issues"])
            and set(canonical["readiness_issues"]).issubset(
                REFREEZE_ELIGIBLE_REASONS
            )
        ),
    }


def prepare_sers_alpha4c5f2_canonical_paper(
    paper_id: str,
    *,
    allow_refreeze: bool,
) -> dict[str, Any]:
    source = resolve_strict_source_attempt_aware(paper_id)
    verify_strict_source_attempt_aware_unchanged(source)

    decisions_path = _decisions_path(paper_id)
    before_rows = read_jsonl(decisions_path)
    before_manual = manual_decisions(before_rows)
    before_decisions_snapshot = snapshot_optional(
        ROOT,
        decisions_path,
    )
    before = _snapshot(paper_id, detailed=False)

    command: list[str] | None = None
    refrozen = False
    if before["ready"] is not True:
        reasons = set(before["readiness_issues"])
        if not reasons.issubset(REFREEZE_ELIGIBLE_REASONS):
            raise CanonicalReadinessError(
                f"{paper_id}: readiness failure is not eligible "
                f"for canonical refreeze: {sorted(reasons)!r}"
            )
        if not allow_refreeze:
            raise CanonicalReadinessError(
                f"{paper_id}: canonical refreeze required but "
                f"not allowed: {sorted(reasons)!r}"
            )

        command = [
            sys.executable,
            "-m",
            "scripts.build_paper_graph",
            "--paper-id",
            paper_id,
            "--config",
            SERS_CONFIG,
            "--domain-profile",
            SERS_DOMAIN_PROFILE_ID,
            "--data-root",
            "data_sers",
            "--run-id",
            source["run_id"],
        ]
        if source["attempt_id"]:
            command.extend(
                ["--attempt-id", source["attempt_id"]]
            )
        if source["requires_allow_incomplete"]:
            command.append("--allow-incomplete")

        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            raise CanonicalReadinessError(
                f"{paper_id}: deterministic canonical refreeze "
                f"failed with exit code {result.returncode}."
            )
        refrozen = True

    verify_strict_source_attempt_aware_unchanged(source)
    after = _snapshot(paper_id, detailed=False)
    if after["ready"] is not True:
        raise CanonicalReadinessError(
            f"{paper_id}: canonical is still not ready after "
            f"preparation: {after['readiness_issues']!r}"
        )
    if after["measurement_xor_issue_count"] != 0:
        raise CanonicalReadinessError(
            f"{paper_id}: nonzero Measurement XOR count "
            "after preparation."
        )

    after_rows = read_jsonl(decisions_path)
    after_manual = manual_decisions(after_rows)
    if before_manual != after_manual:
        raise CanonicalReadinessError(
            f"{paper_id}: manual resolution decisions changed "
            "during canonical preparation."
        )
    after_decisions_snapshot = snapshot_optional(
        ROOT,
        decisions_path,
    )

    return {
        "paper_id": paper_id,
        "readiness_id": ALPHA4C5F21_SERS_READINESS_ID,
        "strict_source_layout_semantics_id": (
            STRICT_SOURCE_LAYOUT_SEMANTICS_ID
        ),
        "strict_source": source,
        "strict_source_verified_unchanged": True,
        "canonical_before": before,
        "canonical": after,
        "refrozen": refrozen,
        "refreeze_command": command,
        "manual_resolution_decisions_preserved": True,
        "resolution_decisions_before":
            before_decisions_snapshot,
        "resolution_decisions": after_decisions_snapshot,
        "new_extraction_llm_calls": 0,
    }


def prepare_sers_alpha4c5f2_readiness_lock(
    *,
    paper_ids: Iterable[str],
    output_path: Path,
    allow_refreeze: bool,
    source_label: str,
) -> dict[str, Any]:
    ordered = [str(value) for value in paper_ids]
    records: dict[str, dict[str, Any]] = {}
    for paper_id in ordered:
        records[paper_id] = (
            prepare_sers_alpha4c5f2_canonical_paper(
                paper_id,
                allow_refreeze=allow_refreeze,
            )
        )

    lock = make_readiness_lock(
        root=ROOT,
        paper_ids=ordered,
        expected_domain_profile_id=SERS_DOMAIN_PROFILE_ID,
        paper_records=records,
        source_label=source_label,
    )
    atomic_json(output_path, lock)
    return lock
