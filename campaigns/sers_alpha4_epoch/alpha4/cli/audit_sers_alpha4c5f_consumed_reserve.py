from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from campaigns.sers_alpha4_epoch.alpha4.alpha4c5f_reserve import (
    DEFAULT_5F_PROTOCOL_PATH,
    load_5f_protocol,
)
from campaigns.sers_alpha4_epoch.readiness.canonical_readiness import (
    atomic_json,
    canonical_graph_snapshot,
    now_iso,
)
from dac_her.measurement_merge_invariants import (
    MEASUREMENT_MERGE_INVARIANT_ID,
)


ROOT = Path.cwd()
SERS_DOMAIN_PROFILE_ID = "sers_au_ag"
DEFAULT_OUTPUT = Path(
    "evaluation/sers_alpha4c5f1/consumed_v3_seen/"
    "pre_repair_readiness_audit.json"
)


class ConsumedReserveAuditError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only alpha4c.5f consumed-v3 canonical readiness audit. "
            "This script never reopens or reruns the failed reserve."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_5F_PROTOCOL_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConsumedReserveAuditError(f"Required JSON missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConsumedReserveAuditError(f"Expected JSON object: {path}")
    return value


def _resolved(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def main() -> int:
    args = parse_args()
    protocol = load_5f_protocol(ROOT / args.protocol)
    old_root = ROOT / protocol.evaluation_root
    marker = old_root / "consumption_started.json"
    failure = old_root / "CAMPAIGN_FAIL.json"
    passed = old_root / "CAMPAIGN_PASS.json"
    source_lock_path = old_root / "canonical_source_lock.json"

    if not marker.exists():
        raise ConsumedReserveAuditError(
            "The alpha4c.5f reserve is not marked consumed; detailed audit "
            "is forbidden for an unconsumed reserve."
        )
    if not failure.exists():
        raise ConsumedReserveAuditError(
            "Expected alpha4c.5f CAMPAIGN_FAIL.json is missing."
        )
    if passed.exists():
        raise ConsumedReserveAuditError(
            "Refusing consumed-failure audit because a PASS marker exists."
        )

    marker_json = read_json(marker)
    failure_json = read_json(failure)
    source_lock = read_json(source_lock_path)
    rows_by_paper = {
        str(row.get("paper_id")): row
        for row in source_lock.get("source_rows", [])
        if isinstance(row, dict)
    }
    if set(rows_by_paper) != set(protocol.reserve_paper_ids):
        raise ConsumedReserveAuditError(
            "Historical canonical_source_lock paper set does not match the "
            "frozen 5f reserve."
        )

    paper_rows: list[dict[str, Any]] = []
    campaign_issue_total = 0
    current_issue_total = 0
    campaign_not_ready = 0
    current_not_ready = 0

    for paper_id in protocol.reserve_paper_ids:
        historical = rows_by_paper[paper_id]
        campaign_path = _resolved(str(historical["campaign_path"]))
        current_path = (
            ROOT
            / "data_sers"
            / "extracted"
            / paper_id
            / f"{paper_id}.graphml"
        )

        campaign = canonical_graph_snapshot(
            campaign_path,
            expected_domain_profile_id=SERS_DOMAIN_PROFILE_ID,
            expected_measurement_merge_invariant_id=(
                MEASUREMENT_MERGE_INVARIANT_ID
            ),
            include_issue_details=True,
        )
        current = canonical_graph_snapshot(
            current_path,
            expected_domain_profile_id=SERS_DOMAIN_PROFILE_ID,
            expected_measurement_merge_invariant_id=(
                MEASUREMENT_MERGE_INVARIANT_ID
            ),
            include_issue_details=True,
        )

        campaign_count = int(campaign.get("measurement_xor_issue_count") or 0)
        current_count = int(current.get("measurement_xor_issue_count") or 0)
        campaign_issue_total += campaign_count
        current_issue_total += current_count
        campaign_not_ready += int(campaign.get("ready") is not True)
        current_not_ready += int(current.get("ready") is not True)

        paper_rows.append(
            {
                "paper_id": paper_id,
                "historical_source_sha256": historical.get("source_sha256"),
                "historical_campaign_sha256": historical.get("campaign_sha256"),
                "historical_source_campaign_sha_equal": (
                    historical.get("source_sha256")
                    == historical.get("campaign_sha256")
                ),
                "campaign_frozen_copy": campaign,
                "current_source_canonical": current,
                "current_matches_historical_source_sha": (
                    current.get("canonical_sha256")
                    == historical.get("source_sha256")
                ),
            }
        )

    report = {
        "phase": "alpha4c.5f.1",
        "audit_id": "alpha4c5f1_consumed_v3_readiness_audit_v1",
        "created_at": now_iso(),
        "historical_campaign_id": protocol.campaign_id,
        "historical_protocol_id": protocol.protocol_id,
        "historical_protocol_sha256": protocol.protocol_sha256,
        "historical_reserve_consumed": marker_json.get("reserve_consumed"),
        "historical_failure_error_type": failure_json.get("error_type"),
        "historical_failure_error": failure_json.get("error"),
        "paper_ids": protocol.reserve_paper_ids,
        "paper_count": len(protocol.reserve_paper_ids),
        "campaign_frozen_copy_not_ready_count": campaign_not_ready,
        "campaign_frozen_copy_xor_issue_total": campaign_issue_total,
        "current_source_not_ready_count": current_not_ready,
        "current_source_xor_issue_total": current_issue_total,
        "detailed_issue_disclosure_allowed": True,
        "disclosure_reason": (
            "The alpha4c.5f v3 reserve is already consumed/seen."
        ),
        "new_extraction_llm_calls": 0,
        "reserve_rerun_attempted": False,
        "historical_campaign_modified": False,
        "papers": paper_rows,
    }
    atomic_json(ROOT / args.output, report)

    print("alpha4c.5f.1 consumed-v3 canonical readiness audit")
    print("Historical reserve consumed: True")
    print("Historical reserve rerun attempted: False")
    print("Papers:", len(protocol.reserve_paper_ids))
    print("Frozen campaign not-ready papers:", campaign_not_ready)
    print("Frozen campaign XOR issues:", campaign_issue_total)
    print("Current source not-ready papers:", current_not_ready)
    print("Current source XOR issues:", current_issue_total)
    print("New extraction LLM calls: 0")
    print("Historical campaign modified: False")
    print("Saved:", ROOT / args.output)

    for row in paper_rows:
        campaign = row["campaign_frozen_copy"]
        if campaign.get("ready") is True:
            continue
        print(
            " -",
            row["paper_id"],
            "campaign readiness issues=",
            campaign.get("readiness_issues"),
            "xor=",
            campaign.get("measurement_xor_issue_count"),
        )
        for issue in campaign.get("measurement_xor_issues") or []:
            print(
                "   *",
                issue.get("id"),
                "metric=",
                issue.get("metric_id"),
                "numeric=",
                repr(issue.get("value_numeric")),
                "text=",
                repr(issue.get("value_text")),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
