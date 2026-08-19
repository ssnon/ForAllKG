from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from campaigns.sers_alpha4_epoch.alpha4.alpha4c5f2_readiness import (
    audit_sers_alpha4c5f2_canonical_readiness,
    prepare_sers_alpha4c5f2_readiness_lock,
)


ROOT = Path.cwd()


class ReadinessCliError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "alpha4c.5f.2 attempt-layout-aware canonical readiness "
            "gate. Structural-only pre-consumption audit; zero LLM calls."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--confirm-canonical-refreeze",
        action="store_true",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReadinessCliError(
            f"Expected JSON object: {path}"
        )
    return value


def _paper_ids(protocol: dict[str, Any]) -> list[str]:
    raw = protocol.get("reserve_paper_ids")
    if raw is None:
        raw = protocol.get("paper_ids")
    if not isinstance(raw, list) or not raw:
        raise ReadinessCliError(
            "Protocol must contain a non-empty "
            "reserve_paper_ids or paper_ids list."
        )
    paper_ids = [str(value) for value in raw]
    if len(set(paper_ids)) != len(paper_ids):
        raise ReadinessCliError(
            "Protocol paper list contains duplicates."
        )
    return paper_ids


def main() -> int:
    args = parse_args()
    protocol_path = (
        args.protocol
        if args.protocol.is_absolute()
        else ROOT / args.protocol
    )
    protocol = _load_json(protocol_path)
    paper_ids = _paper_ids(protocol)

    audits = [
        audit_sers_alpha4c5f2_canonical_readiness(
            paper_id,
            detailed=False,
        )
        for paper_id in paper_ids
    ]
    not_ready = [
        row for row in audits if row["ready"] is not True
    ]
    attempt_layout_count = sum(
        bool(row["strict_attempt_layout"])
        for row in audits
    )

    print(
        "alpha4c.5f.2.1 attempt-aware SERS canonical "
        "readiness pre-consumption gate"
    )
    print("Protocol:", protocol_path)
    print("Papers:", len(paper_ids))
    print("Attempt-layout Strict sources:", attempt_layout_count)
    print(
        "Legacy-flat Strict sources:",
        len(paper_ids) - attempt_layout_count,
    )
    print("Ready:", len(paper_ids) - len(not_ready))
    print("Not ready:", len(not_ready))
    print("Scientific value disclosure:", False)
    print("New extraction LLM calls:", 0)
    print("Reserve consumed:", False)

    for row in not_ready:
        print(
            " -",
            row["paper_id"],
            "issues=",
            row["canonical"]["readiness_issues"],
            "refreeze_eligible=",
            row["refreeze_eligible"],
        )

    if args.preflight:
        if not_ready:
            print("Readiness preflight: BLOCKED")
            return 2
        print("Readiness preflight: PASS")
        return 0

    if args.output is None:
        raise ReadinessCliError(
            "--output is required with --prepare."
        )
    if not args.confirm_canonical_refreeze:
        raise ReadinessCliError(
            "--confirm-canonical-refreeze is required with "
            "--prepare. No extraction LLM is called; only eligible "
            "canonical migration may occur."
        )
    if any(
        not row["refreeze_eligible"] for row in not_ready
    ):
        blocked = [
            row["paper_id"]
            for row in not_ready
            if not row["refreeze_eligible"]
        ]
        raise ReadinessCliError(
            "Non-refreeze-eligible readiness failures: "
            + ", ".join(blocked)
        )

    output_path = (
        args.output
        if args.output.is_absolute()
        else ROOT / args.output
    )
    if output_path.exists():
        raise ReadinessCliError(
            f"Refusing to replace existing readiness lock: "
            f"{output_path}"
        )

    lock = prepare_sers_alpha4c5f2_readiness_lock(
        paper_ids=paper_ids,
        output_path=output_path,
        allow_refreeze=True,
        source_label=str(args.protocol),
    )
    print("Readiness preparation: PASS")
    print("Lock semantics:", lock["semantics_id"])
    print("Lock SHA256:", lock["lock_sha256"])
    print("All ready:", lock["all_ready"])
    print("Scientific value disclosure:", False)
    print("New extraction LLM calls:", 0)
    print("Reserve consumed:", False)
    print("Saved:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
