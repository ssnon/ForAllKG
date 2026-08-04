from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dac_her.resolution_candidates import read_jsonl
from dac_her.run_state import paper_output_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply reviewed same-entity/different-entity clusters to the "
            "stable paper-level decisions.jsonl file."
        )
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--reviewer", default="human_review")
    parser.add_argument("--notes", default=None)
    return parser.parse_args()


def _candidate_id(decision: str, left_id: str, right_id: str) -> str:
    pair = "|".join(sorted((left_id, right_id)))
    digest = hashlib.sha256(f"{decision}|{pair}".encode("utf-8")).hexdigest()[:20]
    return f"manual_resolution:{digest}"


def _pair_key(record: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted((str(record.get("left_id", "")), str(record.get("right_id", "")))))


def _load_plan(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Resolution review plan must be a YAML mapping.")
    return payload


def main() -> None:
    args = parse_args()
    plan_path = Path(args.plan).resolve()
    plan = _load_plan(plan_path)

    paper_root = paper_output_root(PROJECT_ROOT, args.paper_id)
    decisions_path = paper_root / "resolution" / "decisions.jsonl"
    decisions_path.parent.mkdir(parents=True, exist_ok=True)

    existing = read_jsonl(decisions_path)
    by_pair = {_pair_key(record): record for record in existing if all(_pair_key(record))}
    now = datetime.now(timezone.utc).isoformat()

    applied = 0
    clusters = plan.get("same_entity_clusters") or []
    if not isinstance(clusters, list):
        raise ValueError("same_entity_clusters must be a list.")

    for cluster in clusters:
        if not isinstance(cluster, dict):
            raise ValueError("Each same_entity cluster must be a mapping.")
        canonical_id = str(cluster.get("canonical_id", "")).strip()
        members = [
            str(value).strip()
            for value in (cluster.get("members") or [])
            if str(value).strip()
        ]
        if not canonical_id or canonical_id not in members or len(set(members)) < 2:
            raise ValueError(
                "Each cluster needs canonical_id and at least two unique members, "
                "including the canonical_id."
            )
        for member in sorted(set(members)):
            if member == canonical_id:
                continue
            key = tuple(sorted((canonical_id, member)))
            base = by_pair.get(key, {})
            base.update({
                "candidate_id": base.get("candidate_id")
                or _candidate_id("same_entity", canonical_id, member),
                "left_id": canonical_id,
                "right_id": member,
                "decision": "same_entity",
                "approved": True,
                "canonical_id": canonical_id,
                "reviewer": args.reviewer,
                "reviewed_at": now,
                "notes": cluster.get("notes") or args.notes,
            })
            by_pair[key] = base
            applied += 1

    separate_pairs = plan.get("different_entity_pairs") or []
    if not isinstance(separate_pairs, list):
        raise ValueError("different_entity_pairs must be a list.")
    for item in separate_pairs:
        if isinstance(item, dict):
            left_id = str(item.get("left_id", "")).strip()
            right_id = str(item.get("right_id", "")).strip()
            notes = item.get("notes") or args.notes
        elif isinstance(item, list) and len(item) == 2:
            left_id, right_id = map(str, item)
            left_id, right_id = left_id.strip(), right_id.strip()
            notes = args.notes
        else:
            raise ValueError("different_entity_pairs entries need two node IDs.")
        if not left_id or not right_id or left_id == right_id:
            raise ValueError("Invalid different-entity pair.")
        key = tuple(sorted((left_id, right_id)))
        base = by_pair.get(key, {})
        base.update({
            "candidate_id": base.get("candidate_id")
            or _candidate_id("different_entity", left_id, right_id),
            "left_id": left_id,
            "right_id": right_id,
            "decision": "different_entity",
            "approved": True,
            "canonical_id": None,
            "reviewer": args.reviewer,
            "reviewed_at": now,
            "notes": notes,
        })
        by_pair[key] = base
        applied += 1

    if decisions_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(decisions_path, decisions_path.with_suffix(f".jsonl.{stamp}.bak"))

    records = sorted(
        by_pair.values(),
        key=lambda record: (
            str(record.get("left_id", "")),
            str(record.get("right_id", "")),
            str(record.get("candidate_id", "")),
        ),
    )
    decisions_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    print(f"Applied {applied} reviewed pair decision(s).")
    print(f"Wrote: {decisions_path}")


if __name__ == "__main__":
    main()
