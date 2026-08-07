from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Bridge v2.4 recovery/quarantine outcomes for a paper."
        )
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    paper_root = (
        PROJECT_ROOT
        / "data_dac"
        / "extracted"
        / args.paper_id
    )

    if args.run_id:
        run_dir = paper_root / "runs" / args.run_id
    else:
        latest = read_json(
            paper_root / "latest_run.json"
        )
        run_dir = Path(latest["run_directory"])

    pointer = run_dir / "latest_bridge_extraction.json"
    if pointer.exists():
        bridge_pointer = read_json(pointer)
        extraction_dir = Path(
            bridge_pointer["bridge_extraction_directory"]
        )
    else:
        bridge_root = run_dir / "bridge_extractions"
        candidates = sorted(
            bridge_root.glob("*/summary.json"),
            key=lambda item: item.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No Bridge extraction under {bridge_root}"
            )
        extraction_dir = candidates[-1].parent

    summary = read_json(
        extraction_dir / "summary.json"
    )
    manifest_path = extraction_dir / "manifest.jsonl"
    records = []
    if manifest_path.exists():
        for line in manifest_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                records.append(json.loads(line))

    status_counts = Counter(
        str(item.get("status", "unknown"))
        for item in records
    )
    failure_classes = Counter(
        str(item.get("failure_class", ""))
        for item in records
        if item.get("failure_class")
    )

    print("Bridge extraction:", extraction_dir)
    print("Complete:", summary.get("complete"))
    print("Chunks:", summary.get("chunks"))
    print("Status counts:", dict(status_counts))
    print("Failure classes:", dict(failure_classes))
    print(
        "Quarantined candidates:",
        sum(
            int(item.get("quarantined_candidate_count", 0) or 0)
            for item in records
        ),
    )
    print(
        "Quarantined links:",
        sum(
            int(item.get("quarantined_link_count", 0) or 0)
            for item in records
        ),
    )
    print(
        "Repaired candidates:",
        sum(
            int(item.get("repaired_candidate_count", 0) or 0)
            for item in records
        ),
    )
    print(
        "Normalization operations:",
        sum(
            int(item.get("normalization_count", 0) or 0)
            for item in records
        ),
    )

    quarantine_dir = extraction_dir / "quarantine"
    issue_counts: Counter[str] = Counter()
    quarantine_chunks = 0
    if quarantine_dir.exists():
        for path in sorted(
            quarantine_dir.glob("*__quarantine.json")
        ):
            payload = read_json(path)
            items = payload.get("items", [])
            if items:
                quarantine_chunks += 1
            for item in items:
                for issue in item.get("issues", []):
                    issue_counts[str(issue)[:240]] += 1

    print("Chunks with quarantine:", quarantine_chunks)
    if issue_counts:
        print("\nTop quarantine issues:")
        for issue, count in issue_counts.most_common(20):
            print(f"  {count:4d}  {issue}")


if __name__ == "__main__":
    main()
