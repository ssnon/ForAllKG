
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.prior_art_candidate_memory_shadow import (
    build_candidate_memory,
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--historical-source-binding", required=True, type=Path)
    p.add_argument("--historical-packet", required=True, type=Path)
    p.add_argument("--historical-reviews", required=True, type=Path)
    p.add_argument("--current-source-binding", required=True, type=Path)
    p.add_argument("--current-packet", required=True, type=Path)
    p.add_argument("--current-reviews", required=True, type=Path)
    p.add_argument("--output-prefix", required=True, type=Path)
    args = p.parse_args()

    report, packet, reviews = build_candidate_memory(
        historical_source_binding=load(
            args.historical_source_binding
        ),
        historical_packet=load(args.historical_packet),
        historical_reviews=load(args.historical_reviews),
        current_source_binding=load(
            args.current_source_binding
        ),
        current_packet=load(args.current_packet),
        current_reviews=load(args.current_reviews),
    )

    prefix = args.output_prefix
    write(prefix.with_suffix(".report.json"), report)
    write(prefix.with_suffix(".prior_art.json"), packet)
    write(prefix.with_suffix(".selection_reviews.json"), reviews)

    print("===== PRIOR-ART CANDIDATE MEMORY SHADOW =====")
    print("Current claims:", report["current_claim_count"])
    print(
        "Mapped current claims:",
        report["mapped_current_claim_count"],
    )
    print(
        "Claims with reexposed works:",
        report["reexposed_current_claim_count"],
    )
    print(
        "Reexposed claim-work pairs:",
        report["reexposed_claim_work_pair_count"],
    )
    print(
        "Historical works newly added:",
        report["new_work_added_to_current_packet_count"],
    )
    for row in report["mappings"]:
        if row["reexposed_work_ids"]:
            print(
                row["hypothesis_id"],
                "|", row["current_claim_id"],
                "| historical=", row["historical_claim_ids"],
                "| reexposed=", len(row["reexposed_work_ids"]),
            )
    print("MEMORY_IS_EVIDENCE=False")
    print("NEW_SEARCH_PERFORMED=False")
    print("SHADOW_ONLY=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
