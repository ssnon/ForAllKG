from __future__ import annotations

import argparse
import json

from dac_her.llm_telemetry_backfill import backfill_extraction_resolutions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Append artifact-resolution records for existing extraction LLM "
            "telemetry using each paper's latest run artifacts."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--telemetry-path", required=True)
    parser.add_argument("--paper-ids", nargs="+", required=True)
    args = parser.parse_args()

    report = backfill_extraction_resolutions(
        data_root=args.data_root,
        paper_ids=args.paper_ids,
        telemetry_path=args.telemetry_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
