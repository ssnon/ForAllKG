from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.llm_telemetry_report import summarize_usage_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GraphAgents LLM telemetry JSONL.")
    parser.add_argument("path", type=Path, help="Path to llm telemetry JSONL")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()

    report = summarize_usage_file(args.path)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
