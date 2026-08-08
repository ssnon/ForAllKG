from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.explorer_contracts import ExplorationReport, GraphExplorerPacket
from dac_her.explorer_validation import ExplorationReportValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministically validate a Graph Explorer report against its source packet."
    )
    parser.add_argument("--packet", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = GraphExplorerPacket.model_validate_json(Path(args.packet).read_text(encoding="utf-8"))
    report = ExplorationReport.model_validate_json(Path(args.report).read_text(encoding="utf-8"))
    result = ExplorationReportValidator().validate(packet, report)
    payload = result.model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not result.passes:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
