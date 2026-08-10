from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dac_her.ingestion.runtime import SyncConfig, run_sync


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize Drive PDFs into Marker Markdown and an ingestion corpus manifest."
    )
    parser.add_argument("--drive-folder-id", default=os.getenv("DAC_DRIVE_FOLDER_ID"))
    parser.add_argument("--spreadsheet-id", default=os.getenv("DAC_ARTICLE_SHEET_ID"))
    parser.add_argument("--sheet-range", default=os.getenv("DAC_ARTICLE_SHEET_RANGE", "Sheet1!A:H"))
    parser.add_argument(
        "--credentials",
        default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        help="Service-account JSON file. Share the source Drive folder and Sheet with this account.",
    )
    parser.add_argument("--data-root", default="data_dac/ingestion")
    parser.add_argument("--registry", default="data_dac/ingestion/registry/papers.json")
    parser.add_argument("--alias-map", default="configs/ingestion/annotator_aliases.json")
    parser.add_argument("--corpus-id", default="dac_drive_latest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--force-reconvert", action="store_true")
    parser.add_argument("--marker-command", default=os.getenv("DAC_MARKER_COMMAND", "marker_single"))
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=float(os.getenv("DAC_MARKER_HEARTBEAT_SECONDS", "30")),
        help="Print a marker_single heartbeat every N seconds while one document is converting. Use 0 to disable.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable terminal progress messages. progress.json is still written.",
    )
    parser.add_argument(
        "--marker-arg",
        action="append",
        default=[],
        help="Extra marker_single argument. Repeat for multiple arguments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [
        name
        for name, value in [
            ("--drive-folder-id", args.drive_folder_id),
            ("--spreadsheet-id", args.spreadsheet_id),
            ("--credentials", args.credentials),
        ]
        if not value
    ]
    if missing:
        raise SystemExit("Missing required configuration: " + ", ".join(missing))
    alias_path = Path(args.alias_map)
    config = SyncConfig(
        credentials_path=Path(args.credentials),
        drive_folder_id=args.drive_folder_id,
        spreadsheet_id=args.spreadsheet_id,
        sheet_range=args.sheet_range,
        data_root=Path(args.data_root),
        registry_path=Path(args.registry),
        alias_map_path=alias_path if alias_path.exists() else None,
        corpus_id=args.corpus_id,
        dry_run=args.dry_run,
        convert=not args.download_only,
        force_reconvert=args.force_reconvert,
        marker_command=args.marker_command,
        marker_extra_args=tuple(args.marker_arg),
        show_progress=not args.quiet,
        heartbeat_seconds=max(0.0, args.heartbeat_seconds),
    )
    report = run_sync(config)
    print(json.dumps(report["status_counts"], ensure_ascii=False, indent=2))
    print(f"report: {Path(args.data_root) / 'runs' / 'latest.json'}")
    if report.get("manifest_path"):
        print(f"manifest: {report['manifest_path']}")


if __name__ == "__main__":
    main()
