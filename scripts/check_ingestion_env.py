from __future__ import annotations

import argparse
import os
from pathlib import Path

from dac_her.ingestion.marker_runner import MarkerSingleRunner, marker_version


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local prerequisites for Drive ingestion.")
    parser.add_argument("--marker-command", default=os.getenv("DAC_MARKER_COMMAND", "marker_single"))
    parser.add_argument("--credentials", default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    args = parser.parse_args()
    print(f"marker command: {args.marker_command}")
    print(f"marker-pdf version: {marker_version()}")
    MarkerSingleRunner(command=args.marker_command).preflight()
    print("marker CLI: OK")
    if not args.credentials:
        print("Google credentials: MISSING (set GOOGLE_APPLICATION_CREDENTIALS)")
    else:
        path = Path(args.credentials)
        print(f"Google credentials: {'OK' if path.is_file() else 'MISSING'} ({path})")


if __name__ == "__main__":
    main()
