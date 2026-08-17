from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from dac_her.fresh_c_activation import (
    validate_historical_sweep_artifacts,
)
from scripts.build_sers_fresh_c_historical_exclusion_ledger_v1 import (
    DEFAULT_OUTPUT_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the Fresh-C C0.1B historical exclusion ledger "
            "without network, LLM, or Fresh-C access."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
    ).strip()


def main() -> int:
    args = parse_args()
    root = Path(
        _git(Path.cwd(), "rev-parse", "--show-toplevel")
    )
    output = (
        args.output_dir
        if args.output_dir.is_absolute()
        else root / args.output_dir
    )
    manifest, ledger = validate_historical_sweep_artifacts(
        root=root,
        manifest_path=(
            output / "historical_identity_sweep_manifest.json"
        ),
        ledger_path=(
            output / "historical_exclusion_ledger.json"
        ),
    )

    print("Fresh-C C0.1B historical exclusion ledger verifier")
    print(f"Source commit: {manifest.source_commit}")
    print(f"Sweep ID: {manifest.sweep_id}")
    print(f"Sweep SHA256: {manifest.sweep_sha256}")
    print(f"Ledger ID: {ledger.ledger_id}")
    print(f"Ledger SHA256: {ledger.ledger_sha256}")
    print(
        "Canonical historical identities: "
        f"{len(ledger.canonical_ids)}"
    )
    print("Scientific content retained: False")
    print("Network calls: 0")
    print("LLM calls: 0")
    print("Fresh Reserve C consumed: False")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
