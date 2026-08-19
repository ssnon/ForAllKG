from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_activation import (
    build_historical_identity_sweep,
)


DEFAULT_OUTPUT_DIR = Path(
    "evaluation/sers_fresh_c/c0_1b_activation_readiness_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Fresh-C historical-exposure identity ledger. "
            "This scans only pre-existing local repository artifacts, "
            "performs zero network/LLM calls, retains no scientific text, "
            "and does not activate or consume Fresh C."
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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    root = Path(
        _git(Path.cwd(), "rev-parse", "--show-toplevel")
    )

    if subprocess.run(
        ["git", "diff", "--quiet", "--"],
        cwd=root,
        check=False,
    ).returncode != 0:
        raise RuntimeError(
            "Tracked worktree is dirty; refuse historical-ledger build."
        )
    if subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--"],
        cwd=root,
        check=False,
    ).returncode != 0:
        raise RuntimeError(
            "Index is dirty; refuse historical-ledger build."
        )

    source_commit = _git(root, "rev-parse", "HEAD")
    output = (
        args.output_dir
        if args.output_dir.is_absolute()
        else root / args.output_dir
    )
    manifest_path = output / "historical_identity_sweep_manifest.json"
    ledger_path = output / "historical_exclusion_ledger.json"
    if manifest_path.exists() or ledger_path.exists():
        raise FileExistsError(
            "Historical Fresh-C ledger artifacts already exist; "
            "refuse overwrite."
        )

    manifest, ledger = build_historical_identity_sweep(
        root=root,
        source_commit=source_commit,
    )
    _atomic_json(
        manifest_path,
        manifest.model_dump(mode="json"),
    )
    _atomic_json(
        ledger_path,
        ledger.model_dump(mode="json"),
    )

    print("Fresh-C C0.1B historical exclusion ledger build")
    print(f"Source commit: {source_commit}")
    print(f"Scanned files: {manifest.scanned_file_count}")
    print(
        "Files with identities: "
        f"{manifest.files_with_identities}"
    )
    print(
        "Canonical historical identities: "
        f"{manifest.canonical_identity_count}"
    )
    print(f"Sweep ID: {manifest.sweep_id}")
    print(f"Sweep SHA256: {manifest.sweep_sha256}")
    print(f"Ledger ID: {ledger.ledger_id}")
    print(f"Ledger SHA256: {ledger.ledger_sha256}")
    print("Scientific content retained: False")
    print("Network calls: 0")
    print("LLM calls: 0")
    print("Fresh Reserve C consumed: False")
    print("STOP: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
