"""Read-only M4-C0 synthetic revision-policy exercise; no empirical evidence input."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.empirical_revision_policy_m4c0 import (
    build_policy_replay, render_report,
)


def run(*, confrontation: Path, expected_sha256: str, output_dir: Path) -> dict:
    repository = Path(__file__).resolve().parents[2]
    inp = confrontation.expanduser().resolve()
    dest = output_dir.expanduser().resolve()
    if dest == repository or repository in dest.parents:
        raise ValueError("output must be outside code repository")
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite output: {dest}")
    if not inp.is_file():
        raise FileNotFoundError(f"M4-A1 confrontation not found: {inp}")
    sha = hashlib.sha256(inp.read_bytes()).hexdigest()
    if sha != expected_sha256:
        raise ValueError("M4-C0 source SHA-256 mismatch")
    obj = json.loads(inp.read_text(encoding="utf-8"))
    replay = build_policy_replay(obj, source_sha256=sha)
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "M4C0_SYNTHETIC_POLICY_REPLAY_PRIVATE.json").write_text(
        json.dumps(replay, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (dest / "M4C0_REPORT_PRIVATE.md").write_text(render_report(replay), encoding="utf-8")
    return replay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m4a1-report", required=True, type=Path)
    parser.add_argument("--expected-m4a1-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(confrontation=args.m4a1_report, expected_sha256=args.expected_m4a1_sha256, output_dir=args.output_dir)
    except (ValueError, FileNotFoundError, FileExistsError, json.JSONDecodeError) as e:
        parser.exit(2, f"M4C0 STOPPED (source unmodified): {e}\n")
    print("M4C0:", result["status"])
    print("cases:", result["case_count"], "synthetic scenarios:", result["policy_row_count"])
    print("scientific learning certified:", str(result["scientific_learning_demonstrated"]).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
