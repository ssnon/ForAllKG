"""M4-C1 read-only, opt-in evidence-event template + source file SHA checks.

Does NOT read numerical observations or establish scientific evidence. Outputs
private metadata under a new directory outside the Git repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.empirical_evidence_event_m4c1 import (
    _digest, audit_event_template, make_event_template, render_report,
    validate_source,
)


def run(*, m4a1: Path, expected_m4a1_sha256: str, m4c0: Path,
        expected_m4c0_sha256: str, output_dir: Path,
        events: Path | None = None, evidence_root: Path | None = None) -> dict:
    repository = Path(__file__).resolve().parents[2]
    output = output_dir.expanduser().resolve()
    if output == repository or repository in output.parents:
        raise ValueError("private output must stay outside repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    inputs = (m4a1, m4c0) + ((events,) if events else ())
    for p in inputs:
        if not p.expanduser().is_file():
            raise FileNotFoundError(f"input not found: {p}")
        resolved = p.expanduser().resolve()
        if output == resolved or output in resolved.parents:
            raise ValueError("output cannot contain an input")
    s1 = _digest(m4a1.expanduser())
    s0 = _digest(m4c0.expanduser())
    if s1 != expected_m4a1_sha256 or s0 != expected_m4c0_sha256:
        raise ValueError("pinned SHA-256 mismatch; input not modified")
    with m4a1.expanduser().open(encoding="utf-8") as f:
        a1 = json.load(f)
    with m4c0.expanduser().open(encoding="utf-8") as f:
        c0 = json.load(f)
    cases = validate_source(a1, c0, s1)
    if events:
        if evidence_root is None or not evidence_root.is_dir():
            raise ValueError("--evidence-root existing directory is required with --events")
        with events.expanduser().open(encoding="utf-8") as f:
            template = json.load(f)
    else:
        if evidence_root is not None:
            raise ValueError("--evidence-root only makes sense with --events")
        template = make_event_template(cases, m4a1_sha=s1, m4c0_sha=s0)
    audit = audit_event_template(template, cases, root=evidence_root or Path("/nonexistent"),
                                 m4a1_sha=s1, m4c0_sha=s0)
    output.mkdir(parents=True, exist_ok=False)
    if not events:
        (output / "M4C1_EVENT_TEMPLATE_PRIVATE.json").write_text(
            json.dumps(template, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "M4C1_EVENT_AUDIT_PRIVATE.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "M4C1_REPORT_PRIVATE.md").write_text(render_report(audit), encoding="utf-8")
    return audit


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m4a1-report", type=Path, required=True)
    p.add_argument("--expected-m4a1-sha256", required=True)
    p.add_argument("--m4c0-replay", type=Path, required=True)
    p.add_argument("--expected-m4c0-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--events", type=Path)
    p.add_argument("--evidence-root", type=Path)
    a = p.parse_args()
    try:
        report = run(m4a1=a.m4a1_report, expected_m4a1_sha256=a.expected_m4a1_sha256,
                     m4c0=a.m4c0_replay, expected_m4c0_sha256=a.expected_m4c0_sha256,
                     output_dir=a.output_dir, events=a.events, evidence_root=a.evidence_root)
    except (ValueError, FileNotFoundError, FileExistsError, json.JSONDecodeError, OSError) as e:
        p.exit(2, f"M4C1 STOPPED (source unmodified): {e}\n")
    print("M4C1:", report["status"])
    print("event template/intake rows:", report["submitted_event_count"], "/", report["expected_event_count"])
    print("scientific adjudication authorized:", str(report["scientific_truth_or_falsification_authority"]).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
