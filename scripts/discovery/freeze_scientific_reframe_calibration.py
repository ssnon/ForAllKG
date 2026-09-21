from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.reframing.benchmark_audit import (
    ScientificReframeBenchmarkAuditReport,
)
from pipeline_core.discovery.reframing.benchmark_synthesis import (
    ScientificReframeBenchmarkSynthesis,
)
from pipeline_core.discovery.reframing.generalization_validation import (
    build_calibration_freeze,
)
from pipeline_core.discovery.reframing.selective_execution import load_json_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a scientific-reframing calibration cohort and the current "
            "trigger/generation/critic semantics before untouched validation."
        )
    )
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--synthesis", required=True, type=Path)
    parser.add_argument("--domain-label", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    audit = load_json_model(args.audit.resolve(), ScientificReframeBenchmarkAuditReport)
    synthesis = load_json_model(args.synthesis.resolve(), ScientificReframeBenchmarkSynthesis)
    freeze = build_calibration_freeze(
        audit=audit,
        synthesis=synthesis,
        calibration_domain_label=args.domain_label,
        repository_root=args.repository_root,
    )
    output = args.output or Path(audit.benchmark_root) / "scientific_reframing_calibration_freeze.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(freeze.model_dump_json(indent=2), encoding="utf-8")

    print("Scientific reframing calibration freeze complete")
    print("LLM calls: 0")
    print("Domain:", freeze.calibration_domain_label)
    print("Frozen tasks:", freeze.calibration_task_count)
    print("Frozen cases:", len(freeze.case_keys))
    print("Semantics fingerprint:", freeze.semantics_fingerprint)
    print("Future holdout overlap prohibited: true")
    print("Semantic drift invalidates direct comparison: true")
    print("Model configuration fingerprinted: false")
    print("Direct generation-quality comparison requires the same model configuration: true")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
