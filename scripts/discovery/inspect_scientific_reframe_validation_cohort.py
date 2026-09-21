from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.reframing.benchmark_audit import (
    ScientificReframeBenchmarkAuditReport,
)
from pipeline_core.discovery.reframing.generalization_validation import (
    ScientificReframeCalibrationFreeze,
    audit_validation_cohort,
)
from pipeline_core.discovery.reframing.selective_execution import load_json_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether a proposed holdout or cross-domain benchmark is uncontaminated "
            "and is being evaluated under exactly the frozen reframing semantics."
        )
    )
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument(
        "--role",
        required=True,
        choices=("holdout_same_domain", "validation_cross_domain"),
    )
    parser.add_argument("--domain-label", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    freeze = load_json_model(args.freeze.resolve(), ScientificReframeCalibrationFreeze)
    audit = load_json_model(args.audit.resolve(), ScientificReframeBenchmarkAuditReport)
    result = audit_validation_cohort(
        freeze=freeze,
        validation_audit=audit,
        cohort_role=args.role,
        validation_domain_label=args.domain_label,
        repository_root=args.repository_root,
    )
    output = args.output or Path(audit.benchmark_root) / "scientific_reframing_validation_cohort_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    print("Scientific reframing validation cohort inspection complete")
    print("LLM calls: 0")
    print("Role:", result.cohort_role)
    print("Calibration domain:", result.calibration_domain_label)
    print("Validation domain:", result.validation_domain_label)
    print("Validation tasks:", result.validation_task_count)
    print("Overlap detected:", str(result.overlap_detected).lower())
    print("Semantics unchanged:", str(result.semantics_unchanged).lower())
    print("Role/domain consistent:", str(result.role_domain_consistent).lower())
    print("Comparability:", result.comparability_status)
    print("Trigger/cohort validation evaluable:", str(result.validation_evaluable_without_contamination).lower())
    print("Model configuration equivalence verified: false")
    print("Full generation-quality comparison ready: false")
    if result.overlapping_case_keys:
        print("Overlapping cases:", ", ".join(result.overlapping_case_keys))
    print("Output:", output)
    return 0 if result.validation_evaluable_without_contamination else 2


if __name__ == "__main__":
    raise SystemExit(main())
