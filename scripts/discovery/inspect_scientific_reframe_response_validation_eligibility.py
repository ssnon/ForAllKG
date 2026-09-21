from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.reframing.response_semantics_validation import (
    ScientificResponseSemanticsCandidateFreeze,
    inspect_response_semantics_validation_eligibility,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether a task remains untouched for the frozen response-semantics "
            "candidate profile. This performs no scientific trigger evaluation."
        )
    )
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--domain-label", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    freeze = ScientificResponseSemanticsCandidateFreeze.model_validate_json(
        freeze_path.read_text(encoding="utf-8")
    )
    report = inspect_response_semantics_validation_eligibility(
        freeze=freeze,
        validation_task_id=args.task_id,
        validation_domain_label=args.domain_label,
        repository_root=args.repository_root,
    )
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print("Scientific reframing response-semantics validation eligibility")
    print("LLM calls: 0")
    print("Task:", report.validation_task_id)
    print("Domain:", report.validation_domain_label)
    print("Status:", report.status)
    print("Adaptation overlap:", str(report.adaptation_overlap).lower())
    print("Candidate semantics unchanged:", str(report.candidate_semantics_unchanged).lower())
    print("Eligible as untouched validation:", str(report.eligible_as_untouched_validation).lower())
    if args.output:
        print("Output:", args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
