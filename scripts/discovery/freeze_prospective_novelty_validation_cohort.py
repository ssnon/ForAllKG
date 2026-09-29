from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pipeline_core.discovery.prospective_novelty_validation_cohort import (
    ProspectiveNoveltyValidationCohortSpec,
    build_prospective_novelty_validation_cohort_freeze,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze an outcome-blind prospective cohort for validating the "
            "novelty-improvement shadow stack. The freeze consumes only task "
            "definitions and pre-registered evaluation focus; no generated "
            "hypothesis, prior-art, conceptual-knownness, Alpha6, diversity, "
            "or research-value outcome may influence case selection."
        )
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    spec_path = args.spec.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not spec_path.is_file():
        raise ValueError("missing cohort spec: " + str(spec_path))
    if output.exists():
        raise ValueError(
            "prospective novelty cohort freeze is write-once; use a fresh output path"
        )

    spec = ProspectiveNoveltyValidationCohortSpec.model_validate_json(
        spec_path.read_text(encoding="utf-8")
    )

    freeze = build_prospective_novelty_validation_cohort_freeze(
        spec=spec,
        source_spec_sha256=_sha256_file(spec_path),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        freeze.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Prospective novelty-validation cohort frozen")
    print("Freeze:", freeze.freeze_id)
    print("Cases:", freeze.frozen_case_count)
    print("Domains:", freeze.domain_counts)
    print("Roles:", freeze.role_counts)
    print()
    for row in freeze.frozen_cases:
        print(
            row.prospective_case_id,
            "|",
            row.source_case_id,
            "|",
            row.domain_profile_id,
            "|",
            row.case_role,
        )
        print(" question:", row.question)
        print(" focus:", ", ".join(row.evaluation_focus))
        print(
            " research-value capability expected:",
            row.research_value_capability_expected,
        )
    print()
    print("Prior results observed for selection: false")
    print("External novelty outcomes used for selection: false")
    print("Conceptual knownness outcomes used for selection: false")
    print("Alpha6 recommendations used for selection: false")
    print("Research-value outcomes used for selection: false")
    print("Post-freeze case mutation allowed: false")
    print("Production selection authority: false")
    print("Artifact:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
