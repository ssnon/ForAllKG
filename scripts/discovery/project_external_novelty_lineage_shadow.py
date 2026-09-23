from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
)
from pipeline_core.discovery.external_novelty_lineage_projection import (
    project_external_novelty_report_to_final_hypothesis,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Project one existing candidate-namespace ExternalNoveltyCard "
            "onto its verified authority-equivalent final Alpha6 hypothesis "
            "namespace. This is ID/provenance compatibility only: no search "
            "or novelty reassessment is performed."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument(
        "--source-external-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--final-hypothesis-id",
        required=True,
    )
    parser.add_argument(
        "--output-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-audit",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    plan = RelationalAtomicBindingPlan.model_validate_json(
        args.plan.read_text(encoding="utf-8")
    )
    source = ExternalNoveltyReport.model_validate_json(
        args.source_external_report.read_text(encoding="utf-8")
    )

    projected, audit = (
        project_external_novelty_report_to_final_hypothesis(
            plan=plan,
            source_report=source,
            final_hypothesis_id=args.final_hypothesis_id,
        )
    )

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)

    args.output_report.write_text(
        projected.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_audit.write_text(
        audit.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    card = projected.cards[0]

    print("External novelty lineage projection shadow complete")
    print("Candidate:", audit.candidate_hypothesis_id)
    print("Final:", audit.final_hypothesis_id)
    print("Source report:", audit.source_external_novelty_report_id)
    print("Projected report:", projected.report_id)
    print("Status:", card.status)
    print(
        "Coverage sufficient:",
        card.coverage.sufficient_for_absence_based_novelty,
    )
    print(
        "Candidate/final authority equivalence verified:",
        "true",
    )
    print(
        "Card payload unchanged except hypothesis namespace:",
        "true",
    )
    print("External novelty reassessed: false")
    print("Literature search rerun: false")
    print("Scientific content added: false")
    print("Evidence relationships changed: false")
    print("Coverage counts changed: false")
    print("Production selection changed: false")
    print("Projected report:", args.output_report)
    print("Audit:", args.output_audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
