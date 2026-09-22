from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.positive_nonobviousness_adjudication import (
    InstructorPositiveNonObviousnessBackend,
    build_positive_nonobviousness_adjudication_report,
)
from pipeline_core.discovery.positive_nonobviousness_basis import (
    PositiveNonObviousnessBasisReport,
)
from pipeline_core.discovery.scientific_claim_evidence_graph import (
    ScientificClaimEvidenceGraphReport,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Adjudicate qualifying positive non-obviousness basis candidates "
            "and apply a fail-closed bounded authority gate. Hypotheses without "
            "a qualifying/full-coverage basis are skipped deterministically."
        )
    )
    parser.add_argument(
        "--evidence-graph",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--basis",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
    )
    parser.add_argument(
        "--base-url",
        default=None,
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
    )
    args = parser.parse_args()

    graph = ScientificClaimEvidenceGraphReport.model_validate_json(
        args.evidence_graph.read_text(encoding="utf-8")
    )
    basis = PositiveNonObviousnessBasisReport.model_validate_json(
        args.basis.read_text(encoding="utf-8")
    )

    backend = InstructorPositiveNonObviousnessBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        capture_prompts=args.save_prompts,
    )

    report = build_positive_nonobviousness_adjudication_report(
        graph=graph,
        basis=basis,
        backend=backend,
    )

    report_path = args.output_prefix.with_suffix(
        ".review.json"
    )
    prompt_path = args.output_prefix.with_suffix(
        ".prompts.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.save_prompts:
        prompt_path.write_text(
            json.dumps(
                backend.prompt_records,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    print("Positive non-obviousness adjudication shadow complete")
    print("Report:", report.report_id)
    print("Hypotheses:", report.hypothesis_count)
    print(
        "Reviewed hypotheses:",
        report.reviewed_hypothesis_count,
    )
    print(
        "Deterministically skipped hypotheses:",
        report.skipped_hypothesis_count,
    )
    print("LLM calls:", report.llm_calls_performed)
    print(
        "Gate states:",
        report.gate_state_counts,
    )
    print(
        "Positive non-obviousness authorized:",
        report.positive_nonobviousness_authorized_count,
    )
    print(
        "Fatal contradiction blocked:",
        report.fatal_contradiction_blocked_count,
    )

    for decision in report.hypothesis_decisions:
        print()
        print("Hypothesis:", decision.hypothesis_id)
        print("Basis readiness:", decision.basis_readiness)
        print("Gate state:", decision.gate_state)
        print(
            "Eligible claims:",
            decision.eligible_claim_ids,
        )
        print(
            "Reviewed claims:",
            decision.reviewed_claim_ids,
        )
        print(
            "Positive support claims:",
            decision.positive_support_claim_ids,
        )
        print(
            "Fatal contradiction claims:",
            decision.fatal_contradiction_claim_ids,
        )
        print(
            "Direct prior-art blockers:",
            decision.direct_prior_art_blocker_claim_ids,
        )
        print(
            "Unclassified work remains:",
            decision.unclassified_presented_work_remains,
        )
        print(
            "LLM call performed:",
            decision.llm_call_performed,
        )
        if decision.deterministic_skip_reason:
            print(
                "Skip reason:",
                decision.deterministic_skip_reason,
            )

        rows = [
            row
            for row in report.claim_reviews
            if row.hypothesis_id == decision.hypothesis_id
        ]
        for row in rows:
            print(
                " ",
                row.claim_id,
                "| disposition=",
                row.compiled_disposition,
                "| state=",
                row.compiled_state,
                "| positive_authority=",
                row.positive_nonobviousness_authority,
                "| fatal_authority=",
                row.fatal_contradiction_authority,
                "| reasons=",
                row.deterministic_reason_codes,
            )

    print()
    print("Absence-based non-obviousness forbidden: true")
    print("Basis candidate automatically supportive: false")
    print("Counterevidence can be fatal rather than supportive: true")
    print("Partial coverage blocks authority: true")
    print("Direct prior art blocks authority: true")
    print("Novelty verdict created: false")
    print("Certification performed: false")
    print("Production authority created: false")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Production selection changed: false")
    print("Review report:", report_path)
    if args.save_prompts:
        print("Prompts:", prompt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
