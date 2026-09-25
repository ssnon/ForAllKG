from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_llm import (
    InstructorOpenAICompatibleExternalNoveltyBackend,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.hypothesis_semantic_llm import (
    InstructorOpenAICompatibleSemanticCriticBackend,
)
from pipeline_core.discovery.hypothesis_semantic_runtime import (
    HypothesisSemanticCriticRuntime,
)
from pipeline_core.discovery.pre_n10_regeneration_reentry_v2 import (
    execute_pre_n10_regeneration_reentry_v2,
)
from pipeline_core.discovery.pre_n10_regeneration_v1 import (
    PreN10RegenerationExecutionReportV1,
)


def _write_json_exact_or_validate(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    raw = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(
                "existing write-once semantic artifact differs: "
                + str(path)
            )
        return
    with path.open("xb") as handle:
        handle.write(raw)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fresh pre-N10 regenerated hypotheses with the semantic "
            "critic, compile an explicit semantic scientific disposition, and "
            "only for disposition PASS create a fresh claim decomposition and "
            "re-enter the strict pre-N10 scientific contract gate. No retrieval, "
            "external novelty assessment, N9, N10, or second regeneration occurs."
        )
    )
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--regeneration-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--critic-model", required=True)
    parser.add_argument("--decomposition-model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-claims", type=int, default=4)
    parser.add_argument("--max-queries-per-claim", type=int, default=2)
    args = parser.parse_args()

    context = HypothesisContext.model_validate_json(
        args.context.expanduser().resolve().read_text(encoding="utf-8")
    )
    regeneration = PreN10RegenerationExecutionReportV1.model_validate_json(
        args.regeneration_report.expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    root = args.output_dir.expanduser().resolve()

    semantic_backends = {}
    decomposition_backends = {}

    def semantic_runner_factory(source_hypothesis_id: str, lineage_dir: Path):
        backend = InstructorOpenAICompatibleSemanticCriticBackend(
            model=args.critic_model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode="JSON",
            temperature=0.0,
            parse_retries=args.parse_retries,
            timeout=args.timeout_seconds,
            telemetry_path=lineage_dir / "semantic.telemetry.jsonl",
            telemetry_context={
                "pipeline": "pre_n10_regeneration_reentry_v2",
                "source_hypothesis_id": source_hypothesis_id,
                "source_regeneration_report_id": regeneration.report_id,
            },
        )
        semantic_backends[source_hypothesis_id] = backend
        return HypothesisSemanticCriticRuntime(backend)

    def decomposition_backend_factory(
        source_hypothesis_id: str,
        lineage_dir: Path,
    ):
        backend = InstructorOpenAICompatibleExternalNoveltyBackend(
            model=args.decomposition_model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode="JSON",
            temperature=0.0,
            parse_retries=args.parse_retries,
            timeout=args.timeout_seconds,
            capture_prompts=True,
            telemetry_path=lineage_dir / "decomposition.telemetry.jsonl",
            telemetry_context={
                "pipeline": "pre_n10_regeneration_reentry_v2",
                "stage": "claim_decomposition",
                "source_hypothesis_id": source_hypothesis_id,
                "source_regeneration_report_id": regeneration.report_id,
            },
        )
        decomposition_backends[source_hypothesis_id] = backend
        return backend

    report, outcomes = execute_pre_n10_regeneration_reentry_v2(
        context=context,
        regeneration_report=regeneration,
        semantic_runner_factory=semantic_runner_factory,
        decomposition_backend_factory=decomposition_backend_factory,
        output_root=root,
        max_claims=args.max_claims,
        max_queries_per_claim=args.max_queries_per_claim,
    )

    for row in report.lineages:
        lineage_dir = root / "lineage" / (
            row.source_hypothesis_id.replace(":", "_").replace("/", "_")
        )
        outcome = outcomes.get(row.source_hypothesis_id)
        if outcome is not None:
            _write_json_exact_or_validate(
                lineage_dir / "semantic.hard_evaluation.json",
                outcome.evaluation,
            )
            _write_json_exact_or_validate(
                lineage_dir / "semantic.run.json",
                outcome.run_record,
            )
            _write_json_exact_or_validate(
                lineage_dir / "semantic.reference_audit.json",
                outcome.reference_audit,
            )
            if outcome.generation is not None:
                _write_json_exact_or_validate(
                    lineage_dir / "semantic.draft.json",
                    outcome.generation.draft,
                )
            if outcome.sanitized_draft is not None:
                _write_json_exact_or_validate(
                    lineage_dir / "semantic.sanitized.draft.json",
                    outcome.sanitized_draft,
                )
            if outcome.review is not None:
                _write_json_exact_or_validate(
                    lineage_dir / "semantic.review.json",
                    outcome.review,
                )

        backend = decomposition_backends.get(row.source_hypothesis_id)
        if backend is not None:
            _write_json_exact_or_validate(
                lineage_dir / "decomposition.prompts.json",
                {
                    "schema_version":
                        "pre-n10-regeneration-decomposition-prompts-v2",
                    "source_hypothesis_id": row.source_hypothesis_id,
                    "records": [
                        asdict(record)
                        for record in backend.prompt_records
                    ],
                },
            )

    print("Pre-N10 regeneration semantic disposition + contract re-entry-v2 complete")
    print("Report:", report.report_id)
    print("Lineages:", report.lineage_count)
    print(
        "Semantic runtime/critic-LLM/review-valid/admissible/intervention/terminal:",
        report.semantic_runtime_invocation_count,
        "/",
        report.semantic_critic_llm_invocation_count,
        "/",
        report.semantic_review_valid_count,
        "/",
        report.semantic_admissible_count,
        "/",
        report.semantic_intervention_required_count,
        "/",
        report.semantic_terminal_count,
    )
    print(
        "Fresh claim-decomposition requests:",
        report.claim_decomposition_request_count,
    )
    print(
        "Ready for N10 / contract intervention-required:",
        report.ready_for_n10_count,
        "/",
        report.contract_intervention_required_count,
    )
    print()
    for row in report.lineages:
        print(
            row.source_hypothesis_id,
            "| semantic=",
            row.semantic_status,
            "| disposition=",
            row.semantic_disposition,
            "| final=",
            row.final_status,
            "| N10-ready=",
            row.ready_for_n10,
        )
        if row.semantic_disposition_path:
            print("  semantic disposition:", row.semantic_disposition_path)
        if row.semantic_failed_dimensions:
            print("  failed dimensions:", row.semantic_failed_dimensions)
        if row.semantic_warning_dimensions:
            print("  warning dimensions:", row.semantic_warning_dimensions)
        if row.query_plan_path:
            print("  query plan:", row.query_plan_path)
        if row.contract_report_path:
            print("  contract:", row.contract_report_path)

    print()
    print("Semantic fail blocks pre-N10: true")
    print("Semantic warning blocks pre-N10: false")
    print("Semantic disposition is final rejection authority: false")
    print("Semantic disposition is novelty authority: false")
    print("Retrieval performed: false")
    print("External novelty performed: false")
    print("N9/N10 performed: false")
    print("Second regeneration performed: false")
    print("Output:", root / "reentry_v2.report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
