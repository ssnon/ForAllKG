from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    PriorArtPacket,
)
from pipeline_core.discovery.scientific_distinctiveness_contracts import (
    ScientificDistinctivenessReport,
)
from pipeline_core.discovery.semantic_distinctiveness import (
    compile_semantic_distinctiveness_review,
)
from pipeline_core.discovery.semantic_distinctiveness_llm import (
    OpenRouterSemanticDistinctivenessBackend,
)
from pipeline_core.discovery.semantic_distinctiveness_prompt import (
    SemanticDistinctivenessPromptAssembler,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one diagnostic-only semantic scientific-"
            "distinctiveness review over frozen external "
            "prior-art evidence."
        )
    )

    parser.add_argument(
        "--scientific-report",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--external-report",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--external-prior-art",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--hypothesis-id",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--model",
        default="openai/gpt-5.6-luna",
    )

    parser.add_argument(
        "--provider",
        default=None,
    )

    parser.add_argument(
        "--review-pass-index",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--reasoning-effort",
        default="medium",
    )

    parser.add_argument(
        "--telemetry-path",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()



def _sanitize_reference_contract_draft(
    *,
    draft,
    prompt,
):
    """Fail closed on residual hallucinated provenance references.

    This function never guesses or maps one identifier to another.
    Any claim/work reference outside the frozen prompt allowlist is
    removed, and only the affected semantic dimension is downgraded to
    INDETERMINATE.
    """

    allowed_claim_ids = set(
        prompt.allowed_claim_ids
    )
    allowed_work_ids = set(
        prompt.allowed_work_ids
    )

    draft_updates = {}
    issues = []

    model_fields = getattr(
        draft.__class__,
        "model_fields",
        {},
    )

    for field_name in model_fields:
        assessment = getattr(
            draft,
            field_name,
            None,
        )

        if assessment is None:
            continue

        if not (
            hasattr(assessment, "claim_ids")
            and
            hasattr(assessment, "work_ids")
            and
            hasattr(assessment, "model_copy")
        ):
            continue

        original_claim_ids = list(
            assessment.claim_ids
        )

        original_work_ids = list(
            assessment.work_ids
        )

        valid_claim_ids = [
            value
            for value in original_claim_ids
            if value in allowed_claim_ids
        ]

        valid_work_ids = [
            value
            for value in original_work_ids
            if value in allowed_work_ids
        ]

        unknown_claim_ids = sorted(
            set(original_claim_ids)
            - allowed_claim_ids
        )

        unknown_work_ids = sorted(
            set(original_work_ids)
            - allowed_work_ids
        )

        if not (
            unknown_claim_ids
            or
            unknown_work_ids
        ):
            continue

        updates = {
            "claim_ids":
                valid_claim_ids,

            "work_ids":
                valid_work_ids,
        }

        if hasattr(
            assessment,
            "level",
        ):
            updates[
                "level"
            ] = "INDETERMINATE"

        if hasattr(
            assessment,
            "rationale",
        ):
            updates[
                "rationale"
            ] = (
                "INDETERMINATE under the frozen evidence: "
                "the generated assessment contained "
                "out-of-allowlist provenance references. "
                "No identifier substitution or fuzzy correction "
                "was performed."
            )

        draft_updates[
            field_name
        ] = assessment.model_copy(
            update=updates
        )

        issues.append(
            {
                "dimension":
                    field_name,

                "unknown_claim_ids":
                    unknown_claim_ids,

                "unknown_work_ids":
                    unknown_work_ids,
            }
        )

    if not draft_updates:
        return draft, []

    sanitized = draft.model_copy(
        update=draft_updates
    )

    # Revalidate the full object rather than trusting model_copy updates.
    sanitized = (
        draft.__class__
        .model_validate(
            sanitized.model_dump(
                mode="python"
            )
        )
    )

    return sanitized, issues


def main() -> int:
    args = parse_args()

    scientific = (
        ScientificDistinctivenessReport
        .model_validate_json(
            args.scientific_report.read_text(
                encoding="utf-8"
            )
        )
    )

    external = (
        ExternalNoveltyReport
        .model_validate_json(
            args.external_report.read_text(
                encoding="utf-8"
            )
        )
    )

    packet = (
        PriorArtPacket
        .model_validate_json(
            args.external_prior_art.read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        scientific
        .source_external_novelty_report_id
        != external.report_id
    ):
        raise ValueError(
            "scientific/external report provenance mismatch"
        )

    if (
        scientific
        .source_prior_art_packet_id
        != packet.packet_id
    ):
        raise ValueError(
            "scientific/prior-art provenance mismatch"
        )

    scientific_by_id = {
        row.hypothesis_id:
            row
        for row in scientific.reviews
    }

    external_by_id = {
        row.hypothesis_id:
            row
        for row in external.cards
    }

    hypothesis_id = str(
        args.hypothesis_id
    )

    if hypothesis_id not in scientific_by_id:
        raise ValueError(
            "hypothesis absent from scientific report: "
            f"{hypothesis_id}"
        )

    if hypothesis_id not in external_by_id:
        raise ValueError(
            "hypothesis absent from external report: "
            f"{hypothesis_id}"
        )

    scientific_review = (
        scientific_by_id[
            hypothesis_id
        ]
    )

    external_card = (
        external_by_id[
            hypothesis_id
        ]
    )

    prompt = (
        SemanticDistinctivenessPromptAssembler()
        .build(
            scientific_review,
            external_card,
            packet,
        )
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.dry_run:
        body = {
            "dry_run":
                True,

            "hypothesis_id":
                hypothesis_id,

            "prompt_version":
                prompt.prompt_version,

            "prompt_sha256":
                prompt.prompt_sha256,

            "system_prompt_chars":
                len(
                    prompt.system_prompt
                ),

            "user_prompt_chars":
                len(
                    prompt.user_prompt
                ),

            "allowed_claim_ids":
                list(
                    prompt
                    .allowed_claim_ids
                ),

            "allowed_work_ids":
                list(
                    prompt
                    .allowed_work_ids
                ),

            "retrieval_performed":
                False,

            "model_review_performed":
                False,

            "action_policy_applied":
                False,

            "scientific_selection_changed":
                False,
        }

        args.output.write_text(
            json.dumps(
                body,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "Semantic distinctiveness prompt dry-run complete"
        )

        print(
            "Hypothesis:",
            hypothesis_id,
        )

        print(
            "Prompt SHA:",
            prompt.prompt_sha256,
        )

        print(
            "Claims:",
            len(
                prompt.allowed_claim_ids
            ),
        )

        print(
            "Works:",
            len(
                prompt.allowed_work_ids
            ),
        )

        print(
            "User prompt chars:",
            len(
                prompt.user_prompt
            ),
        )

        print(
            "MODEL_REVIEW_PERFORMED=False"
        )

        return 0

    if args.review_pass_index < 1:
        raise ValueError(
            "review-pass-index must be >= 1"
        )

    backend = (
        OpenRouterSemanticDistinctivenessBackend(
            model=args.model,
            provider=args.provider,
            temperature=args.temperature,
            reasoning_effort=(
                args.reasoning_effort
            ),
            telemetry_path=(
                args.telemetry_path
            ),
            default_debug_path=(
                args.output.with_suffix(
                    ".raw_response.json"
                )
            ),
        )
    )

    generation = (
        backend.review(
            prompt,
            review_pass_index=(
                args.review_pass_index
            ),
            debug_path=(
                args.output.with_suffix(
                    ".raw_response.json"
                )
            ),
        )
    )

    active_prompt = prompt

    repair_count = 0
    repair_issues: list[str] = []

    try:
        result = (
            compile_semantic_distinctiveness_review(
                scientific_report=
                    scientific,

                scientific_review=
                    scientific_review,

                prompt=
                    active_prompt,

                draft=
                    generation.draft,

                backend_name=
                    backend.backend_name,

                requested_model=
                    generation.requested_model,

                served_model=
                    generation.served_model,

                review_pass_index=
                    args.review_pass_index,

                reference_contract_repair_count=
                    repair_count,

                reference_contract_repair_issues=
                    repair_issues,
            )
        )

    except ValueError as exc:
        issue = str(
            exc
        )

        repairable_prefixes = (
            "semantic dimension references unknown work IDs:",
            "semantic dimension references unknown claim IDs:",
        )

        if not issue.startswith(
            repairable_prefixes
        ):
            raise

        repair_count = 1
        repair_issues = [
            issue
        ]

        repair_prompt = (
            SemanticDistinctivenessPromptAssembler()
            .build_reference_validation_repair(
                original_prompt=
                    prompt,

                previous_draft=
                    generation.draft,

                issues=
                    repair_issues,
            )
        )

        print(
            "REFERENCE_CONTRACT_REPAIR_TRIGGERED=True"
        )

        print(
            "REFERENCE_CONTRACT_REPAIR_ISSUES =",
            repair_issues,
        )

        generation = (
            backend.review(
                repair_prompt,
                review_pass_index=(
                    args.review_pass_index
                ),
                debug_path=(
                    args.output.with_suffix(
                        ".repair_raw_response.json"
                    )
                ),
            )
        )

        active_prompt = repair_prompt

        final_draft, deterministic_reference_issues = (
            _sanitize_reference_contract_draft(
                draft=generation.draft,
                prompt=active_prompt,
            )
        )

        if deterministic_reference_issues:
            print(
                "REFERENCE_CONTRACT_DETERMINISTIC_SANITIZE_USED=True"
            )
            print(
                "REFERENCE_CONTRACT_DETERMINISTIC_SANITIZE_ISSUES=",
                deterministic_reference_issues,
            )

            repair_issues.extend(
                [
                    (
                        "deterministic_fail_closed:"
                        + str(value)
                    )
                    for value
                    in deterministic_reference_issues
                ]
            )

        result = (
            compile_semantic_distinctiveness_review(
                scientific_report=
                    scientific,

                scientific_review=
                    scientific_review,

                prompt=
                    active_prompt,

                draft=
                    final_draft,

                backend_name=
                    backend.backend_name,

                requested_model=
                    generation.requested_model,

                served_model=
                    generation.served_model,

                review_pass_index=
                    args.review_pass_index,

                reference_contract_repair_count=
                    repair_count,

                reference_contract_repair_issues=
                    repair_issues,
            )
        )

    print(
        "REFERENCE_CONTRACT_REPAIR_USED=",
        bool(
            repair_count
        ),
    )

    args.output.write_text(
        json.dumps(
            result.model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Semantic distinctiveness review complete"
    )

    print(
        "Hypothesis:",
        result.hypothesis_id,
    )

    print(
        "Evidence pattern:",
        result.source_evidence_pattern,
    )

    print(
        "Overall tier:",
        result.overall_tier,
    )

    print(
        "Confidence:",
        result.confidence,
    )

    print(
        "Pass:",
        result.review_pass_index,
    )

    print(
        "Reference contract repair count:",
        result.reference_contract_repair_count,
    )

    print(
        "Requested model:",
        result.requested_model,
    )

    print(
        "Served model:",
        result.served_model,
    )

    print(
        "RETRIEVAL_PERFORMED=False"
    )

    print(
        "ACTION_POLICY_APPLIED=False"
    )

    print(
        "SCIENTIFIC_SELECTION_CHANGED=False"
    )

    print(
        "Saved:",
        args.output,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
