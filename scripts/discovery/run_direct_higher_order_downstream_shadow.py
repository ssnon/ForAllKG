from __future__ import annotations

import hashlib

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_core.discovery.direct_higher_order_shadow_bundle import (
    DirectHigherOrderExternalNoveltyDisposition,
    DirectHigherOrderSemanticDisposition,
    DirectHigherOrderShadowArmBundle,
    build_direct_higher_order_structural_views,
    finalize_direct_higher_order_shadow_bundle,
)
from pipeline_core.discovery.direct_higher_order_hypothesis_runtime import (
    materialize_direct_higher_order_hypothesis_context,
)
from pipeline_core.discovery.direct_higher_order_synthesis_context import (
    DirectHigherOrderSynthesisContext,
)
from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.hypothesis_semantic_llm import (
    InstructorOpenAICompatibleSemanticCriticBackend,
)
from pipeline_core.discovery.hypothesis_semantic_runtime import (
    HypothesisSemanticCriticRuntime,
)

def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()




def _json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _semantic_artifacts(
    *,
    prefix: Path,
    outcome,
) -> None:
    _write(
        prefix.with_suffix(".hard_evaluation.json"),
        outcome.evaluation,
    )
    _write(
        prefix.with_suffix(".run.json"),
        outcome.run_record,
    )
    _write(
        prefix.with_suffix(".reference_audit.json"),
        outcome.reference_audit,
    )

    if outcome.review is not None:
        _write(
            prefix.with_suffix(".review.json"),
            outcome.review,
        )

    if outcome.sanitized_draft is not None:
        _write(
            prefix.with_suffix(".sanitized.draft.json"),
            outcome.sanitized_draft,
        )


def _run_external_novelty(
    *,
    portfolio_path: Path,
    domain_profile: str,
    model: str,
    base_url: str | None,
    api_key_env: str,
    provider_plan: Path | None,
    results_per_query: int,
    output_prefix: Path,
) -> tuple[
    DirectHigherOrderExternalNoveltyDisposition,
    str,
]:
    command = [
        sys.executable,
        "-m",
        "scripts.discovery.run_external_novelty",
        "--portfolio",
        str(portfolio_path),
        "--domain-profile",
        domain_profile,
        "--model",
        model,
        "--api-key-env",
        api_key_env,
        "--results-per-query",
        str(results_per_query),
        "--output-prefix",
        str(output_prefix),
        "--pre-review-coverage-shadow",
    ]

    if base_url:
        command.extend(
            [
                "--base-url",
                base_url,
            ]
        )

    if provider_plan is not None:
        command.extend(
            [
                "--provider-plan",
                str(provider_plan),
            ]
        )

    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )

    log_path = output_prefix.with_suffix(
        ".subprocess.log.txt"
    )
    log_path.write_text(
        "COMMAND\n=======\n"
        + " ".join(command)
        + "\n\nSTDOUT\n======\n"
        + completed.stdout
        + "\n\nSTDERR\n======\n"
        + completed.stderr,
        encoding="utf-8",
    )

    report_path = output_prefix.with_suffix(
        ".report.json"
    )

    if (
        completed.returncode != 0
        or not report_path.is_file()
    ):
        message = (
            f"external novelty subprocess failed "
            f"returncode={completed.returncode}; "
            f"log={log_path}"
        )

        return (
            DirectHigherOrderExternalNoveltyDisposition(
                status="ERROR",
                error=message,
            ),
            message,
        )

    report = ExternalNoveltyReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )

    query_path = output_prefix.with_suffix(
        ".claims_queries.json"
    )
    prior_path = output_prefix.with_suffix(
        ".prior_art.json"
    )

    return (
        DirectHigherOrderExternalNoveltyDisposition(
            status="COMPLETED",
            report_path=str(report_path),
            query_plan_path=(
                str(query_path)
                if query_path.is_file()
                else None
            ),
            prior_art_path=(
                str(prior_path)
                if prior_path.is_file()
                else None
            ),
            card_statuses=[
                str(card.status)
                for card in report.cards
            ],
        ),
        "",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Non-blocking semantic + external-novelty downstream closure "
            "for the direct higher-order shadow generation lane."
        )
    )
    parser.add_argument(
        "--generation-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--source-context",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--synthesis-context-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--domain-profile",
        required=True,
    )
    parser.add_argument(
        "--semantic-model",
        required=True,
    )
    parser.add_argument(
        "--novelty-model",
        required=True,
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
    )
    parser.add_argument(
        "--semantic-parse-retries",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--provider-plan",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=12,
    )
    args = parser.parse_args()

    generation = _json(
        args.generation_report
    )

    if (
        generation.get("schema_version")
        != "direct-higher-order-shadow-lane-generation-v1"
    ):
        raise RuntimeError(
            "unexpected direct-HO generation report schema"
        )

    source_context = (
        HypothesisContext.model_validate_json(
            args.source_context.read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        source_context.domain_profile_id
        != args.domain_profile
    ):
        raise RuntimeError(
            "source context / domain profile mismatch"
        )

    synthesis_payload = _json(
        args.synthesis_context_report
    )

    contexts = {
        row.context_id: row
        for row in (
            DirectHigherOrderSynthesisContext.model_validate(
                item
            )
            for item in synthesis_payload.get(
                "contexts",
                [],
            )
        )
    }

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    semantic_backend = (
        InstructorOpenAICompatibleSemanticCriticBackend(
            model=args.semantic_model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            temperature=0.0,
            parse_retries=args.semantic_parse_retries,
            telemetry_path=(
                out
                / "semantic.telemetry.jsonl"
            ),
            telemetry_context={
                "pipeline":
                    "direct_higher_order_downstream_shadow",
            },
        )
    )

    semantic_runtime = (
        HypothesisSemanticCriticRuntime(
            semantic_backend
        )
    )

    arm_bundles = []
    errors = []

    for arm in generation.get(
        "arms",
        [],
    ):
        arm_index = int(
            arm["arm_index"]
        )
        direct_context_id = str(
            arm["direct_context_id"]
        )
        direct_topology_id = str(
            arm["direct_topology_id"]
        )
        modifier_component_id = str(
            arm["modifier_component_id"]
        )
        modifier_text = str(
            arm["modifier_text"]
        )
        generation_status = str(
            arm["status"]
        )

        portfolio_payload = arm.get(
            "accepted_portfolio"
        )

        if (
            generation_status != "proposed"
            or not isinstance(
                portfolio_payload,
                dict,
            )
        ):
            arm_bundles.append(
                DirectHigherOrderShadowArmBundle(
                    arm_index=arm_index,
                    direct_context_id=direct_context_id,
                    direct_topology_id=direct_topology_id,
                    modifier_component_id=modifier_component_id,
                    modifier_text=modifier_text,
                    generation_status=generation_status,
                    semantic=DirectHigherOrderSemanticDisposition(
                        status="NOT_RUN",
                    ),
                    external_novelty=DirectHigherOrderExternalNoveltyDisposition(
                        status="NOT_RUN",
                    ),
                )
            )
            continue

        direct_context = contexts.get(
            direct_context_id
        )

        if direct_context is None:
            error = (
                "missing synthesis context for arm "
                + str(arm_index)
                + ": "
                + direct_context_id
            )
            errors.append(error)

            arm_bundles.append(
                DirectHigherOrderShadowArmBundle(
                    arm_index=arm_index,
                    direct_context_id=direct_context_id,
                    direct_topology_id=direct_topology_id,
                    modifier_component_id=modifier_component_id,
                    modifier_text=modifier_text,
                    generation_status=generation_status,
                    semantic=DirectHigherOrderSemanticDisposition(
                        status="ERROR",
                        error=error,
                    ),
                    external_novelty=DirectHigherOrderExternalNoveltyDisposition(
                        status="NOT_RUN",
                    ),
                )
            )
            continue

        portfolio = HypothesisPortfolio.model_validate(
            portfolio_payload
        )

        projection = (
            materialize_direct_higher_order_hypothesis_context(
                source_context=source_context,
                direct_context=direct_context,
            )
        )

        arm_dir = (
            out
            / f"arm_{arm_index:02d}"
        )
        arm_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        portfolio_path = (
            arm_dir
            / "portfolio.json"
        )
        context_path = (
            arm_dir
            / "derived_context.json"
        )

        _write(
            portfolio_path,
            portfolio,
        )
        _write(
            context_path,
            projection.context,
        )

        structural_views = list(
            build_direct_higher_order_structural_views(
                direct_context=direct_context,
                portfolio=portfolio,
            )
        )

        semantic_prefix = (
            arm_dir
            / "semantic"
        )

        try:
            semantic_outcome = (
                semantic_runtime.run(
                    projection.context,
                    portfolio,
                )
            )

            _semantic_artifacts(
                prefix=semantic_prefix,
                outcome=semantic_outcome,
            )

            semantic = (
                DirectHigherOrderSemanticDisposition(
                    status=(
                        "ACCEPTED"
                        if semantic_outcome.accepted
                        else "REJECTED"
                    ),
                    hard_gate_passed=(
                        semantic_outcome
                        .evaluation
                        .hard_gate_passed
                    ),
                    review_id=(
                        semantic_outcome.review.review_id
                        if semantic_outcome.review
                        is not None
                        else None
                    ),
                    run_id=(
                        semantic_outcome
                        .run_record
                        .run_id
                    ),
                    failure_stage=(
                        semantic_outcome
                        .run_record
                        .failure_stage
                    ),
                    artifact_prefix=str(
                        semantic_prefix
                    ),
                )
            )
        except Exception as exc:
            message = (
                type(exc).__name__
                + ": "
                + str(exc)
            )
            errors.append(
                f"arm {arm_index} semantic: {message}"
            )

            semantic = (
                DirectHigherOrderSemanticDisposition(
                    status="ERROR",
                    error=message,
                    artifact_prefix=str(
                        semantic_prefix
                    ),
                )
            )

        if semantic.status == "ACCEPTED":
            novelty_prefix = (
                arm_dir
                / "external_novelty"
            )

            novelty, novelty_error = (
                _run_external_novelty(
                    portfolio_path=portfolio_path,
                    domain_profile=args.domain_profile,
                    model=args.novelty_model,
                    base_url=args.base_url,
                    api_key_env=args.api_key_env,
                    provider_plan=args.provider_plan,
                    results_per_query=args.results_per_query,
                    output_prefix=novelty_prefix,
                )
            )

            if novelty_error:
                errors.append(
                    f"arm {arm_index} novelty: "
                    + novelty_error
                )
        else:
            novelty = (
                DirectHigherOrderExternalNoveltyDisposition(
                    status="SKIPPED_SEMANTIC_NOT_ACCEPTED"
                )
            )

        arm_bundles.append(
            DirectHigherOrderShadowArmBundle(
                arm_index=arm_index,
                direct_context_id=direct_context_id,
                direct_topology_id=direct_topology_id,
                modifier_component_id=modifier_component_id,
                modifier_text=modifier_text,
                generation_status=generation_status,
                accepted_portfolio_id=(
                    portfolio.portfolio_id
                ),
                accepted_portfolio_sha256=(
                    _sha256_json(portfolio)
                ),
                portfolio_artifact=str(
                    portfolio_path
                ),
                derived_context_artifact=str(
                    context_path
                ),
                semantic=semantic,
                external_novelty=novelty,
                structural_views=structural_views,
            )
        )

    bundle = (
        finalize_direct_higher_order_shadow_bundle(
            source_generation_report=str(
                args.generation_report
            ),
            source_context_id=(
                source_context.context_id
            ),
            source_context_sha256=(
                source_context.context_sha256
            ),
            source_direct_relationpattern_report_id=str(
                generation.get(
                    "direct_relationpattern_report_id",
                    "",
                )
            ),
            domain_profile_id=args.domain_profile,
            arms=arm_bundles,
        )
    )

    bundle_path = (
        out
        / "direct_higher_order.shadow_bundle.json"
    )
    _write(
        bundle_path,
        bundle,
    )

    summary = {
        "schema_version":
            "direct-higher-order-downstream-shadow-report-v1",
        "bundle":
            str(bundle_path),
        "bundle_id":
            bundle.bundle_id,
        "semantic_attempted_count":
            bundle.semantic_attempted_count,
        "semantic_accepted_count":
            bundle.semantic_accepted_count,
        "external_novelty_completed_count":
            bundle.external_novelty_completed_count,
        "structural_view_count":
            bundle.structural_view_count,
        "conceptual_knownness_integrated":
            False,
        "errors":
            errors,
        "authority": {
            "shadow_only":
                True,
            "candidate_survival_authority":
                False,
            "semantic_rejection_authority":
                False,
            "novelty_authority_created":
                False,
            "positive_premise_authority_created":
                False,
            "production_selection_authority":
                False,
            "stage8_input_changed":
                False,
        },
    }

    report_path = (
        out
        / "downstream_report.json"
    )
    _write(
        report_path,
        summary,
    )

    print(
        "=== DIRECT HIGHER-ORDER DOWNSTREAM SHADOW ==="
    )
    print(
        "bundle:",
        bundle.bundle_id,
    )
    print(
        "semantic accepted:",
        bundle.semantic_accepted_count,
        "/",
        bundle.semantic_attempted_count,
    )
    print(
        "external novelty completed:",
        bundle.external_novelty_completed_count,
    )
    print(
        "structural views:",
        bundle.structural_view_count,
    )
    print(
        "conceptual knownness integrated:",
        False,
    )
    print(
        "non-blocking errors:",
        len(errors),
    )
    print(
        "artifact:",
        report_path,
    )
    print(
        "DIRECT_HO_DOWNSTREAM_SHADOW_COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
