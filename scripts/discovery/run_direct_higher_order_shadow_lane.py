from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pipeline_core.discovery.direct_backbone_modifier_eligibility import (
    screen_direct_backbone_candidate_modifiers,
)
from pipeline_core.discovery.direct_higher_order_hypothesis_runtime import (
    DirectHigherOrderShadowHypothesisRuntime,
    authorize_direct_higher_order_shadow_generation,
    materialize_direct_higher_order_hypothesis_context,
)
from pipeline_core.discovery.direct_higher_order_synthesis_context import (
    build_direct_higher_order_synthesis_contexts,
    render_direct_higher_order_shadow_prompt,
)
from pipeline_core.discovery.direct_higher_order_topology import (
    compose_direct_higher_order_topologies,
)
from pipeline_core.discovery.direct_relationpattern_task_shadow import (
    DirectRelationPatternAssessment,
    DirectRelationPatternCandidate,
    DirectRelationPatternTaskShadowReport,
)
from pipeline_core.discovery.direct_task_relation_backbone import (
    materialize_direct_task_relation_backbone,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
)
from pipeline_core.discovery.hypothesis_llm import (
    InstructorOpenAICompatibleHypothesisBackend,
)
from scripts.discovery.run_higher_order_shadow_lane import (
    _candidate_modifier_components_from_canonical_root,
)


def _write(
    path: Path,
    value: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(
        value,
        "model_dump",
    ):
        value = value.model_dump(
            mode="json"
        )

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


def _portfolio_payload(
    outcome,
):
    portfolio = (
        outcome
        .canonical_outcome
        .accepted_portfolio
    )

    if portfolio is None:
        return None

    return portfolio.model_dump(
        mode="json"
    )


def _generation_arms(
    *,
    source_context: HypothesisContext,
    contexts,
    model: str,
    base_url: str | None,
    api_key_env: str,
    parse_retries: int,
    max_repairs: int,
    telemetry_path: Path,
) -> list[dict[str, Any]]:
    if not contexts:
        return []

    backend = (
        InstructorOpenAICompatibleHypothesisBackend(
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=0.0,
            parse_retries=parse_retries,
            telemetry_path=telemetry_path,
            telemetry_context={
                "pipeline":
                    "direct_higher_order_shadow_lane",
                "stage":
                    "generation",
            },
        )
    )

    runtime = (
        DirectHigherOrderShadowHypothesisRuntime(
            backend,
            max_repairs=max_repairs,
        )
    )

    arms: list[dict[str, Any]] = []

    for index, direct_context in enumerate(
        contexts,
        start=1,
    ):
        projection = (
            materialize_direct_higher_order_hypothesis_context(
                source_context=source_context,
                direct_context=direct_context,
            )
        )

        authorization = (
            authorize_direct_higher_order_shadow_generation(
                projection=projection,
                direct_context=direct_context,
            )
        )

        outcome = runtime.run(
            projection=projection,
            direct_context=direct_context,
            authorization=authorization,
        )

        canonical = (
            outcome.canonical_outcome
        )

        arms.append(
            {
                "arm_index":
                    index,
                "direct_context_id":
                    direct_context.context_id,
                "direct_topology_id":
                    direct_context.direct_higher_order_topology_id,
                "modifier_component_id":
                    direct_context.modifier_component_id,
                "modifier_text":
                    direct_context
                    .structural_opportunity
                    .modifier_text,
                "status":
                    outcome.status,
                "shadow_contract_passed":
                    outcome.shadow_contract_passed,
                "shadow_failure_code":
                    outcome.shadow_failure_code,
                "shadow_failure_message":
                    outcome.shadow_failure_message,
                "materialization":
                    projection.materialization.model_dump(
                        mode="json"
                    ),
                "authorization":
                    authorization.model_dump(
                        mode="json"
                    ),
                "canonical_run_record":
                    canonical.run_record.model_dump(
                        mode="json"
                    ),
                "final_draft":
                    (
                        canonical.final_draft.model_dump(
                            mode="json"
                        )
                        if canonical.final_draft
                        is not None
                        else None
                    ),
                "accepted_portfolio":
                    _portfolio_payload(
                        outcome
                    ),
            }
        )

    return arms


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Shadow-only continuation from Stage-7.52 direct accepted "
            "RelationPattern recovery into direct higher-order hypothesis "
            "generation. Production Stage-8 inputs are never changed."
        )
    )
    parser.add_argument(
        "--context",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--direct-relationpattern-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--canonical-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--domain-profile",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "OPENAI_BASE_URL"
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
    )
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        choices=(0, 1),
        default=1,
    )
    args = parser.parse_args()

    if args.max_contexts < 1:
        raise ValueError(
            "--max-contexts must be >= 1"
        )

    source_context = (
        HypothesisContext.model_validate_json(
            args.context.read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        source_context.domain_profile_id
        != args.domain_profile
    ):
        raise RuntimeError(
            "source HypothesisContext domain_profile_id "
            "does not match --domain-profile"
        )

    direct_report = (
        DirectRelationPatternTaskShadowReport
        .model_validate_json(
            args.direct_relationpattern_report
            .read_text(
                encoding="utf-8"
            )
        )
    )

    if not direct_report.shadow_only:
        raise RuntimeError(
            "direct RelationPattern report must remain shadow-only"
        )

    out = args.output_dir.expanduser().resolve()
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = {
        row.candidate_id: row
        for row in direct_report.candidates
    }
    assessments = {
        row.candidate_id: row
        for row in direct_report.assessments
    }

    if (
        len(candidates)
        != len(direct_report.candidates)
        or len(assessments)
        != len(direct_report.assessments)
    ):
        raise RuntimeError(
            "duplicate direct RelationPattern candidate/assessment identity"
        )

    backbones = []
    backbone_rejections = []

    for candidate_id, candidate in candidates.items():
        assessment = assessments.get(
            candidate_id
        )

        if assessment is None:
            raise RuntimeError(
                "direct RelationPattern candidate missing assessment: "
                + candidate_id
            )

        backbone = (
            materialize_direct_task_relation_backbone(
                candidate=candidate,
                assessment=assessment,
                requested_source=direct_report.retrieval_source,
                requested_target=direct_report.retrieval_target,
            )
        )

        if backbone is None:
            backbone_rejections.append(
                {
                    "candidate_id":
                        candidate_id,
                    "task_class":
                        assessment.task_class,
                    "decision_stable":
                        assessment.decision_stable,
                    "stable_status":
                        assessment.stable_status,
                    "stable_role":
                        assessment.stable_role,
                    "reason":
                        "NOT_DIRECT_BACKBONE_ELIGIBLE",
                }
            )
            continue

        backbones.append(
            backbone
        )

    backbone_path = (
        out
        / "direct_backbones.json"
    )
    _write(
        backbone_path,
        {
            "schema_version":
                "direct-task-relation-backbone-shadow-report-v1",
            "source_report":
                str(
                    args.direct_relationpattern_report
                ),
            "requested_source":
                direct_report.retrieval_source,
            "requested_target":
                direct_report.retrieval_target,
            "input_candidate_count":
                len(candidates),
            "materialized_backbone_count":
                len(backbones),
            "rejection_count":
                len(backbone_rejections),
            "backbones": [
                row.model_dump(
                    mode="json"
                )
                for row in backbones
            ],
            "rejections":
                backbone_rejections,
            "authority": {
                "shadow_only":
                    True,
                "endpoint_equivalence_authorized":
                    False,
                "scientific_identity_asserted":
                    False,
                "positive_premise_authority_created":
                    False,
                "novelty_authority_created":
                    False,
                "production_selection_authority":
                    False,
            },
        },
    )

    (
        candidate_components,
        candidate_replay,
    ) = (
        _candidate_modifier_components_from_canonical_root(
            canonical_root=args.canonical_root,
            domain_profile=args.domain_profile,
        )
    )

    if (
        candidate_replay.get("status")
        == "CANDIDATE_MODIFIER_REPLAY_COMPLETE"
        and not bool(
            candidate_replay.get(
                "replay_authoritative_payload_matches_canonical",
                False,
            )
        )
    ):
        raise RuntimeError(
            "candidate modifier replay changed the authoritative "
            "DiscoveryBundle payload"
        )

    replay_path = (
        out
        / "candidate_modifier_replay_audit.json"
    )
    _write(
        replay_path,
        candidate_replay,
    )

    modifier_screen = (
        screen_direct_backbone_candidate_modifiers(
            backbones=backbones,
            components=candidate_components,
        )
    )

    modifier_path = (
        out
        / "modifier_eligibility_audit.json"
    )
    _write(
        modifier_path,
        modifier_screen.audit,
    )

    topologies = (
        compose_direct_higher_order_topologies(
            backbones=backbones,
            modifier_components=candidate_components,
            eligible_modifier_records=(
                modifier_screen
                .eligible_records
            ),
            audit_source=str(
                modifier_path
            ),
        )
    )

    topology_path = (
        out
        / "topologies.json"
    )
    _write(
        topology_path,
        {
            "schema_version":
                "direct-higher-order-topology-shadow-report-v1",
            "direct_backbone_source":
                str(backbone_path),
            "modifier_audit_source":
                str(modifier_path),
            "canonical_root":
                str(args.canonical_root),
            "direct_backbone_count":
                len(backbones),
            "candidate_modifier_component_count":
                len(candidate_components),
            "eligible_modifier_record_count":
                len(
                    modifier_screen
                    .eligible_records
                ),
            "topology_count":
                len(topologies),
            "candidate_replay":
                candidate_replay,
            "topologies": [
                row.model_dump(
                    mode="json"
                )
                for row in topologies
            ],
            "authority": {
                "shadow_only":
                    True,
                "topology_is_interaction_evidence":
                    False,
                "interaction_claim_authorized":
                    False,
                "endpoint_equivalence_authorized":
                    False,
                "scientific_identity_asserted":
                    False,
                "novelty_authority_created":
                    False,
                "positive_premise_authority_created":
                    False,
                "production_selection_authority":
                    False,
            },
        },
    )

    contexts = (
        build_direct_higher_order_synthesis_contexts(
            topologies=topologies
        )
    )

    selected_contexts = tuple(
        contexts[
            : args.max_contexts
        ]
    )

    context_path = (
        out
        / "synthesis_contexts.json"
    )

    _write(
        context_path,
        {
            "schema_version":
                "direct-higher-order-synthesis-context-report-v1",
            "topology_report":
                str(topology_path),
            "topology_count":
                len(topologies),
            "context_count":
                len(contexts),
            "selected_context_count":
                len(selected_contexts),
            "selection_policy":
                "deterministic_topology_order_prefix",
            "contexts": [
                row.model_dump(
                    mode="json"
                )
                for row in contexts
            ],
            "selected_context_ids": [
                row.context_id
                for row in selected_contexts
            ],
            "rendered_prompts": [
                render_direct_higher_order_shadow_prompt(
                    row
                )
                for row in selected_contexts
            ],
            "authority": {
                "shadow_only":
                    True,
                "prompt_rendering_authorized":
                    True,
                "llm_call_authorized":
                    False,
                "interaction_claim_authorized":
                    False,
                "novelty_authority_created":
                    False,
                "positive_premise_authority_created":
                    False,
                "production_selection_authority":
                    False,
            },
        },
    )

    arms = _generation_arms(
        source_context=source_context,
        contexts=selected_contexts,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        parse_retries=args.parse_retries,
        max_repairs=args.max_repairs,
        telemetry_path=(
            out
            / "generation.telemetry.jsonl"
        ),
    )

    if not backbones:
        status = "NO_DIRECT_BACKBONE"
    elif not candidate_components:
        status = "NO_CANDIDATE_MODIFIER_COMPONENT"
    elif not (
        modifier_screen.eligible_records
    ):
        status = "NO_ELIGIBLE_MODIFIER"
    elif not topologies:
        status = "NO_DIRECT_HIGHER_ORDER_TOPOLOGY"
    elif not selected_contexts:
        status = "NO_SYNTHESIS_CONTEXT"
    elif any(
        row["status"] == "proposed"
        for row in arms
    ):
        status = "PROPOSED"
    else:
        status = "EVALUATED_NO_PROPOSAL"

    report = {
        "schema_version":
            "direct-higher-order-shadow-lane-generation-v1",
        "status":
            status,
        "source_context":
            str(args.context),
        "source_context_id":
            source_context.context_id,
        "source_context_sha256":
            source_context.context_sha256,
        "direct_relationpattern_report":
            str(
                args.direct_relationpattern_report
            ),
        "direct_relationpattern_report_id":
            direct_report.report_id,
        "direct_relationpattern_report_sha256":
            direct_report.report_sha256,
        "canonical_root":
            str(args.canonical_root),
        "domain_profile_id":
            args.domain_profile,
        "requested_source":
            direct_report.retrieval_source,
        "requested_target":
            direct_report.retrieval_target,
        "model":
            args.model,
        "direct_backbone_artifact":
            str(backbone_path),
        "candidate_modifier_replay_artifact":
            str(replay_path),
        "modifier_eligibility_artifact":
            str(modifier_path),
        "topology_artifact":
            str(topology_path),
        "synthesis_context_artifact":
            str(context_path),
        "direct_backbone_count":
            len(backbones),
        "candidate_modifier_component_count":
            len(candidate_components),
        "eligible_modifier_count":
            len(
                modifier_screen
                .eligible_records
            ),
        "topology_count":
            len(topologies),
        "context_count":
            len(contexts),
        "selected_context_count":
            len(selected_contexts),
        "proposed_count":
            sum(
                row["status"]
                == "proposed"
                for row in arms
            ),
        "abstained_count":
            sum(
                row["status"]
                == "abstained"
                for row in arms
            ),
        "canonical_rejected_count":
            sum(
                row["status"]
                == "canonical_rejected"
                for row in arms
            ),
        "shadow_contract_rejected_count":
            sum(
                row["status"]
                == "shadow_contract_rejected"
                for row in arms
            ),
        "arms":
            arms,
        "authority": {
            "shadow_only":
                True,
            "direct_relationpattern_task_authority_reused":
                True,
            "candidate_modifier_replay_selection_changed":
                False,
            "modifier_screen_is_interaction_evidence":
                False,
            "topology_is_interaction_evidence":
                False,
            "restricted_composition_relations_as_positive_premise":
                False,
            "external_novelty_review_performed":
                False,
            "conceptual_knownness_performed":
                False,
            "n10_review_performed":
                False,
            "task_conditioned_axis_plan_changed":
                False,
            "dual_context_changed":
                False,
            "stage8_input_changed":
                False,
            "production_selection_changed":
                False,
            "canonical_graph_mutated":
                False,
        },
    }

    report_path = (
        out
        / "generation_report.json"
    )
    _write(
        report_path,
        report,
    )

    print(
        "=== DIRECT HIGHER-ORDER SHADOW LANE ==="
    )
    print(
        "status:",
        status,
    )
    print(
        "direct backbones:",
        len(backbones),
    )
    print(
        "candidate modifiers:",
        len(candidate_components),
    )
    print(
        "eligible modifiers:",
        len(
            modifier_screen
            .eligible_records
        ),
    )
    print(
        "topologies:",
        len(topologies),
    )
    print(
        "selected contexts:",
        len(selected_contexts),
    )
    print(
        "proposed:",
        report["proposed_count"],
    )

    for arm in arms:
        print(
            "\narm",
            arm["arm_index"],
            "| modifier=",
            arm["modifier_text"],
            "| status=",
            arm["status"],
        )

        portfolio = arm.get(
            "accepted_portfolio"
        )
        if portfolio is None:
            continue

        for card in portfolio.get(
            "hypotheses",
            [],
        ):
            print(
                " hypothesis:",
                card[
                    "hypothesis_statement"
                ],
            )

    print(
        "\nShadow only; Stage-8 production input unchanged."
    )
    print(
        "artifact:",
        report_path,
    )
    print(
        "DIRECT_HO_SHADOW_COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
